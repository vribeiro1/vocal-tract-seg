import argparse
import funcy
import numpy as np
import os
import pandas as pd
import yaml

from functools import partial
from glob import glob
from multiprocessing import Pool
from skimage.metrics import structural_similarity
from tqdm import tqdm

from track_incisors import *


def optimize(search_arr, mask, metric, how=max):
    search_H, search_W = search_arr.shape
    mask_H, mask_W = mask.shape

    x_shifts = list(range(0, search_W - mask_W))
    y_shifts = list(range(0, search_H - mask_H))

    optim_data = []
    for shift_x in x_shifts:
        for shift_y in y_shifts:
            x_start = shift_x
            x_end = shift_x + mask_W

            y_start = shift_y
            y_end = shift_y + mask_H

            region_arr = search_arr[y_start:y_end, x_start:x_end]
            val = metric(region_arr, mask)
            item = (shift_x, shift_y, val)
            optim_data.append(item)

    best_shift_x, best_shift_y, val = how(optim_data, key=lambda tup: tup[-1])

    xc = best_shift_x + mask_W // 2
    yc = best_shift_y + mask_H // 2

    optim = {
        "xc": xc,
        "yc": yc,
        "value": val
    }

    return optim


def process_frame(
    filepath,
    cropped_mask,
    search_space,
    metric
):
    x0_search = search_space["x0"]
    y0_search = search_space["y0"]
    x1_search = search_space["x1"]
    y1_search = search_space["y1"]

    other_arr = load_input_image(filepath)
    search_arr = other_arr[y0_search:y1_search, x0_search:x1_search]

    optim = optimize(search_arr, cropped_mask, metric)

    xc = x0_search + optim["xc"]
    yc = y0_search + optim["yc"]

    rel_filepath = filepath.replace(os.path.dirname(datadir), "").strip("/")
    item = {
        "subject": subject,
        "sequence": sequence,
        "frame": int(os.path.basename(filepath).split(".")[0]),
        "filepath": rel_filepath,
        "x0": xc,
        "y0": yc
    }

    return item


def compute_references(
    datadir,
    database,
    subject,
    sequence,
    search_space,
    save_to=None,
):
    ref_mask = get_sequence_reference_mask(database, datadir, subject, "upper-incisor")
    metric = structural_similarity

    process_other = partial(
        process_frame,
        cropped_mask=ref_mask,
        search_space=search_space,
        metric=metric
    )

    dcm_filepaths = sorted(glob(os.path.join(datadir, subject, sequence, "NPY_MR", "*.npy")))
    progress_bar = tqdm(dcm_filepaths, desc=f"{subject} {sequence}")
    with Pool(10) as pool:
        data = pool.map(process_other, progress_bar)

    x0s = funcy.lmap(lambda d: d["x0"], data)
    avg_x0s = moving_average(x0s, window_size=10)
    y0s = funcy.lmap(lambda d: d["y0"], data)
    avg_y0s = moving_average(y0s, window_size=10)

    [d.update({
        "avg_x0": avg_x0s[i],
        "avg_y0": avg_y0s[i],
    }) for i, d in enumerate(data)]

    df = pd.DataFrame(data)
    if save_to is not None:
        df.to_csv(save_to, index=False)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_filepath")
    args = parser.parse_args()

    with open(args.config_filepath) as f:
        cfg = yaml.safe_load(f)

    overwrite = cfg.get("overwrite", False)
    datadir = cfg["datadir"]
    database = cfg["database"]
    save_to = cfg["save_to"]
    search_space = cfg["search_space"]
    control_params = cfg["control_params"]
    sequences = sequences_from_dict(datadir, cfg["sequences"])

    for subject, sequence in sequences:
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        csv_filepath = os.path.join(save_to, f"optimization_upper-incisor_{subject}-{sequence}.csv")

        if os.path.isfile(csv_filepath) and not overwrite:
            df = pd.read_csv(csv_filepath)
        else:
            df = compute_references(
                datadir,
                database,
                subject,
                sequence,
                search_space,
                csv_filepath
            )

        for i, row in tqdm(df.iterrows(), desc=f"{subject} {sequence} Drawing upper incisor", total=len(df)):
            frame = "%04d" % row["frame"]
            x0 = row["avg_x0"]
            y0 = row["avg_y0"]
            upper_incisor = draw_incisor((x0, y0), control_params)

            save_dir = os.path.join(save_to, subject, sequence, "inference_contours")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_filepath = os.path.join(save_dir, f"{frame}_upper-incisor.npy")
            np.save(save_filepath, upper_incisor)
