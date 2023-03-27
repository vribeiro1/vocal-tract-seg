import argparse
import funcy
import numpy as np
import os
import pandas as pd
import yaml

from glob import glob
from multiprocessing import Pool
from scipy.ndimage import rotate
from skimage.metrics import structural_similarity

from track_incisors import *


def optimize(search_arr, rotations, metric, how=max):
    search_H, search_W = search_arr.shape
    _, rot_arr = list(rotations.items())[0]
    mask_H, mask_W = rot_arr.shape

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
            for angle, rot_arr in rotations.items():
                val = metric(region_arr, rot_arr)
                item = (shift_x, shift_y, angle, val)
                optim_data.append(item)

    best_shift_x, best_shift_y, best_angle, val = how(optim_data, key=lambda tup: tup[-1])

    xc = best_shift_x + mask_W // 2
    yc = best_shift_y + mask_H // 2

    optim = {
        "xc": xc,
        "yc": yc,
        "angle": best_angle,
        "value": val
    }

    return optim


def center_rotate(arr, deg, center):
    xc, yc = center

    rad = np.deg2rad(deg)

    arr_ = arr.copy()
    arr_[:, 0] = arr_[:, 0] - xc
    arr_[:, 1] = arr_[:, 1] - yc

    rot_mtx = np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad), np.cos(rad)]
    ])

    rot_arr = np.matmul(arr_, rot_mtx)
    rot_arr[:, 0] = rot_arr[:, 0] + xc
    rot_arr[:, 1] = rot_arr[:, 1] + yc

    return rot_arr


def draw_lower_incisor(e0, angle, control_params):
    keypoints = draw_incisor(e0, control_params)
    keypoints = center_rotate(keypoints, angle, e0)
    return keypoints


def create_rotated_masks(mask, min_deg, max_deg, step, margin=3, include_zero=True):
    angles = list(range(min_deg, max_deg, step))
    if include_zero and 0 not in angles:
        angles.append(0)
        angles = sorted(angles)

    rm_margin = margin - 1  # Keep 1-pixel margin
    rotations = {}
    for angle in angles:
        rot_mask = rotate(mask, angle, reshape=False)
        rotations[angle] = rot_mask[rm_margin:-rm_margin, rm_margin:-rm_margin]

    return rotations


def compute_references(
    datadir,
    database,
    subject,
    sequence,
    search_space,
    save_to=None
):
    ref_mask = get_sequence_reference_mask(database, datadir, subject, "lower-incisor")
    rotations = create_rotated_masks(ref_mask, min_deg=-10, max_deg=21, step=1)
    angles = list(rotations.keys())
    metric = structural_similarity

    x0_search = search_space["x0"]
    y0_search = search_space["y0"]
    x1_search = search_space["x1"]
    y1_search = search_space["y1"]

    data = []
    dcm_filepaths = sorted(glob(os.path.join(datadir, subject, sequence, "NPY_MR", "*.npy")))
    curr_angle = 0
    for other_filepath in dcm_filepaths:
        other_arr = load_input_image(other_filepath)
        search_arr = other_arr[y0_search:y1_search, x0_search:x1_search]

        # Restrict the angles search space since the angle cannot vary abruptly
        idx = angles.index(curr_angle)
        range_min = max(0, idx - 2)
        range_max = min(len(angles), idx + 2)
        use_angles = angles[range_min:range_max + 1]
        use_rotations = {angle: rotations[angle] for angle in use_angles}

        optim = optimize(search_arr, use_rotations, metric)
        angle = optim["angle"]
        xc = x0_search + optim["xc"]
        yc = y0_search + optim["yc"]

        rel_filepath = other_filepath.replace(os.path.dirname(DATA_DIR), "").strip("/")
        item = {
            "subject": subject,
            "sequence": sequence,
            "filepath": rel_filepath,
            "frame": int(os.path.basename(other_filepath).split(".")[0]),
            "x0": xc,
            "y0": yc,
            "angle": angle
        }
        data.append(item)

    x0s = funcy.lmap(lambda d: d["x0"], data)
    y0s = funcy.lmap(lambda d: d["y0"], data)
    angles = funcy.lmap(lambda d: d["angle"], data)

    avg_x0s = moving_average(x0s, window_size=5)
    avg_y0s = moving_average(y0s, window_size=5)
    avg_angles = moving_average(angles, window_size=3)

    [d.update({
        "avg_x0": avg_x0s[i],
        "avg_y0": avg_y0s[i],
        "avg_angle": avg_angles[i]
    }) for i, d in enumerate(data)]

    df = pd.DataFrame(data)
    if save_to is not None:
        df.to_csv(save_to, index=False)

    return df


def process_sequence(item):
    subject, sequence, cfg = item
    print(f"Processing {subject}-{sequence}")

    overwrite = cfg.get("overwrite", False)
    datadir = cfg["datadir"]
    database = cfg["database"]
    save_to = cfg["save_to"]
    search_space = cfg["search_space"]
    control_params = cfg["control_params"]

    if not os.path.exists(save_to):
            os.makedirs(save_to)
    csv_filepath = os.path.join(save_to, f"optimization_lower-incisor_{subject}-{sequence}.csv")

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

    for i, row in df.iterrows():
        frame = "%04d" % row["frame"]
        x0 = row["avg_x0"]
        y0 = row["avg_y0"]
        angle = row["avg_angle"]
        upper_incisor = draw_lower_incisor((x0, y0), angle, control_params)

        save_dir = os.path.join(save_to, subject, sequence, "inference_contours")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_filepath = os.path.join(save_dir, f"{frame}_lower-incisor.npy")
        np.save(save_filepath, upper_incisor)

    print(f"Finished processing {subject}-{sequence}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_filepath")
    args = parser.parse_args()

    with open(args.config_filepath) as f:
        cfg = yaml.safe_load(f)

    datadir = cfg["datadir"]
    sequences = sequences_from_dict(datadir, cfg["sequences"])
    items = [(subject, sequence, cfg) for subject, sequence in sequences]
    with Pool(10) as pool:
        pool.map(process_sequence, items)
