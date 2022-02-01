import funcy
import numpy as np
import os
import yaml

from argparse import ArgumentParser
from tqdm import tqdm
from vt_tools.bs_regularization import regularize_Bsplines


def track_arytenoid_muscle(pharynx, vocal_folds):
    x0_vfolds = vocal_folds[:, 0].min()
    y0_vfolds = vocal_folds[:, 1].min()
    x1_vfolds = vocal_folds[:, 0].max()
    y1_vfolds = vocal_folds[:, 1].max()

    x0_pharynx = pharynx[:, 0].min()
    y0_pharynx = pharynx[:, 1].min()
    x1_pharynx = pharynx[:, 0].max()
    y1_pharynx = pharynx[:, 1].max()

    ext0_x, ext0_y = x1_pharynx, y1_pharynx
    ext1_x, ext1_y = x1_vfolds, y0_vfolds

    p1_x = ext0_x - 0.2 * np.abs(ext1_x - ext0_x)
    p1_y = ext0_y + 0.3 * np.abs(ext1_y - ext0_y)

    p2_x = ext0_x - 0.95 * np.abs(ext1_x - ext0_x)
    p2_y = ext0_y + 0.2 * np.abs(ext1_y - ext0_y)

    reg_x, reg_y = regularize_Bsplines(np.array([
        (ext0_x, ext0_y),
        (p1_x, p1_y),
        (p2_x, p2_y),
        (ext1_x, ext1_y)
    ]), 3)

    amuscle_tmp = np.array([reg_x, reg_y]).T
    amuscle_list = funcy.lfilter(lambda p: p[1] < ext1_y, amuscle_tmp)

    reg_x, reg_y = regularize_Bsplines(amuscle_list, 3)
    amuscle_arr = np.array([reg_x, reg_y]).T

    return amuscle_arr


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", dest="config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f.read())

    datadir = cfg["datadir"]
    subjects_sequences = cfg["subjects_sequences"]

    for subject, sequences in subjects_sequences.items():
        for sequence in sequences:
            sequence_dir = os.path.join(datadir, subject, sequence)
            dcm_list = filter(lambda s: s.endswith(".dcm"), os.listdir(os.path.join(sequence_dir, "dicoms")))
            inumber_list = funcy.lmap(lambda s: s.split(".")[0], dcm_list)

            for inumber in tqdm(inumber_list, desc=f"{subject} {sequence}"):
                pharynx_filepath = os.path.join(sequence_dir, "inference_contours", f"{inumber}_pharynx.npy")
                pharynx_arr = np.load(pharynx_filepath)

                vfolds_filepath = os.path.join(sequence_dir, "inference_contours", f"{inumber}_vocal-folds.npy")
                vfolds_arr = np.load(vfolds_filepath)

                arytenoid_arr = track_arytenoid_muscle(pharynx_arr, vfolds_arr)
                arytenoid_filepath = os.path.join(sequence_dir, "inference_contours", f"{inumber}_arytenoid-muscle.npy")
                np.save(arytenoid_filepath, arytenoid_arr)
