import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import ujson
import yaml

from roifile import roiread
from tqdm import tqdm
from vt_tools import COLORS
from vt_tools.bs_regularization import regularize_Bsplines

from evaluation import load_articulator_array
from settings import DatasetConfig


ANONYM_SUBJECT_MAP = {
    "1612": "S1",
    "1618": "S2",
    "1635": "S3",
    "1638": "S4",
    "1640": "S5",
    "1659": "S6",
    "1662": "S7",
    "1775": "S8"
}


def consolidate_results_dataframes(results_filepaths):
    df_results = None
    for result_filepath in results_filepaths:
        experiment_dir = os.path.dirname(result_filepath)
        experiment = os.path.basename(experiment_dir)
        df = pd.read_csv(result_filepath)

        df["subject"] = df["subject"].astype(str)
        df["frame"] = df["frame"].astype(str)
        df["frame"] = df["frame"].str.rjust(4, "0")
        df["p2cp_mean"] = df["p2cp_mean"] * DatasetConfig.PIXEL_SPACING
        df["p2cp_rms"] = df["p2cp_rms"] * DatasetConfig.PIXEL_SPACING
        df["experiment"] = experiment

        config_filepath = os.path.join(experiment_dir, "config.json")
        with open(config_filepath) as f:
            config = ujson.load(f)
        test_subjects = list(config["test_sequences"].keys())
        _, left_out_subject = test_subjects[0].split("/")
        df["left_out_subject"] = left_out_subject

        if df_results is None:
            df_results = df
        else:
            df_results = pd.concat([df_results, df])

    df_results["anonym_subject"] = df_results["subject"].map(ANONYM_SUBJECT_MAP)
    df_results["left_out_anonym_subject"] = df_results["left_out_subject"].map(ANONYM_SUBJECT_MAP)

    return df_results


def pivot_table(df, values, index, columns):
    df_pivot = pd.pivot_table(
        df,
        values=values,
        index=index,
        columns=columns
    )
    df_pivot = df_pivot.reorder_levels([1, 0], axis=1)

    return df_pivot


def plot_frame(
    img_filepath,
    df_frame,
    save_filepaths
):
    lw = 5
    alpha = 0.7
    fontsize = 22

    if isinstance(save_filepaths, str):
        save_filepaths = [save_filepaths]

    plt.figure(figsize=(10, 10))

    img_array = np.load(img_filepath)
    plt.imshow(img_array, cmap="gray")

    for i, row in df_frame.sort_values("pred_class").reset_index(drop=True).iterrows():
        target_filepath = row["target_filepath"]
        pred_filepath = row["pred_filepath"]
        articulator = row["pred_class"]

        target_array = roiread(target_filepath).coordinates()
        reg_x, reg_y = regularize_Bsplines(target_array, degree=2)
        reg_target_array = np.array([reg_x, reg_y]).T
        plt.plot(*reg_target_array.T, "--", color=COLORS[articulator], lw=lw, alpha=alpha)

        if not pd.isna(pred_filepath):
            pred_array = load_articulator_array(pred_filepath)
            reg_x, reg_y = regularize_Bsplines(pred_array, degree=2)
            reg_pred_array = np.array([reg_x, reg_y]).T
            plt.plot(*reg_pred_array.T, "-", color=COLORS[articulator], lw=lw, alpha=alpha)

    anonym_subject = df_frame["anonym_subject"].iloc[0]
    left_out_subject = df_frame["left_out_anonym_subject"].iloc[0]
    plt.text(5, 125, f"Target subject: {anonym_subject}", color="yellow", fontsize=fontsize)
    plt.text(5, 130, f"Left-out subject: {left_out_subject}", color="yellow", fontsize=fontsize)

    plt.axis("off")
    plt.tight_layout()

    for save_filepath in save_filepaths:
        plt.savefig(save_filepath)
    plt.close()


def main(datadir, results_filepaths, save_dir):
    df = consolidate_results_dataframes(results_filepaths)

    df_grouped = df.groupby([
        "anonym_subject",
        "pred_class"
    ]).agg({
        "p2cp_mean": ["mean", "std"],
        "p2cp_rms": ["mean", "std"],
        "jaccard_index": ["mean", "std"]
    })
    df_grouped.columns = df_grouped.columns.map('{0[0]}.{0[1]}'.format)
    df_grouped = df_grouped.reset_index()

    tabular_dir = os.path.join(save_dir, "tabular")
    figures_dir = os.path.join(save_dir, "figures")
    dirs_ = [tabular_dir, figures_dir]
    for dir_ in dirs_:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

    save_filepath = os.path.join(tabular_dir, "p2cp_mean.csv")
    df_p2cp_mean = pivot_table(
        df=df_grouped,
        values=["p2cp_mean.mean", "p2cp_mean.std"],
        index=["anonym_subject"],
        columns=["pred_class"]
    )
    df_p2cp_mean.to_csv(save_filepath, index=True)

    save_filepath = os.path.join(tabular_dir, "p2cp_rms.csv")
    df_p2cp_rms = pivot_table(
        df=df_grouped,
        values=["p2cp_rms.mean", "p2cp_rms.std"],
        index=["anonym_subject"],
        columns=["pred_class"]
    )
    df_p2cp_rms.to_csv(save_filepath, index=True)

    save_filepath = os.path.join(tabular_dir, "jaccard.csv")
    df_jacc = pivot_table(
        df=df_grouped,
        values=["jaccard_index.mean", "jaccard_index.std"],
        index=["anonym_subject"],
        columns=["pred_class"]
    )
    df_jacc.to_csv(save_filepath, index=True)

    grouped = df.groupby(["subject", "sequence", "frame"])
    total = len(df[["subject", "sequence", "frame"]].drop_duplicates())
    progress_bar = tqdm(grouped, desc="Plotting results", total=total)
    for (subject, sequence, frame), df_frame in progress_bar:
        img_filepath = os.path.join(datadir, subject, sequence, "NPY_MR", f"{frame}.npy")
        save_filepaths = [
            os.path.join(figures_dir, f"{subject}_{sequence}_{frame}.pdf"),
            os.path.join(figures_dir, f"{subject}_{sequence}_{frame}.png"),
        ]
        plot_frame(img_filepath, df_frame, save_filepaths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="cfg_filepath")
    args = parser.parse_args()

    with open(args.cfg_filepath) as f:
        cfg = yaml.safe_load(f.read())

    main(**cfg)