import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

from consolidate_tracking_results import ANONYM_SUBJECT_MAP, consolidate_results_dataframes
from dataset import VocalTractMaskRCNNDataset

PINK = np.array([255, 0, 85, 255]) / 255
BLUE = np.array([0, 139, 231, 255]) / 255

cmap = plt.get_cmap("hsv")
num_subjects = len(ANONYM_SUBJECT_MAP)
SUBJECTS_COLORS = {
    anonym_subject: cmap(i / num_subjects)
    for i, (_, anonym_subject) in enumerate(ANONYM_SUBJECT_MAP.items())
}


def plot_fine_tuning(articulator, df, first_axis, second_axis=None, save_filepaths=None):
    if save_filepaths is None:
        save_filepaths = []
    elif isinstance(save_filepaths, str):
        save_filepaths = [save_filepaths]

    fontsize = 28
    labelsize = 22
    lw = 2
    alpha = 0.2
    alpha_aux = 1.0

    df_grouped = df.groupby("fine_tuning_size").agg("mean", "std").reset_index(drop=False)
    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.gca()
    ax1.set_ylim([0, 2.5])
    ax1.set_ylabel(first_axis.replace("_", " "), fontsize=fontsize)
    ax1.tick_params(axis="both", which="major", labelsize=labelsize)

    ax1.plot(
        df_grouped["fine_tuning_size"],
        df_grouped[f"{first_axis}.mean"] - df_grouped[f"{first_axis}.std"],
        color=PINK,
        alpha=alpha
    )
    ax1 .plot(
        df_grouped["fine_tuning_size"],
        df_grouped[f"{first_axis}.mean"],
        color=PINK,
        lw=lw
    )
    ax1.plot(
        df_grouped["fine_tuning_size"],
        df_grouped[f"{first_axis}.mean"] + df_grouped[f"{first_axis}.std"],
        color=PINK,
        alpha=alpha
    )
    ax1.fill_between(
        df_grouped["fine_tuning_size"],
        df_grouped[f"{first_axis}.mean"] - df_grouped[f"{first_axis}.std"],
        df_grouped[f"{first_axis}.mean"] + df_grouped[f"{first_axis}.std"],
        color=PINK,
        alpha=alpha
    )

    for subject, group in df.groupby("anonym_subject"):
        ax1.plot(
            group["fine_tuning_size"],
            group[f"{first_axis}.mean"],
            linestyle="dotted",
            color=SUBJECTS_COLORS[subject],
            alpha=alpha_aux
        )

    if second_axis is not None:
        ax2 = ax1.twinx()
        ax2.set_ylim([0, 1])
        ax2.set_ylabel(second_axis.replace("_", " "), fontsize=fontsize)
        ax2.tick_params(axis="both", which="major", labelsize=labelsize)

        ax2.plot(
            df_grouped["fine_tuning_size"],
            df_grouped[f"{second_axis}.mean"] - df_grouped[f"{second_axis}.std"],
            color=BLUE,
            alpha=alpha
        )
        ax2.plot(
            df_grouped["fine_tuning_size"],
            df_grouped[f"{second_axis}.mean"],
            color=BLUE,
            lw=lw
        )
        ax2.plot(
            df_grouped["fine_tuning_size"],
            df_grouped[f"{second_axis}.mean"] + df_grouped[f"{second_axis}.std"],
            color=BLUE,
            alpha=alpha
        )
        ax2.fill_between(
            df_grouped["fine_tuning_size"],
            df_grouped[f"{second_axis}.mean"] - df_grouped[f"{second_axis}.std"],
            df_grouped[f"{second_axis}.mean"] + df_grouped[f"{second_axis}.std"],
            color=BLUE,
            alpha=alpha
        )

        for subject, group in df.groupby("anonym_subject"):
            ax2.plot(
                group["fine_tuning_size"],
                group[f"{second_axis}.mean"],
                linestyle="dashdot",
                color=SUBJECTS_COLORS[subject],
                alpha=alpha_aux
            )

    ax1.set_xlabel("fine tuning size", fontsize=fontsize)
    plt.tight_layout()

    for filepath in save_filepaths:
        plt.savefig(filepath)
    plt.close()


def main(datadir, results_filepaths, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    groupby_cols = ["anonym_subject", "pred_class", "fine_tuning_size"]
    df = consolidate_results_dataframes(results_filepaths)
    df_grouped = df.groupby(groupby_cols).agg({
        "p2cp_mean": ["mean", "std"],
        "p2cp_rms": ["mean", "std"],
        "jaccard_index": ["mean", "std"],
    })
    df_grouped.columns = df_grouped.columns.map('{0[0]}.{0[1]}'.format)
    df_grouped = df_grouped.reset_index()

    tabular_dir = os.path.join(save_dir, "tabular")
    figures_dir = os.path.join(save_dir, "figures")
    dirs_ = [tabular_dir, figures_dir]
    for dir_ in dirs_:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

    df.to_csv(os.path.join(tabular_dir, "raw.csv"))

    for articulator in df.pred_class.unique():
        save_filepaths = [
            os.path.join(figures_dir, f"{articulator}.png"),
            os.path.join(figures_dir, f"{articulator}.pdf"),
        ]

        df_articulator = df_grouped[df_grouped.pred_class == articulator]
        closed_articulator = articulator in VocalTractMaskRCNNDataset.closed_articulators
        plot_fine_tuning(
            articulator,
            df_articulator,
            first_axis="p2cp_rms",
            second_axis="jaccard_index" if closed_articulator else None,
            save_filepaths=save_filepaths,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="cfg_filepath")
    args = parser.parse_args()

    with open(args.cfg_filepath) as f:
        cfg = yaml.safe_load(f.read())

    main(**cfg)
