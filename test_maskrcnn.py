import pdb

import argparse
import funcy
import json
import numpy as np
import os
import torch
import yaml

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn

from connect_points import (connect_with_active_contours,
                            connect_points_graph_based,
                            draw_contour,
                            evaluate_model)
from dataset import VocalTractMaskRCNNDataset
from evaluation import run_evaluation, run_inference
from helpers import set_seeds
from settings import *


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(cfg["classes"])

    dataset = VocalTractMaskRCNNDataset(
        cfg["datadir"],
        cfg["sequences"],
        cfg["classes"],
        size=cfg["size"],
        mode=cfg["mode"]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        worker_init_fn=set_seeds,
        collate_fn=dataset.collate_fn
    )

    model = maskrcnn_resnet50_fpn(pretrained=True)
    if cfg["state_dict_fpath"] is not None:
        state_dict = torch.load(cfg["state_dict_fpath"], map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)

    if not os.path.exists(cfg["save_to"]):
        os.mkdir(cfg["save_to"])

    class_map = {i: c for c, i in dataset.classes_dict.items()}
    outputs = run_inference(
        epoch=0,
        model=model,
        dataloader=dataloader,
        device=device,
        class_map=class_map,
        outputs_dir=cfg["save_to"]
    )

    results = run_evaluation(
        outputs,
        0.7,
        dataset.classes,
        cfg["save_to"],
        lambda fp: dataset.resize(dataset.read_dcm(fp))
    )

    results_filepath = os.path.join(cfg["save_to"], "test_results.json")
    with open(results_filepath, "w") as f:
        json.dump(results, f)

    llip_mean, llip_std = results[LOWER_LIP]
    soft_palate_mean, soft_palate_std = results[SOFT_PALATE]
    tongue_mean, tongue_std = results[TONGUE]
    ulip_mean, ulip_std = results[UPPER_LIP]

    print(f"""
Results:

Lower lip: MSD = {llip_mean} +- {llip_std}
Soft palate: MSD = {soft_palate_mean} +- {soft_palate_std}
Tongue: MSD = {tongue_mean} +- {tongue_std}
Upper lip: MSD = {ulip_mean} +- {ulip_std}
""")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="cfg_filepath")
    args = parser.parse_args()

    with open(args.cfg_filepath) as f:
        cfg = yaml.safe_load(f.read())

    main(cfg)
