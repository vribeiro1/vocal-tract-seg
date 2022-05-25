import pdb

import argparse
import json
import os
import torch
import yaml

from torch.utils.data import DataLoader
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn

from dataset import VocalTractMaskRCNNDataset
from evaluation import run_evaluation, test_runners
from helpers import set_seeds
from settings import *


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cfg["model_name"]
    image_folder = cfg["image_folder"]
    image_ext = cfg["image_ext"]

    dataset = VocalTractMaskRCNNDataset(
        cfg["datadir"],
        cfg["sequences"],
        cfg["classes"],
        size=cfg["size"],
        mode=cfg["mode"],
        image_folder=image_folder,
        image_ext=image_ext,
        include_bkg=(model_name == "maskrcnn")
    )

    collate_fn = getattr(VocalTractMaskRCNNDataset, f"{model_name}_collate_fn")
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        worker_init_fn=set_seeds,
        collate_fn=collate_fn
    )

    model = maskrcnn_resnet50_fpn(pretrained=True)
    if cfg["state_dict_fpath"] is not None:
        state_dict = torch.load(cfg["state_dict_fpath"], map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)

    if not os.path.exists(cfg["save_to"]):
        os.mkdir(cfg["save_to"])

    class_map = {i: c for c, i in dataset.classes_dict.items()}
    run_test = test_runners[model_name]
    outputs = run_test(
        epoch=0,
        model=model,
        dataloader=dataloader,
        device=device,
        class_map=class_map,
        outputs_dir=os.path.join(cfg["save_to"], "test_outputs")
    )

    read_fn = getattr(VocalTractMaskRCNNDataset, f"read_{image_ext}")
    results = run_evaluation(
        outputs,
        dataset.classes,
        os.path.join(cfg["save_to"], "test_outputs"),
        lambda fp: dataset.resize(read_fn(fp))
    )

    results_filepath = os.path.join(cfg["save_to"], "test_results.json")
    with open(results_filepath, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="cfg_filepath")
    args = parser.parse_args()

    with open(args.cfg_filepath) as f:
        cfg = yaml.safe_load(f.read())

    main(cfg)
