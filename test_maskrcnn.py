####################################################################################################
#
# Test a convolutional neural network to perform border segmentation of vocal tract articulators.
#
####################################################################################################

import argparse
import funcy
import logging
import os
import pandas as pd
import torch
import ujson
import yaml

from torch.utils.data import DataLoader
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn

from dataset import VocalTractMaskRCNNDataset
from evaluation import run_evaluation, test_runners
from helpers import set_seeds
from settings import *


def main(
    datadir,
    sequences,
    classes,
    model_name,
    batch_size,
    size,
    mode,
    image_folder,
    image_ext,
    state_dict_fpath,
    save_to,
    exclusion_list_filepath=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    if exclusion_list_filepath is not None:
        with open(exclusion_list_filepath) as f:
            exclusion_list = funcy.lmap(tuple, ujson.load(f))
    else:
        exclusion_list = None

    dataset = VocalTractMaskRCNNDataset(
        datadir,
        sequences,
        classes,
        size=size,
        mode=mode,
        image_folder=image_folder,
        image_ext=image_ext,
        include_bkg=(model_name == "maskrcnn"),
        exclusion_list=exclusion_list,
    )

    collate_fn = getattr(VocalTractMaskRCNNDataset, f"{model_name}_collate_fn")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=set_seeds,
        collate_fn=collate_fn
    )

    model = maskrcnn_resnet50_fpn(pretrained=True)
    if state_dict_fpath is not None:
        state_dict = torch.load(state_dict_fpath, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)

    total_parameters = sum(p.numel() for p in model.parameters())
    print(f"""
Mask R-CNN -- {total_parameters} parameters
""")

    if not os.path.exists(save_to):
        os.mkdir(save_to)

    outputs_dir = os.path.join(save_to, "test_outputs")
    class_map = {i: c for c, i in dataset.classes_dict.items()}
    run_test = test_runners[model_name]
    outputs = run_test(
        epoch=0,
        model=model,
        dataloader=dataloader,
        device=device,
        class_map=class_map,
        outputs_dir=outputs_dir
    )

    read_fn = getattr(VocalTractMaskRCNNDataset, f"read_{image_ext}")
    test_results = run_evaluation(
        outputs=outputs,
        save_to=outputs_dir,
        load_fn=lambda fp: dataset.resize(read_fn(fp))
    )

    test_results_filepath = os.path.join(save_to, "test_results.csv")
    pd.DataFrame(test_results).to_csv(test_results_filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="cfg_filepath")
    args = parser.parse_args()

    with open(args.cfg_filepath) as f:
        cfg = yaml.safe_load(f.read())

    main(**cfg)
