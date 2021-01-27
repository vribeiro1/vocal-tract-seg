import pdb

import funcy
import json
import numpy as np
import os
import torch
import torch.nn as nn

from sacred import Experiment
from sacred.observers import FileStorageObserver
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
from tqdm import tqdm

from augmentations import (MultiCompose,
                           MultiRandomHorizontalFlip,
                           MultiRandomRotation,
                           MultiRandomVerticalFlip)
from copy import deepcopy
from dataset import VocalTractMaskRCNNDataset
from evaluation import run_evaluation, run_test
from helpers import set_seeds
from settings import *

ex = Experiment()
fs_observer = FileStorageObserver.create(os.path.join(BASE_DIR, "results"))
ex.observers.append(fs_observer)


def run_epoch(phase, epoch, model, dataloader, optimizer, writer=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = phase == TRAIN

    losses = []
    model.train()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - {phase}")
    for i, (_, inputs, targets_dict) in enumerate(progress_bar):
        inputs = inputs.to(device)
        targets_dict = [{
            k: v.to(device) for k, v in d.items()
        } for d in targets_dict]

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(inputs, targets_dict)

            loss = (
                outputs["loss_classifier"] + \
                outputs["loss_box_reg"] + \
                outputs["loss_mask"]
            )

            if training:
                loss.backward()
                optimizer.step()

        losses.append({
            "loss_classifier": outputs["loss_classifier"].item(),
            "loss_box_reg": outputs["loss_box_reg"].item(),
            "loss_mask": outputs["loss_mask"].item(),
            "loss": loss.item()
        })

        mean_loss = np.mean([l["loss"] for l in losses])
        progress_bar.set_postfix(loss=mean_loss)

    mean_loss = np.mean([l["loss"] for l in losses])
    loss_tag = f"{phase}/loss"
    if writer is not None:
        writer.add_scalar(loss_tag, mean_loss, epoch)

    info = {
        "loss": mean_loss
    }

    return info


@ex.automain
def main(_run, datadir, batch_size, n_epochs, patience, learning_rate,
         train_sequences, valid_sequences, test_sequences, classes, size, mode,
         state_dict_fpath=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(classes)

    writer = SummaryWriter(os.path.join(BASE_DIR, "runs", f"experiment-{_run._id}"))
    best_model_path = os.path.join(fs_observer.dir, "best_model.pth")
    last_model_path = os.path.join(fs_observer.dir, "last_model.pth")

    outputs_dir = os.path.join(fs_observer.dir, "outputs")
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)

    model = maskrcnn_resnet50_fpn(pretrained=True)
    if state_dict_fpath is not None:
        state_dict = torch.load(state_dict_fpath, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    base_lr = learning_rate / 10
    max_lr = learning_rate * 10
    scheduler = CyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        cycle_momentum=False
    )

    augmentations = MultiCompose([
        MultiRandomHorizontalFlip(),
        MultiRandomVerticalFlip(),
        MultiRandomRotation([-5, 5]),
    ])

    train_dataset = VocalTractMaskRCNNDataset(
        datadir,
        train_sequences,
        classes,
        size=size,
        augmentations=augmentations,
        mode=mode
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=set_seeds,
        collate_fn=train_dataset.collate_fn
    )

    valid_dataset = VocalTractMaskRCNNDataset(
        datadir,
        valid_sequences,
        classes,
        size=size,
        mode=mode
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=set_seeds,
        collate_fn=valid_dataset.collate_fn
    )

    info = {}
    epochs = range(1, n_epochs + 1)
    best_metric = np.inf
    epochs_since_best = 0

    for epoch in epochs:
        info[TRAIN] = run_epoch(
            phase=TRAIN,
            epoch=epoch,
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            writer=writer,
            device=device
        )

        info[VALID] = run_epoch(
            phase=VALID,
            epoch=epoch,
            model=model,
            dataloader=valid_dataloader,
            optimizer=optimizer,
            writer=writer,
            device=device
        )

        scheduler.step()

        if info[VALID]["loss"] < best_metric:
            best_metric = info[VALID]["loss"]
            torch.save(model.state_dict(), best_model_path)
            epochs_since_best = 0

            run_test(
                epoch=epoch,
                model=model,
                dataloader=valid_dataloader,
                device=device,
                class_map={i: c for c, i in valid_dataset.classes_dict.items()},
                outputs_dir=outputs_dir
            )
        else:
            epochs_since_best += 1

        torch.save(model.state_dict(), last_model_path)

        if epochs_since_best > patience:
            break

    test_dataset = VocalTractMaskRCNNDataset(
        datadir,
        test_sequences,
        classes,
        size=size,
        mode=mode
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=set_seeds,
        collate_fn=test_dataset.collate_fn
    )

    best_model = maskrcnn_resnet50_fpn(pretrained=True)
    state_dict = torch.load(best_model_path, map_location=device)
    best_model.load_state_dict(state_dict)
    best_model.to(device)

    outputs_dir = os.path.join(fs_observer.dir, "test_outputs")
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)

    test_outputs = run_test(
        epoch=0,
        model=best_model,
        dataloader=test_dataloader,
        device=device,
        class_map={i: c for c, i in test_dataset.classes_dict.items()},
        outputs_dir=outputs_dir
    )

    test_results = run_evaluation(
        outputs=test_outputs,
        classes=test_dataset.classes,
        save_to=outputs_dir,
        load_fn=lambda fp: test_dataset.resize(test_dataset.read_dcm(fp))
    )

    test_results_filepath = os.path.join(fs_observer.dir, "test_results.json")
    with open(test_results_filepath, "w") as f:
        json.dump(test_results, f)
