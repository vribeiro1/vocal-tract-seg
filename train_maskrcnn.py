import logging
import pdb

import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn

from sacred import Experiment
from sacred.observers import FileStorageObserver
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from torch.utils.data import DataLoader
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.transforms import Compose, GaussianBlur
from tqdm import tqdm

from augmentations import MultiCompose, MultiRandomRotation
from dataset import VocalTractMaskRCNNDataset
from evaluation import run_evaluation, test_runners
from helpers import set_seeds
from loss import SoftJaccardBCEWithLogitsLoss
from settings import *

ex = Experiment()
fs_observer = FileStorageObserver.create(os.path.join(BASE_DIR, "results"))
ex.observers.append(fs_observer)


def load_deeplabv3(pretrained, num_classes):
    deeplabv3 = deeplabv3_resnet101(pretrained=pretrained)

    deeplabv3.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    deeplabv3.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    return deeplabv3


def load_maskrcnn(pretrained, *args, **kwargs):
    return maskrcnn_resnet50_fpn(pretrained=pretrained)


model_loaders = {
    "maskrcnn": load_maskrcnn,
    "deeplabv3": load_deeplabv3
}


def run_maskrcnn_epoch(phase, epoch, model, dataloader, optimizer, *args, scheduler=None, writer=None, device=None, **kwargs):
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
                outputs["loss_mask"] + \
                outputs["loss_objectness"] + \
                outputs["loss_rpn_box_reg"]
            )

            if training:
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

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


def run_deeplabv3_epoch(phase, epoch, model, dataloader, optimizer, criterion, *args, scheduler=None, writer=None, device=None, **kwargs):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = phase == TRAIN

    losses = []
    model.train()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - {phase}")
    for i, (_, inputs, targets) in enumerate(progress_bar):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(inputs)["out"]
            loss = criterion(outputs, targets)

            if training:
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

        losses.append({"loss": loss.item()})
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
def main(_run, model_name, datadir, batch_size, n_epochs, patience, learning_rate, weight_decay,
         train_sequences, valid_sequences, test_sequences, classes, size, mode,
         image_folder, image_ext, scheduler_type, state_dict_fpath=None):
    assert model_name in model_loaders.keys(), f"Unsuported model '{model_name}'"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    writer = SummaryWriter(os.path.join(fs_observer.dir, f"experiment-{_run._id}"))
    best_model_path = os.path.join(fs_observer.dir, "best_model.pt")
    last_model_path = os.path.join(fs_observer.dir, "last_model.pt")

    outputs_dir = os.path.join(fs_observer.dir, "outputs")
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)

    num_classes = len(classes)
    model = model_loaders[model_name](pretrained=True, num_classes=num_classes)
    if state_dict_fpath is not None:
        state_dict = torch.load(state_dict_fpath, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if scheduler_type == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)
    elif scheduler_type == "cyclic_lr":
        base_lr = learning_rate / 50
        max_lr = learning_rate
        scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, cycle_momentum=False)
    else:
        logging.info(f"Unrecognized scheduler '{scheduler_type}'. No scheduler will be used instead.")

    # The loss function will be ignored in the case of maskrcnn
    loss_fn = SoftJaccardBCEWithLogitsLoss(jaccard_weight=8)

    input_augmentations = Compose([
        GaussianBlur(kernel_size=3)
    ])

    input_target_augmentations = MultiCompose([
        MultiRandomRotation([-5, 5]),
    ])

    collate_fn = getattr(VocalTractMaskRCNNDataset, f"{model_name}_collate_fn")

    train_dataset = VocalTractMaskRCNNDataset(
        datadir,
        train_sequences,
        classes,
        size=size,
        input_augs=input_augmentations,
        input_target_augs=input_target_augmentations,
        mode=mode,
        image_folder=image_folder,
        image_ext=image_ext,
        include_bkg=(model_name == "maskrcnn")
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=set_seeds,
        collate_fn=collate_fn
    )

    valid_dataset = VocalTractMaskRCNNDataset(
        datadir,
        valid_sequences,
        classes,
        size=size,
        mode=mode,
        image_folder=image_folder,
        image_ext=image_ext,
        include_bkg=(model_name == "maskrcnn")
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=set_seeds,
        collate_fn=collate_fn
    )

    info = {}
    epochs = range(1, n_epochs + 1)
    best_metric = np.inf
    epochs_since_best = 0

    run_epoch = run_maskrcnn_epoch if model_name == "maskrcnn" else run_deeplabv3_epoch
    run_test = test_runners[model_name]
    for epoch in epochs:
        info[TRAIN] = run_epoch(
            phase=TRAIN,
            epoch=epoch,
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler if scheduler_type == "cyclic_lr" else None,
            criterion=loss_fn,
            writer=writer,
            device=device
        )

        info[VALID] = run_epoch(
            phase=VALID,
            epoch=epoch,
            model=model,
            dataloader=valid_dataloader,
            optimizer=optimizer,
            scheduler=scheduler if scheduler_type == "cyclic_lr" else None,
            criterion=loss_fn,
            writer=writer,
            device=device
        )

        if scheduler_type == "reduce_on_plateau":
            scheduler.step(info[VALID]["loss"])

        if info[VALID]["loss"] < best_metric:
            best_metric = info[VALID]["loss"]
            torch.save(model.state_dict(), best_model_path)
            epochs_since_best = 0
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
        mode=mode,
        image_folder=image_folder,
        image_ext=image_ext,
        include_bkg=(model_name == "maskrcnn")
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=set_seeds,
        collate_fn=collate_fn
    )

    best_model = model_loaders[model_name](pretrained=True, num_classes=num_classes)
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

    read_fn = getattr(VocalTractMaskRCNNDataset, f"read_{image_ext}")
    test_results = run_evaluation(
        outputs=test_outputs,
        save_to=outputs_dir,
        load_fn=lambda fp: test_dataset.resize(read_fn(fp))
    )

    test_results_filepath = os.path.join(fs_observer.dir, "test_results.csv")
    pd.DataFrame(test_results).to_csv(test_results_filepath, index=False)
