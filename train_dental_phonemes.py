import pdb

import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import densenet121
from tqdm import tqdm

from dataset import DentalPhonemesDataset
from helpers import set_seeds
from settings import *

ex = Experiment()
fs_observer = FileStorageObserver.create(os.path.join(BASE_DIR, "results-dental-phonemes"))
ex.observers.append(fs_observer)


def run_epoch(phase, epoch, model, dataloader, optimizer, criterion, writer=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training = phase == TRAIN
    if training:
        model.train()
    else:
        model.eval()

    final_targets = []
    final_outputs = []
    losses = []
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - {phase}")
    for _, inputs, targets in progress_bar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if training:
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        mean_loss = np.mean(losses)
        final_targets.extend([t.item() for t in targets])
        final_outputs.extend([o.item() for o in outputs[:, 1]])
        roc_auc = roc_auc_score(final_targets, final_outputs)
        progress_bar.set_postfix(loss=mean_loss, roc_auc=roc_auc)

    roc_auc = roc_auc_score(final_targets, final_outputs)
    roc_auc_tag = f"{phase}/roc_auc"

    mean_loss = np.mean(losses)
    loss_tag = f"{phase}/loss"

    if writer is not None:
        writer.add_scalar(loss_tag, mean_loss, epoch)
        writer.add_scalar(roc_auc_tag, roc_auc, epoch)

    info = {
        "loss": mean_loss,
        "roc_auc": roc_auc
    }

    return info


def run_test(epoch, model, dataloader, criterion, device, save_filepath=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    final_targets = []
    final_outputs = []
    losses = []
    results = []
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - test")
    for dcm_fpaths, inputs, targets in progress_bar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        losses.append(loss.item())
        mean_loss = np.mean(losses)
        final_targets.extend([t.item() for t in targets])
        final_outputs.extend([o.item() for o in outputs[:, 1]])
        roc_auc = roc_auc_score(final_targets, final_outputs)
        progress_bar.set_postfix(loss=mean_loss, roc_auc=roc_auc)

        outputs = torch.softmax(outputs, dim=1)
        results.extend([
            {
                "filepath": fpath,
                "target": target.item(),
                "pos_prob": output[1].item()
            } for fpath, target, output in zip(dcm_fpaths, targets, outputs)
        ])

    if save_filepath is not None:
        df = pd.DataFrame(results)
        df.to_csv(save_filepath, index=False)

    roc_auc = roc_auc_score(final_targets, final_outputs)
    mean_loss = np.mean(losses)

    info = {
        "loss": mean_loss,
        "roc_auc": roc_auc
    }

    return info, results


@ex.automain
def main(
    _run, datadir, batch_size, n_epochs, patience, learning_rate, weight_decay,
    train_filepath, valid_filepath, test_filepath, size=(136, 136), class_weights=(1, 1),
    mode="gray", state_dict_fpath=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on '{device.type}'")

    writer = SummaryWriter(os.path.join(fs_observer.dir, f"experiment-{_run._id}"))
    best_model_path = os.path.join(fs_observer.dir, "best_model.pt")
    last_model_path = os.path.join(fs_observer.dir, "last_model.pt")

    model = densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    if state_dict_fpath is not None:
        state_dict = torch.load(state_dict_fpath, map_location=device)
        model.load_state_dict(state_dict)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

    class_weights = torch.tensor(class_weights, dtype=torch.float)
    loss_fn = nn.CrossEntropyLoss()

    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-5, 5))
    ])

    train_dataset = DentalPhonemesDataset(
        datadir=datadir,
        filepath=train_filepath,
        size=size,
        augmentation=augmentation,
        mode=mode
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=set_seeds
    )

    valid_dataset = DentalPhonemesDataset(
        datadir=datadir,
        filepath=valid_filepath,
        size=size,
        mode=mode
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=set_seeds
    )

    epochs = range(1, n_epochs + 1)
    best_metric = 0
    epochs_since_best = 0

    for epoch in epochs:
        info_train = run_epoch(
            phase=TRAIN,
            epoch=epoch,
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=loss_fn,
            writer=writer,
            device=device
        )

        info_valid = run_epoch(
            phase=VALID,
            epoch=epoch,
            model=model,
            dataloader=valid_dataloader,
            optimizer=optimizer,
            criterion=loss_fn,
            writer=writer,
            device=device
        )

        scheduler.step(info_valid["loss"])

        if info_valid["roc_auc"] > best_metric:
            best_metric = info_valid["roc_auc"]
            torch.save(model.state_dict(), best_model_path)
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        torch.save(model.state_dict(), last_model_path)

        if epochs_since_best > patience:
            break

    test_dataset = DentalPhonemesDataset(
        datadir=datadir,
        filepath=test_filepath,
        size=size,
        mode=mode
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=set_seeds
    )

    best_model = densenet121(pretrained=True)
    best_model.classifier = nn.Linear(best_model.classifier.in_features, 2)
    state_dict = torch.load(best_model_path, map_location=device)
    best_model.load_state_dict(state_dict)
    best_model.to(device)

    info_test, results = run_test(
        epoch=0,
        model=best_model,
        dataloader=test_dataloader,
        criterion=loss_fn,
        device=device,
        save_filepath=os.path.join(fs_observer.dir, "test_outputs.csv")
    )
