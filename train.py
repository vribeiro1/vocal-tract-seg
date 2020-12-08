import pdb

import funcy
import numpy as np
import os
import random
import torch
import torch.nn as nn

from PIL import Image
from sacred import Experiment
from sacred.observers import FileStorageObserver
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from loss import MaskedBCEWithLogitsLoss
from dataset import VocalTractDataset
from torchtools.models import load_model
from settings import BASE_DIR

TRAIN = "train"
VALID = "validation"
TEST = "test"

ex = Experiment()
fs_observer = FileStorageObserver.create(os.path.join(BASE_DIR, "results"))
ex.observers.append(fs_observer)


def set_seeds(worker_id):
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed + 1)
    random.seed(seed + 2)


def save_outputs(fnames, inputs, outputs, outputs_dir, threshold=0.1):
    to_pil = transforms.ToPILImage()
    for fname, in_, out in zip(fnames, inputs, outputs):
        out_arr = out.cpu()
        for c, c_out in enumerate(out_arr):
            out_img = to_pil(c_out)
            out_img.save(os.path.join(outputs_dir, fname + f"_{c}.png"))


def run_epoch(phase, epoch, model, dataloader, optimizer, criterion, writer=None, outputs_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_outputs_dir = None
    if outputs_dir:
        epoch_outputs_dir = os.path.join(outputs_dir, phase, str(epoch))
        if not os.path.exists(epoch_outputs_dir):
            os.makedirs(epoch_outputs_dir)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - {phase}")
    training = phase == TRAIN

    if training:
        model.train()
    else:
        model.eval()

    losses = []
    for i, (fpaths, inputs, targets, masks) in enumerate(progress_bar):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)["out"]
        bs, c, h, w = outputs.shape
        prob_outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, targets, masks)

        if training:
            loss.backward()
            optimizer.step()

        if epoch_outputs_dir:
            fnames = funcy.lmap(lambda fpath: os.path.basename(fpath).split(".")[0], fpaths)
            save_outputs(fnames, inputs, prob_outputs, epoch_outputs_dir)

        losses.append(loss.item())
        progress_bar.set_postfix(loss=np.mean(losses))

    mean_loss = np.mean(losses)
    loss_tag = f"{phase}/loss"
    writer.add_scalar(loss_tag, mean_loss, epoch)

    info = {
        "loss": np.mean(losses)
    }

    return info


@ex.automain
def main(_run, architecture, datadir, batch_size, n_epochs, patience, learning_rate,
         train_sequences, valid_sequences, test_sequences, classes, state_dict_fpath=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(classes)

    writer = SummaryWriter(os.path.join(BASE_DIR, "runs", f"experiment-{_run._id}"))
    best_model_path = os.path.join(fs_observer.dir, "best_model.pth")
    last_model_path = os.path.join(fs_observer.dir, "last_model.pth")

    outputs_dir = os.path.join(fs_observer.dir, "outputs")
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)

    model, info = load_model(
        "segmentation",
        architecture,
        num_classes=num_classes,
        pretrained=True
    )

    if state_dict_fpath is not None:
        state_dict = torch.load(state_dict_fpath, map_location=device)
        model.load_state_dict(state_dict)

    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    base_lr = learning_rate / 10
    max_lr = learning_rate * 10
    scheduler = CyclicLR(
        optimizer,
        base_lr=learning_rate/10,
        max_lr=10*learning_rate,
        cycle_momentum=False
    )

    loss_fn = MaskedBCEWithLogitsLoss()

    train_dataset = VocalTractDataset(datadir, train_sequences, classes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=set_seeds
    )

    valid_dataset = VocalTractDataset(datadir, valid_sequences, classes)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=set_seeds
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
            criterion=loss_fn,
            writer=writer
        )

        info[VALID] = run_epoch(
            phase=VALID,
            epoch=epoch,
            model=model,
            dataloader=valid_dataloader,
            optimizer=optimizer,
            criterion=loss_fn,
            writer=writer,
            outputs_dir=outputs_dir
        )

        scheduler.step()

        if info[VALID]["loss"] < best_metric:
            best_metric = info[VALID]["loss"]
            torch.save(model.state_dict(), best_model_path)
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        torch.save(model.state_dict(), last_model_path)

        if epochs_since_best > patience:
            break

    test_dataset = VocalTractDataset(datadir, test_sequences, classes)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=set_seeds
    )

    best_model_state_dict = torch.load(best_model_path, map_location=device)
    best_model, info = load_model(
        "segmentation",
        "deeplabv3",
        num_classes=num_classes,
        pretrained=False
    )
    best_model.load_state_dict(best_model_state_dict)
    best_model = best_model.to(device)

    info[TEST] = run_epoch(
        phase=TEST,
        epoch=0,
        model=best_model,
        dataloader=test_dataloader,
        optimizer=optimizer,
        criterion=loss_fn,
        writer=writer,
        outputs_dir=outputs_dir
    )
