import argparse
import logging
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import os
import tempfile
import torch
import torch.nn as nn
import yaml
import shutil

from glob import glob
from PIL import Image
from roifile import ImagejRoi
from skimage.measure import regionprops
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.transforms import ToTensor
from tqdm import tqdm
from vt_tools.bs_regularization import regularize_Bsplines
from vt_tracker.border_segmentation import detect_borders
from vt_tracker.input import imagenet_normalize

from dataset import VocalTractMaskRCNNDataset
from helpers import set_seeds, sequences_from_dict
from settings import BASE_DIR, DATASET_CONFIG, TRAIN, VALID, TEST

TMPFILES = os.path.join(BASE_DIR, "tmp")
TMP_DIR = tempfile.mkdtemp(dir=TMPFILES)
RESULTS_DIR = os.path.join(TMP_DIR, "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


class RegressionNetwork(nn.Module):
    def __init__(self, num_samples, dropout=0.):
        super().__init__()

        densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.backbone = densenet.features  # 4 x 1024 DenseNet features
        self.pool = nn.MaxPool2d(kernel_size=4)
        self.regressor = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.x_layer = nn.Sequential(
            nn.Linear(256, num_samples),
        )
        self.y_layer = nn.Sequential(
            nn.Linear(256, num_samples),
        )

    def forward(self, x):
        """
        Args:
            x (torch.tensor): Tensor of shape (bs, channels, height, width)
        """
        features = self.backbone(x)
        max_features = self.pool(features)
        max_features = max_features.squeeze(dim=-1).squeeze(dim=-1)
        latent = self.regressor(max_features)
        x_contour = self.x_layer(latent).unsqueeze(dim=1)
        y_contour = self.y_layer(latent).unsqueeze(dim=1)
        contour = torch.concat([x_contour, y_contour], dim=1)
        return contour


class RandomTranslation(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def largest_bbox(props):
        min_y, min_x, max_y, max_x = props.bbox
        bbox_height = max_y - min_y
        bbox_width = max_x - min_x
        return bbox_height * bbox_width

    def forward(self, img, target):
        img = np.array(img)
        regions = regionprops(img)

        if len(regions) == 0:
            return img, target

        props = max(regions, key=self.largest_bbox)
        min_y, min_x, max_y, max_x = props.bbox

        bbox_height = max_y - min_y
        bbox_width = max_x - min_x
        img_height, img_width = img.shape
        x_start = 0
        y_start = 0
        x_end = img_width - bbox_width
        y_end = img_height - bbox_height

        x_translate = np.random.randint(x_start, x_end)
        y_translate = np.random.randint(y_start, y_end)

        crop = img[min_y:max_y, min_x:max_x]
        new_img = np.zeros_like(img)
        new_img[y_translate:(y_translate+bbox_height), x_translate:(x_translate+bbox_width)] = crop
        new_img = Image.fromarray(new_img)

        new_target = np.zeros_like(target)
        new_target[0, ...] = target[0, ...] - min_x + x_translate
        new_target[1, ...] = target[1, ...] - min_y + y_translate

        return new_img, target


class PostprocessingDataset(Dataset):
    def __init__(
            self,
            database_name,
            inputs_dir,
            targets_dir,
            sequences,
            articulators,
            augmentation=True,
        ):
        self.dataset_config = DATASET_CONFIG[database_name]
        self.inputs_dir = inputs_dir
        self.targets_dir = targets_dir
        self.articulators = articulators
        self.to_tensor = ToTensor()
        sequences = sequences_from_dict(inputs_dir, sequences)
        self.data = self._collect_data(
            inputs_dir=inputs_dir,
            targets_dir=targets_dir,
            sequences=sequences,
            articulators=articulators,
        )
        self.augmentation = augmentation
        self.random_translation = RandomTranslation()

    @staticmethod
    def _collect_data(
        inputs_dir,
        targets_dir,
        sequences,
        articulators,
    ):
        data = []
        for subject, sequence in sequences:
            masks_filepaths = glob(os.path.join(
                inputs_dir,
                subject,
                sequence,
                f"*_*.png"
            ))

            for fp_mask in masks_filepaths:
                sequence_dir = os.path.join(targets_dir, subject, sequence)
                filename, _ = os.path.basename(fp_mask).split(".")
                frame, articulator = filename.split("_")

                if articulator not in articulators:
                    continue

                fp_target = os.path.join(sequence_dir, "contours", f"{filename}.roi")

                item = {
                    "subject": subject,
                    "sequence": sequence,
                    "frame": frame,
                    "articulator": articulator,
                    "mask_filepath": fp_mask,
                    "target_filepath": fp_target,
                }
                data.append(item)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        fp_mask = item["mask_filepath"]
        fp_target = item["target_filepath"]

        roi = ImagejRoi.fromfile(fp_target)
        coords = roi.coordinates()
        x_coords, y_coords = regularize_Bsplines(coords, degree=2)
        targets = np.array([x_coords, y_coords])
        mask = Image.open(fp_mask)

        if self.augmentation:
            mask, targets = self.random_translation(mask, targets)

        mask_tensor = self.to_tensor(mask.convert("RGB"))
        mask_tensor = imagenet_normalize(mask_tensor)

        targets = (targets / self.dataset_config.RES)
        targets = torch.from_numpy(targets).type(torch.float)

        return item, mask_tensor, targets


class DataPrep:
    def __init__(
        self,
        dataset_kwargs,
        dataloader_kwargs,
        save_dir,
        device=None,
    ):
        dataset = VocalTractMaskRCNNDataset(**dataset_kwargs)
        self.dataloader = DataLoader(dataset=dataset, **dataloader_kwargs)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.save_dir = save_dir

    def prepare(self):
        for input_infos, inputs, _ in tqdm(self.dataloader, desc="Preprocessing data"):
            detected = detect_borders(inputs, device=self.device.type)
            for (
                subject,
                sequence,
                instance_number,
                im_detected
            ) in zip(
                input_infos["subject"],
                input_infos["sequence"],
                input_infos["instance_number"],
                detected
            ):
                frame_id = "%04d" % instance_number

                save_to = os.path.join(self.save_dir, subject, sequence)
                if not os.path.exists(save_to):
                    os.makedirs(save_to)

                for articulator, articulator_detected in im_detected.items():
                    mask = Image.fromarray(255 * articulator_detected["mask"])
                    mask = mask.convert("L")
                    mask.save(os.path.join(save_to, f"{frame_id}_{articulator}.png"))


def run_epoch(
    phase,
    epoch,
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler=None,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = phase == TRAIN

    if training:
        model.train()
    else:
        model.eval()

    losses = []
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - {phase}")
    for _, masks, targets in progress_bar:
        masks = masks.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(masks)
            loss = criterion(outputs, targets)

            if training:
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

            losses.append(loss.item())
            mean_loss = np.mean(losses)
            progress_bar.set_postfix(loss=mean_loss)

    mean_loss = np.mean(losses)
    info = {"loss": mean_loss}
    return info


def run_test(
    phase,
    epoch,
    model,
    criterion,
    dataloader,
    save_dir,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    losses = []
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - {phase}")
    for items, masks, targets in progress_bar:
        masks = masks.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(masks)
            loss = criterion(outputs, targets)

        losses.append(loss.item())
        mean_loss = np.mean(losses)
        progress_bar.set_postfix(loss=mean_loss)

        for (
            subject,
            sequence,
            frame_id,
            articulator,
            mask,
            output,
            target
        ) in zip(
            items["subject"],
            items["sequence"],
            items["frame"],
            items["articulator"],
            masks,
            outputs,
            targets
        ):
            mask = mask.permute(1, 2, 0)
            mask = mask.detach().cpu()
            mask = (255 * mask.numpy()).astype(np.uint8)
            output = output.detach().cpu()
            target = target.detach().cpu()

            plt.figure(figsize=(5, 5))
            plt.plot(*output, color="blue")
            plt.plot(*target, color="red")
            plt.xlim([0, 1])
            plt.ylim([1, 0])
            plt.tight_layout()
            filename = f"{subject}_{sequence}_{frame_id}_{articulator}".replace("/", "-")
            filepath = os.path.join(save_dir, f"{filename}.png")
            plt.savefig(filepath)
            plt.close()

    mean_loss = np.mean(losses)
    info = {"loss": mean_loss}
    return info


def main(
    database_name,
    datadir,
    batch_size,
    num_epochs,
    patience,
    learning_rate,
    weight_decay,
    train_sequences,
    valid_sequences,
    test_sequences,
    articulators,
    mode="rgb",
    image_folder="NPY_MR",
    image_ext="npy",
    size=(136, 136),
    num_workers=0,
    state_dict_filepath=None,
    checkpoint_filepath=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    best_model_path = os.path.join(RESULTS_DIR, "best_model.pt")
    last_model_path = os.path.join(RESULTS_DIR, "last_model.pt")
    save_checkpoint_path = os.path.join(RESULTS_DIR, "checkpoint.pt")

    model = RegressionNetwork(
        num_samples=50,
        dropout=0.2
    )
    if state_dict_filepath is not None:
        state_dict = torch.load(state_dict_filepath, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=10,
    )

    dataloader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
    )

    train_dataset_kwargs = dict(
        datadir=datadir,
        subj_sequences=train_sequences,
        classes=articulators,
        size=size,
        mode=mode,
        image_folder=image_folder,
        image_ext=image_ext,
        include_bkg=False,
        allow_missing=True,
    )
    train_inputs_dir = os.path.join(TMP_DIR, "train")
    train_prep = DataPrep(
        dataset_kwargs=train_dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
        save_dir=train_inputs_dir,
        device=device,
    )
    train_prep.prepare()

    train_dataset = PostprocessingDataset(
        database_name=database_name,
        inputs_dir=train_inputs_dir,
        targets_dir=datadir,
        sequences=train_sequences,
        articulators=articulators,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
    )

    valid_dataset_kwargs = dict(
        datadir=datadir,
        subj_sequences=valid_sequences,
        classes=articulators,
        size=size,
        mode=mode,
        image_folder=image_folder,
        image_ext=image_ext,
        include_bkg=False,
        allow_missing=True,
    )
    valid_inputs_dir = os.path.join(TMP_DIR, "valid")
    valid_prep = DataPrep(
        dataset_kwargs=valid_dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
        save_dir=valid_inputs_dir,
        device=device,
    )
    valid_prep.prepare()

    valid_dataset = PostprocessingDataset(
        database_name=database_name,
        inputs_dir=valid_inputs_dir,
        targets_dir=datadir,
        sequences=valid_sequences,
        articulators=articulators,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
    )

    epochs = range(1, num_epochs + 1)
    best_metric = np.inf
    epochs_since_best = 0

    if checkpoint_filepath is not None:
        checkpoint = torch.load(checkpoint_filepath, map_location=device)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"]
        epochs = range(epoch, num_epochs + 1)
        best_metric = checkpoint["best_metric"]
        epochs_since_best = checkpoint["epochs_since_best"]

        print(f"""
Loaded checkpoint -- Launching training from epoch {epoch} with best metric
so far {best_metric} seen {epochs_since_best} epochs ago.
""")

    for epoch in epochs:
        info_train = run_epoch(
            phase=TRAIN,
            epoch=epoch,
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        mlflow.log_metrics(
            {"loss": info_train["loss"]},
            step=epoch
        )

        info_valid = run_epoch(
            phase=VALID,
            epoch=epoch,
            model=model,
            dataloader=valid_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        mlflow.log_metrics(
            {"loss": info_train["loss"]},
            step=epoch
        )

        scheduler.step(info_valid["loss"])

        if info_valid["loss"] < best_metric:
            best_metric = info_valid["loss"]
            torch.save(model.state_dict(), best_model_path)
            mlflow.log_artifact(best_model_path)
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        torch.save(model.state_dict(), last_model_path)
        mlflow.log_artifact(last_model_path)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_metric": best_metric,
            "epochs_since_best": epochs_since_best,
            "best_model_path": best_model_path,
            "last_model_path": last_model_path
        }
        torch.save(checkpoint, save_checkpoint_path)
        mlflow.log_artifact(save_checkpoint_path)

        print(f"""
Finished training epoch {epoch}
Best metric: {'%0.4f' % best_metric}, Epochs since best: {epochs_since_best}
""")

        if epochs_since_best > patience:
            break

    test_dataset_kwargs = dict(
        datadir=datadir,
        subj_sequences=test_sequences,
        classes=articulators,
        size=size,
        mode=mode,
        image_folder=image_folder,
        image_ext=image_ext,
        include_bkg=False,
        allow_missing=True,
    )
    test_inputs_dir = os.path.join(TMP_DIR, "test")
    test_prep = DataPrep(
        dataset_kwargs=test_dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
        save_dir=test_inputs_dir,
        device=device,
    )
    test_prep.prepare()

    test_dataset = PostprocessingDataset(
        database_name=database_name,
        inputs_dir=test_inputs_dir,
        targets_dir=datadir,
        sequences=test_sequences,
        articulators=articulators,
        augmentation=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
    )

    best_model = RegressionNetwork(num_samples=50)
    state_dict = torch.load(best_model_path, map_location=device)
    best_model.load_state_dict(state_dict)
    best_model.to(device)

    test_outputs_dir = os.path.join(RESULTS_DIR, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    test_results = run_test(
        phase=TEST,
        epoch=0,
        model=best_model,
        criterion=criterion,
        dataloader=test_dataloader,
        save_dir=test_outputs_dir,
        device=device
    )

    mlflow.log_artifact(test_outputs_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_filepath")
    parser.add_argument("--mlflow", dest="mlflow_tracking_uri", default=None)
    parser.add_argument("--experiment", dest="experiment_name", default="postprocessing")
    parser.add_argument("--run", dest="run_name", default=None)
    args = parser.parse_args()

    if args.mlflow_tracking_uri is not None:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    with open(args.config_filepath) as f:
        cfg = yaml.safe_load(f)

    experiment = mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=args.run_name
    ):
        mlflow.log_artifact(args.config_filepath)
        try:
            main(**cfg)
        finally:
            shutil.rmtree(TMP_DIR)
