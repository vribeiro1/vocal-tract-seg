import pdb

import cv2
import funcy
import logging
import numpy as np
import os
import pydicom
import torch

from glob import glob
from kornia.augmentation import Normalize
from PIL import Image
from scipy import interpolate
from skimage.measure import regionprops
from torch.utils.data import Dataset
from torchvision import transforms

from connect_points import uint16_to_uint8
from read_roi import read_roi_zip

MASK = "mask"
ROI = "roi"


class VocalTractDataset(Dataset):
    def __init__(self, datadir, subj_sequences, classes, size=(224, 224), annotation=MASK, augmentations=None):
        if annotation not in [MASK, ROI]:
            raise ValueError(f"Annotation level should be either '{MASK}' or {ROI}")

        sequences = []
        for subj, seqs in subj_sequences.items():
            use_seqs = seqs
            if len(seqs) == 0:
                # Use all sequences
                use_seqs = filter(
                    lambda s: s.startswith("S") and os.path.isdir(os.path.join(datadir, subj, s)),
                    os.listdir(os.path.join(datadir, subj))
                )

            sequences.extend([(subj, seq) for seq in use_seqs])

        self.data = self._collect_data(datadir, sequences, classes)
        if len(self.data) == 0:
            raise Exception("Empty VocalTractDataset")

        self.classes = sorted(classes)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        )

        self.annotation = annotation
        self.resize = transforms.Resize(size)
        self.augmentations = augmentations

    @staticmethod
    def _collect_data(datadir, sequences, classes):
        data = []

        for subject, sequence in sequences:
            images = glob(os.path.join(datadir, subject, sequence, "dicoms", "*.dcm"))
            for image_filepath in images:
                image_dirname = os.path.dirname(os.path.dirname(image_filepath))
                image_name = os.path.basename(image_filepath).rsplit(".", maxsplit=1)[0]

                zip_filepath = os.path.join(image_dirname, "contours", image_name + ".zip")
                if not os.path.exists(zip_filepath):
                    rois = {art: None for art in classes}
                else:
                    rois = VocalTractDataset._collect_rois(zip_filepath, classes)

                targets_dirname = os.path.join(image_dirname, "masks")
                targets = VocalTractDataset._collect_seg_masks(targets_dirname, image_name, classes)

                item = {
                    "subject": subject,
                    "sequence": sequence,
                    "dcm_filepath": image_filepath,
                    "rois": rois,
                    "seg_masks": targets
                }

                is_none = lambda x: x[1] is None
                if all(map(is_none, rois.items())) and all(map(is_none, targets.items())):
                    # Skip if all annotations are None
                    continue

                data.append(item)

        return data

    @staticmethod
    def _collect_rois(zip_filepath, classes):
        rois_tmp = read_roi_zip(zip_filepath)
        rois = {art.split("_")[1]: roi for art, roi in rois_tmp.items()}
        rois.update({art: None for art in classes if art not in rois})

        return rois

    @staticmethod
    def _collect_seg_masks(masks_dirname, image_name, classes):
        masks = {art: None for art in classes}
        for art in classes:
            mask_filepath = os.path.join(masks_dirname, f"{image_name}_{art}.png")
            if not os.path.exists(mask_filepath):
                mask_filepath = None

            masks[art] = mask_filepath

        return masks

    @staticmethod
    def read_dcm(dcm_fpath):
        ds = pydicom.dcmread(dcm_fpath, force=True)
        img = Image.fromarray(uint16_to_uint8(ds.pixel_array))

        return img.convert("RGB")

    @staticmethod
    def _remove_duplicates(iterable):
        seen = set()
        return [item for item in iterable if not (item in seen or seen.add(item))]

    @staticmethod
    def roi_to_target_tensor(roi, size):
        points = VocalTractDataset._remove_duplicates(zip(roi["x"], roi["y"]))
        x = funcy.lmap(lambda t: t[0], points)
        y = funcy.lmap(lambda t: t[1], points)

        tck, _ = interpolate.splprep([x, y], s=0)
        unew = np.arange(0, 1, 0.001)
        out_y, out_x = interpolate.splev(unew, tck)

        res_x = funcy.lmap(lambda v: int(round(v)), out_x)
        res_y = funcy.lmap(lambda v: int(round(v)), out_y)

        target_arr = torch.zeros(size)
        target_arr[res_x, res_y] = 1.0

        return target_arr

    def create_target(self, rois, original_size):
        targets = []
        missing = []

        for i, art in enumerate(self.classes):
            roi = rois[art]

            if roi is None:
                target_arr = torch.zeros(self.resize.size)
                missing.append(i)
            else:
                target_arr = self.roi_to_target_tensor(roi, original_size)
                target_arr = self.resize(target_arr)
            targets.append(target_arr)

        target = torch.stack(targets)
        return target.float(), missing

    def load_target(self, targets_fpaths):
        targets = []
        missing = []

        for i, art in enumerate(self.classes):
            filepath = targets_fpaths[art]

            if filepath is None:
                target_arr = torch.zeros(self.resize.size)
                missing.append(i)
            else:
                target_img = Image.open(filepath).convert("L")
                target_img = self.resize(target_img)
                target_arr = self.to_tensor(target_img).squeeze(0)
            targets.append(target_arr)

        target = torch.stack(targets)
        return target.float(), missing

    def create_mask(self, target, missing_indices):
        mask = torch.ones_like(target, dtype=torch.float)
        mask[missing_indices] = torch.max(target, dim=0)[0]
        return mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data_item = self.data[item]
        dcm_fpath = data_item["dcm_filepath"]
        img = self.read_dcm(dcm_fpath)

        img_arr = self.to_tensor(self.resize(img))
        if self.annotation == ROI:
            target, missing = self.create_target(data_item["rois"], img.size)
        else:
            target, missing = self.load_target(data_item["seg_masks"])

        img_arr = self.normalize(img_arr)
        mask = self.create_mask(target, missing)

        if self.augmentations:
            img_arr, target, mask = self.augmentations(img_arr, target, mask)

        return dcm_fpath, img_arr, target, mask


class VocalTractMaskRCNNDataset(Dataset):
    def __init__(self, datadir, subj_sequences, classes, size=(224, 224), annotation=MASK, augmentations=None):
        if annotation not in [MASK, ROI]:
            raise ValueError(f"Annotation level should be either '{MASK}' or {ROI}")

        sequences = []
        for subj, seqs in subj_sequences.items():
            use_seqs = seqs
            if len(seqs) == 0:
                # Use all sequences
                use_seqs = filter(
                    lambda s: s.startswith("S") and os.path.isdir(os.path.join(datadir, subj, s)),
                    os.listdir(os.path.join(datadir, subj))
                )

            sequences.extend([(subj, seq) for seq in use_seqs])

        self.data = self._collect_data(datadir, sequences, classes)
        if len(self.data) == 0:
            raise Exception("Empty VocalTractMaskRCNNDataset")

        self.classes = sorted(classes)
        self.classes_dict = {c: i + 1 for i, c in enumerate(self.classes)}

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        )

        self.annotation = annotation
        self.resize = transforms.Resize(size)
        self.augmentations = augmentations

    @staticmethod
    def _collect_data(datadir, sequences, classes):
        data = []

        for subject, sequence in sequences:
            images = glob(os.path.join(datadir, subject, sequence, "dicoms", "*.dcm"))

            for image_filepath in images:
                image_dirname = os.path.dirname(os.path.dirname(image_filepath))
                image_name = os.path.basename(image_filepath).rsplit(".", maxsplit=1)[0]

                zip_filepath = os.path.join(image_dirname, "contours", image_name + ".zip")
                if not os.path.exists(zip_filepath):
                    rois = {art: None for art in classes}
                else:
                    rois = VocalTractMaskRCNNDataset._collect_rois(zip_filepath, classes)

                targets_dirname = os.path.join(image_dirname, "masks")
                targets = VocalTractMaskRCNNDataset._collect_seg_masks(targets_dirname, image_name, classes)

                item = {
                    "subject": subject,
                    "sequence": sequence,
                    "instance_number": image_name,
                    "dcm_filepath": image_filepath,
                    "rois": rois,
                    "seg_masks": targets
                }

                is_none = lambda x: x[1] is None
                if all(map(is_none, rois.items())) and all(map(is_none, targets.items())):
                    # Skip if all annotations are None
                    continue

                data.append(item)

        return data

    @staticmethod
    def _collect_rois(zip_filepath, classes):
        rois_tmp = read_roi_zip(zip_filepath)
        rois = {art.split("_")[1]: roi for art, roi in rois_tmp.items()}
        rois.update({art: None for art in classes if art not in rois})

        return rois

    @staticmethod
    def _collect_seg_masks(masks_dirname, image_name, classes):
        masks = {art: None for art in classes}
        for art in classes:
            mask_filepath = os.path.join(masks_dirname, f"{image_name}_{art}.png")
            if not os.path.exists(mask_filepath):
                mask_filepath = None

            masks[art] = mask_filepath

        return masks

    @staticmethod
    def read_dcm(dcm_fpath):
        ds = pydicom.dcmread(dcm_fpath, force=True)
        img = Image.fromarray(uint16_to_uint8(ds.pixel_array))

        return img.convert("RGB")

    @staticmethod
    def _remove_duplicates(iterable):
        seen = set()
        return [item for item in iterable if not (item in seen or seen.add(item))]

    @staticmethod
    def roi_to_target_tensor(roi, size):
        points = VocalTractMaskRCNNDataset._remove_duplicates(zip(roi["x"], roi["y"]))
        x = funcy.lmap(lambda t: t[0], points)
        y = funcy.lmap(lambda t: t[1], points)

        tck, _ = interpolate.splprep([x, y], s=0)
        unew = np.arange(0, 1, 0.001)
        out_y, out_x = interpolate.splev(unew, tck)

        res_x = funcy.lmap(lambda v: int(round(v)), out_x)
        res_y = funcy.lmap(lambda v: int(round(v)), out_y)

        target_arr = torch.zeros(size)
        target_arr[res_x, res_y] = 1.0

        return target_arr

    @staticmethod
    def collate_fn(batch):
        info = [b[0] for b in batch]
        inputs = torch.stack([b[1] for b in batch])
        targets = [b[2] for b in batch]
        return info, inputs, targets

    def create_target(self, rois, original_size):
        targets = []
        missing = []

        for i, art in enumerate(self.classes):
            roi = rois[art]

            if roi is None:
                target_arr = torch.zeros(self.resize.size)
                missing.append(i)
            else:
                target_arr = self.roi_to_target_tensor(roi, original_size)
                target_arr = self.resize(target_arr)
            targets.append(target_arr)

        target = torch.stack(targets)
        return target.float(), missing

    def load_target(self, targets_fpaths):
        targets = []
        missing = []

        for i, art in enumerate(self.classes):
            filepath = targets_fpaths[art]

            if filepath is None:
                target_arr = torch.zeros(self.resize.size)
                missing.append(i)
            else:
                target_img = Image.open(filepath).convert("L")
                target_img = self.resize(target_img)
                target_arr = self.to_tensor(target_img).squeeze(0)
            targets.append(target_arr)

        target = torch.stack(targets)
        return target.float(), missing

    @staticmethod
    def create_background_mask(masks):
        return 1 - torch.max(masks, dim=0)[0]

    @staticmethod
    def get_box_from_mask_tensor(mask, margin=0):
        mask_np = mask.numpy().astype(np.uint8)
        props = regionprops(mask_np)
        y0, x0, y1, x1 = props[0]["bbox"]
        return torch.tensor([x0 - margin, y0 - margin, x1 + margin, y1 + margin], dtype=torch.float)

    @staticmethod
    def calc_bounding_box_area(bbox):
        x0, y0, x1, y1 = bbox
        l1 = abs(x1 - x0)
        l2 = abs(y1 - y0)

        return l1 * l2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data_item = self.data[item]
        dcm_fpath = data_item["dcm_filepath"]
        img = self.read_dcm(dcm_fpath)

        img_arr = self.to_tensor(self.resize(img))
        if self.annotation == ROI:
            masks, missing = self.create_target(data_item["rois"], img.size)
        else:
            masks, missing = self.load_target(data_item["seg_masks"])

        masks[masks > 0.5] = 1.0
        masks[masks <= 0.5] = 0.0

        img_arr = self.normalize(img_arr)
        if self.augmentations:
            img_arr, masks = self.augmentations(img_arr, masks)

        h, w = self.resize.size
        boxes = torch.stack(
            [torch.tensor([0., 0., h, w])] +
            [self.get_box_from_mask_tensor(mask, margin=3) for mask in masks]
        )
        labels = torch.tensor([0] + list(self.classes_dict.values()), dtype=torch.int64)

        background = self.create_background_mask(masks).unsqueeze(dim=0)
        masks = torch.cat([background, masks])

        target_dict = {
            "boxes": boxes,  # (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H.
            "labels": labels,  # (Int64Tensor[N]): the label for each bounding box. 0 represents always the background class.
            "masks": masks,  # (UInt8Tensor[N, H, W]): The segmentation masks for each one of the objects.
        }

        info = {
            "subject": data_item["subject"],
            "sequence": data_item["sequence"],
            "instance_number": data_item["instance_number"],
            "dcm_filepath": dcm_fpath
        }

        return info, img_arr, target_dict
