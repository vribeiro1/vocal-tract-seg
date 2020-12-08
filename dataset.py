import pdb

import cv2
import funcy
import logging
import numpy as np
import os
import pydicom
import torch

from glob import glob
from kornia.geometry import Resize
from kornia.augmentation import Normalize
from PIL import Image
from scipy import interpolate
from torch.utils.data import Dataset
from torchvision import transforms

from read_roi import read_roi_zip

MASK = "mask"
ROI = "roi"


class VocalTractDataset(Dataset):
    def __init__(self, datadir, subj_sequences, classes, size=(224, 224), annotation=MASK):
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
        self.normalize = Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        )

        self.annotation = annotation
        if annotation == ROI:
            self.resize = Resize(size)
        else:
            self.resize = transforms.Resize(size)

    @staticmethod
    def _collect_data(datadir, sequences, classes):
        data = []

        for subject, sequence in sequences:
            images = glob(os.path.join(datadir, subject, sequence, "*.dcm"))

            for image_filepath in images:
                image_dirname = os.path.dirname(image_filepath)
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

                is_none = lambda x: x is None
                if all(map(is_none, rois)) and all(map(is_none, targets)):
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
        img = Image.fromarray(ds.pixel_array)

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

    def create_target(self, rois):
        targets = []
        missing = []

        for i, art in enumerate(self.classes):
            roi = rois[art]

            if roi is None:
                target_arr = torch.zeros(self.resize.size)
                missing.append(i)
            else:
                target_arr = self.roi_to_target_tensor(roi, self.resize.size)
            targets.append(target_arr)

        target = self.resize(torch.stack(targets).unsqueeze(0)).squeeze(0)

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

        if self.annotation == ROI:
            img_arr = self.to_tensor(img)
            img_arr = self.resize(img_arr.unsqueeze(0)).squeeze(0)
            target, missing = self.create_target(data_item["rois"])
        else:
            img_arr = self.to_tensor(self.resize(img))
            target, missing = self.load_target(data_item["seg_masks"])

        img_arr = self.normalize(img_arr)
        mask = self.create_mask(target, missing)

        return dcm_fpath, img_arr, target, mask
