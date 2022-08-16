import pdb

import funcy
import numpy as np
import os
import pandas as pd
import torch

from glob import glob
from PIL import Image
from scipy import interpolate
from skimage.measure import regionprops
from scipy.ndimage import binary_fill_holes
from torch.utils.data import Dataset
from torchvision import transforms
from vt_tools import THYROID_CARTILAGE, VOCAL_FOLDS
from vt_tracker.input import InputLoaderMixin

from read_roi import read_roi_zip

MASK = "mask"
ROI = "roi"


class VocalTractMaskRCNNDataset(Dataset, InputLoaderMixin):
    closed_articulators = [
        THYROID_CARTILAGE, VOCAL_FOLDS
    ]

    def __init__(
        self, datadir, subj_sequences, classes, size=(224, 224), annotation=MASK, input_augs=None, input_target_augs=None,
        mode="gray", image_folder="dicoms", image_ext="dcm", allow_missing=False, include_bkg=True
    ):
        if annotation not in (MASK, ROI):
            raise ValueError(f"Annotation level should be either '{MASK}' or '{ROI}'")

        if mode not in ("rgb", "gray"):
            raise ValueError(f"Mode should be either 'rgb' or 'gray'")

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

        self.data = self._collect_data(
            datadir=datadir,
            sequences=sequences,
            classes=classes,
            annotation=annotation,
            img_folder=image_folder,
            img_ext=image_ext,
            allow_missing=allow_missing
        )
        if len(self.data) == 0:
            raise Exception("Empty VocalTractMaskRCNNDataset")

        self.mode = mode
        self.include_bkg = include_bkg
        self.classes = sorted(classes)
        self.classes_dict = {c: i + int(include_bkg) for i, c in enumerate(self.classes)}

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        )

        self.annotation = annotation
        self.resize = transforms.Resize(size)
        self.input_augs = input_augs
        self.input_target_augs = input_target_augs

    @staticmethod
    def _collect_data(datadir, sequences, classes, annotation, img_folder="dicoms", img_ext="dcm", allow_missing=False):
        data = []

        for subject, sequence in sequences:
            images = glob(os.path.join(datadir, subject, sequence, img_folder, f"*.{img_ext}"))

            for image_filepath in images:
                image_dirname = os.path.dirname(os.path.dirname(image_filepath))
                image_name, _ = os.path.basename(image_filepath).rsplit(".", maxsplit=1)
                instance_number = int(image_name)

                instance_number_m1 = instance_number - 1
                image_m1_filepath = os.path.join(
                    os.path.dirname(image_filepath),
                    "%04d" % instance_number_m1 + f".{img_ext}"
                )
                if not os.path.exists(image_m1_filepath):
                    image_m1_filepath = None

                instance_number_p1 = instance_number + 1
                image_p1_filepath = os.path.join(
                    os.path.dirname(image_filepath),
                    "%04d" % instance_number_p1 + f".{img_ext}"
                )
                if not os.path.exists(image_p1_filepath):
                    image_p1_filepath = None

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
                    "instance_number": instance_number,
                    "img_m1_filepath": image_m1_filepath,
                    "dcm_filepath": image_filepath,
                    "img_p1_filepath": image_p1_filepath,
                    "rois": rois,
                    "seg_masks": targets
                }

                is_none = lambda x: x[1] is None
                if annotation == ROI:
                    any_is_missing = any(map(is_none, rois.items()))
                else:
                    any_is_missing = any(map(is_none, targets.items()))
                if not allow_missing and any_is_missing:
                    continue

                if annotation == ROI:
                    all_is_missing = all(map(is_none, rois.items()))
                else:
                    all_is_missing = all(map(is_none, targets.items()))
                if all_is_missing:
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
    def _remove_duplicates(iterable):
        seen = set()
        return [item for item in iterable if not (item in seen or seen.add(item))]

    @staticmethod
    def roi_to_target_tensor(roi, size, closed=False):
        points = VocalTractMaskRCNNDataset._remove_duplicates(zip(roi["x"], roi["y"]))
        x = funcy.lmap(lambda t: t[0], points)
        y = funcy.lmap(lambda t: t[1], points)

        if closed:
            x += [x[0]]
            y += [y[0]]

        tck, _ = interpolate.splprep([x, y], s=0)
        unew = np.arange(0, 1, 0.001)
        out_y, out_x = interpolate.splev(unew, tck)

        res_x = funcy.lmap(lambda v: int(round(v)), out_x)
        res_y = funcy.lmap(lambda v: int(round(v)), out_y)

        target_arr = torch.zeros(size)
        target_arr[res_x, res_y] = 1.0

        return target_arr

    @staticmethod
    def maskrcnn_collate_fn(batch):
        info = [b[0] for b in batch]
        inputs = torch.stack([b[1] for b in batch])
        targets = [b[2] for b in batch]
        return info, inputs, targets


    @staticmethod
    def deeplabv3_collate_fn(batch):
        info = [b[0] for b in batch]
        inputs = torch.stack([b[1] for b in batch])
        targets = torch.stack([b[2]["masks"] for b in batch])

        return info, inputs, targets


    @staticmethod
    def create_background_mask(masks):
        return 1 - torch.max(masks, dim=0)[0]

    @staticmethod
    def get_box_from_mask_tensor(mask, margin=0):
        mask_np = mask.numpy().astype(np.uint8)
        w, h = mask_np.shape

        props = regionprops(mask_np)
        y0, x0, y1, x1 = props[0]["bbox"]

        bbox_x0 = max(0, x0 - margin)
        bbox_y0 = max(0, y0 - margin)
        bbox_x1 = min(w - 1, x1 + margin)
        bbox_y1 = min(h - 1, y1 + margin)

        return torch.tensor([bbox_x0, bbox_y0, bbox_x1, bbox_y1], dtype=torch.float)

    @staticmethod
    def calc_bounding_box_area(bbox):
        x0, y0, x1, y1 = bbox
        l1 = abs(x1 - x0)
        l2 = abs(y1 - y0)

        return l1 * l2

    def create_target(self, rois, original_size):
        targets = []
        missing = []

        for i, art in enumerate(self.classes):
            roi = rois[art]

            if roi is None:
                target_arr = torch.zeros(self.resize.size)
                missing.append(i)
            else:
                target_arr = self.roi_to_target_tensor(roi, original_size, closed=art in self.closed_articulators)
                if art in self.closed_articulators:
                    target_arr = binary_fill_holes(target_arr).astype(float)
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data_item = self.data[item]
        dcm_fpath = data_item["dcm_filepath"]
        img = self.load_input(
            data_item["img_m1_filepath"],
            dcm_fpath,
            data_item["img_p1_filepath"],
            mode=self.mode,
            resize=self.resize
        )

        img_arr = self.to_tensor(img)
        if self.annotation == ROI:
            masks, _ = self.create_target(data_item["rois"], img.size)
        else:
            masks, _ = self.load_target(data_item["seg_masks"])

        masks[masks > 0.5] = 1.0
        masks[masks <= 0.5] = 0.0

        img_arr = self.normalize(img_arr)
        if self.input_augs:
            img_arr = self.input_augs(img_arr)
        if self.input_target_augs:
            img_arr, masks = self.input_target_augs(img_arr, masks)

        boxes = torch.stack([self.get_box_from_mask_tensor(mask, margin=3) for mask in masks])
        labels = torch.tensor(list(self.classes_dict.values()), dtype=torch.int64)

        if self.include_bkg:
            h, w = self.resize.size
            background_bbox = torch.tensor([0., 0., h, w]).unsqueeze(dim=0)
            background_label = torch.tensor([0], dtype = torch.int64)
            background_mask = self.create_background_mask(masks).unsqueeze(dim=0)

            boxes = torch.cat([background_bbox, boxes])
            labels = torch.cat([background_label, labels])
            masks = torch.cat([background_mask, masks])

        target_dict = {
            "boxes": boxes,  # (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H.
            "labels": labels,  # (Int64Tensor[N]): the label for each bounding box. 0 represents always the background class.
            "masks": masks,  # (UInt8Tensor[N, H, W]): The segmentation masks for each one of the objects.
        }

        info = {
            "subject": data_item["subject"],
            "sequence": data_item["sequence"],
            "instance_number": data_item["instance_number"],
            "img_filepath": dcm_fpath
        }

        return info, img_arr, target_dict


class DentalPhonemesDataset(Dataset, InputLoaderMixin):
    def __init__(self, datadir, filepath, size=(224, 224), augmentation=None, mode="gray"):
        if mode not in ("rgb", "gray"):
            raise ValueError(f"Mode should be either 'rgb' or 'gray'")

        self.datadir = datadir
        self.mode = mode
        self.df = pd.read_csv(filepath)

        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(size)
        self.augmentation = augmentation
        self.normalize = transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]

        dcm_fpath = item["img_filepath"]
        dcm_m1_fpath = item["img_m1_filepath"]
        dcm_p1_fpath = item["img_p1_filepath"]

        dcm_fpath = os.path.join(self.datadir, dcm_fpath)
        dcm_m1_fpath = os.path.join(self.datadir, dcm_m1_fpath) if not pd.isna(dcm_m1_fpath) else None
        dcm_p1_fpath = os.path.join(self.datadir, dcm_p1_fpath) if not pd.isna(dcm_p1_fpath) else None

        img = self.load_input(dcm_m1_fpath, dcm_fpath, dcm_p1_fpath, mode=self.mode)

        img_arr = self.to_tensor(self.resize(img))
        img_arr = self.normalize(img_arr)
        if self.augmentation is not None:
            img_arr = self.augmentation(img_arr)

        target = torch.tensor(item["target"], dtype=torch.long)

        return item["img_filepath"], img_arr, target
