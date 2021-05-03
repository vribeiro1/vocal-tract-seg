import pdb

import argparse
import funcy
import numpy as np
import os
import torch
import yaml

from copy import deepcopy
from glob import glob
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
from tqdm import tqdm

from bs_regularization import regularize_Bsplines
from dataset import VocalTractMaskRCNNDataset
from evaluation import calculate_contour
from helpers import set_seeds
from settings import *

COLORS = {
    "epiglottis": (153, 204, 0),
    "lower-lip": (102, 255, 51),
    "upper-lip": (255, 51, 204),
    "tongue": (255, 102, 0),
    "soft-palate": (102, 102, 255),
    "pharynx": (255, 128, 128)
}


class InferenceVocalTractMaskRCNNDataset(VocalTractMaskRCNNDataset):
    @staticmethod
    def _collect_data(datadir, sequences, classes):
        data = []

        for subject, sequence in sequences:
            images = sorted(glob(os.path.join(datadir, subject, sequence, "dicoms", "*.dcm")))

            for image_filepath in images:
                image_dirname = os.path.dirname(os.path.dirname(image_filepath))
                image_name = os.path.basename(image_filepath).rsplit(".", maxsplit=1)[0]
                instance_number = int(image_name)

                instance_number_m1 = instance_number - 1
                image_m1_filepath = os.path.join(
                    os.path.dirname(image_filepath),
                    "%04d" % instance_number_m1 + ".dcm"
                )
                if not os.path.exists(image_m1_filepath):
                    image_m1_filepath = None

                instance_number_p1 = instance_number + 1
                image_p1_filepath = os.path.join(
                    os.path.dirname(image_filepath),
                    "%04d" % instance_number_p1 + ".dcm"
                )
                if not os.path.exists(image_p1_filepath):
                    image_p1_filepath = None

                item = {
                    "subject": subject,
                    "sequence": sequence,
                    "instance_number": instance_number,
                    "dcm_m1_filepath": image_m1_filepath,
                    "dcm_filepath": image_filepath,
                    "dcm_p1_filepath": image_p1_filepath,
                }

                data.append(item)

        return data

    def __getitem__(self, item):
        data_item = self.data[item]
        dcm_fpath = data_item["dcm_filepath"]
        img = self.load_input(
            data_item["dcm_m1_filepath"],
            dcm_fpath,
            data_item["dcm_p1_filepath"]
        )

        img_arr = self.to_tensor(self.resize(img))
        img_arr = self.normalize(img_arr)

        if self.augmentations:
            img_arr = self.augmentations(img_arr)

        info = {
            "subject": data_item["subject"],
            "sequence": data_item["sequence"],
            "instance_number": data_item["instance_number"],
            "dcm_filepath": dcm_fpath
        }

        return info, img_arr, {}


def run_inference(model, dataloader, outputs_dir, class_map, threshold=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    model.eval()
    progress_bar = tqdm(dataloader, desc=f"Inference")

    return_outputs = []
    for i, (info, inputs, _) in enumerate(progress_bar):
        inputs = inputs.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            for j, (im_info, im_outputs) in enumerate(zip(info, outputs)):
                zipped = list(zip(
                    im_outputs["boxes"],
                    im_outputs["labels"],
                    im_outputs["scores"],
                    im_outputs["masks"]
                ))

                detected = []
                for c_idx, c in class_map.items():
                    art_list = funcy.lfilter(lambda t: t[1].item() == c_idx, zipped)
                    if len(art_list) > 0:
                        art = max(art_list, key=lambda t: t[2])
                        detected.append(art)

                for box, label, score, mask in detected:
                    mask_arr = mask.squeeze(dim=0).cpu().numpy()
                    if threshold is not None:
                        mask_arr[mask_arr > threshold] = 1.
                        mask_arr[mask_arr <= threshold] = 0.
                    mask_arr = mask_arr * 255
                    mask_arr = mask_arr.astype(np.uint8)

                    mask_img = Image.fromarray(mask_arr)

                    mask_dirname = os.path.join(
                        outputs_dir,
                        im_info["subject"],
                        im_info["sequence"]
                    )
                    if not os.path.exists(mask_dirname):
                        os.makedirs(mask_dirname)

                    pred_cls = class_map[label.item()]
                    mask_filepath = os.path.join(
                        mask_dirname,
                        f"{'%04d' % im_info['instance_number']}_{pred_cls}.png"
                    )
                    mask_img.save(mask_filepath)

                    im_outputs_with_info = deepcopy(im_info)
                    im_outputs_with_info.update({
                        "box": list(box),
                        "label": label,
                        "pred_cls": pred_cls,
                        "score": score,
                        "mask": mask.squeeze(dim=0).cpu().numpy()
                    })
                    return_outputs.append(im_outputs_with_info)

    return return_outputs


def load_outputs_from_directory(outputs_dir, subj_sequences, classes):
    outputs = []

    sequences = []
    for subj, seqs in subj_sequences.items():
        use_seqs = seqs
        if len(seqs) == 0:
            # Use all sequences
            use_seqs = filter(
                lambda s: s.startswith("S") and os.path.isdir(os.path.join(outputs_dir, subj, s)),
                os.listdir(os.path.join(outputs_dir, subj))
            )

        sequences.extend([(subj, seq) for seq in use_seqs])

    for subject, sequence in sequences:
        seq_dir = os.path.join(outputs_dir, subject, sequence)

        for filepath in sorted(glob(os.path.join(seq_dir, "*.png"))):
            basename = os.path.basename(filepath)
            name, _ = basename.split(".")
            instance_number, pred_class = name.split("_")

            if pred_class not in classes:
                continue

            img = Image.open(filepath).convert("L")
            img_arr = np.array(img) / 255.

            # if pred_class == "tongue":
            #     img_arr[85:135, 0:75] = np.zeros((135 - 85, 75 - 0))
            # img_arr[img_arr < 0.1] = 0.

            out = {
                "subject": subject,
                "sequence": sequence,
                "instance_number": int(instance_number),
                "dcm_filepath": None,
                "box": None,
                "label": None,
                "pred_cls": pred_class,
                "score": None,
                "mask": img_arr
            }

            outputs.append(out)

    return outputs


def smooth_contour(contour):
    resX, resY = regularize_Bsplines(contour, 3)
    return np.array([resX, resY]).transpose(1, 0)


def process_out(out, save_to):
    subject = out["subject"]
    sequence = out["sequence"]
    instance_number = out["instance_number"]
    pred_class = out["pred_cls"]

    outputs_dir = os.path.join(cfg["save_to"], "inference_contours", subject, sequence)
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    mask = out["mask"]
    contour = calculate_contour(pred_class, mask)
    if len(contour) > 0:
        contour = smooth_contour(contour)

        npy_filepath = os.path.join(outputs_dir, f"{'%04d' % instance_number}_{pred_class}.npy")
        with open(npy_filepath, "wb") as f:
            np.save(f, contour)


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(cfg["classes"])

    dataset = InferenceVocalTractMaskRCNNDataset(
        cfg["datadir"],
        cfg["sequences"],
        cfg["classes"],
        size=cfg["size"],
        mode=cfg["mode"]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        worker_init_fn=set_seeds,
        collate_fn=dataset.collate_fn
    )

    inference_directory = cfg.get("inference_dir")

    model = maskrcnn_resnet50_fpn(pretrained=True)
    if cfg["state_dict_fpath"] is not None:
        state_dict = torch.load(cfg["state_dict_fpath"], map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)

    if not os.path.exists(cfg["save_to"]):
        os.mkdir(cfg["save_to"])

    if inference_directory is None:
        class_map = {i: c for c, i in dataset.classes_dict.items()}
        outputs = run_inference(
            model=model,
            dataloader=dataloader,
            device=device,
            class_map=class_map,
            outputs_dir=os.path.join(cfg["save_to"], "inference")
        )
    else:
        outputs = load_outputs_from_directory(inference_directory, cfg["sequences"], cfg["classes"])

    for out in tqdm(outputs, desc="Calculating contours"):
        process_out(out, cfg["save_to"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="cfg_filepath")
    args = parser.parse_args()

    with open(args.cfg_filepath) as f:
        cfg = yaml.safe_load(f.read())

    main(cfg)
