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
from tqdm import tqdm
from vt_tools.bs_regularization import regularize_Bsplines
from vt_tracker import border_segmentation
from vt_tracker.postprocessing import POST_PROCESSING, dental_articulation
from vt_tracker.postprocessing.calculate_contours import calculate_contour

from dataset import VocalTractMaskRCNNDataset
from helpers import set_seeds
from settings import *


class InferenceVocalTractMaskRCNNDataset(VocalTractMaskRCNNDataset):
    @staticmethod
    def _collect_data(datadir, sequences, *args, **kwargs):
        data = []

        for subject, sequence in sequences:
            images = sorted(glob(os.path.join(datadir, subject, sequence, "dicoms", "*.dcm")))

            for image_filepath in images:
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
            data_item["dcm_p1_filepath"],
            self.mode
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


def run_border_segmentation_inference(model, dataloader, outputs_dir, class_map, threshold=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    model.eval()
    progress_bar = tqdm(dataloader, desc=f"Inference")

    return_outputs = []
    for info, inputs, _ in progress_bar:
        inputs = inputs.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            for _, (im_info, im_outputs) in enumerate(zip(info, outputs)):
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

            out = {
                "subject": subject,
                "sequence": sequence,
                "instance_number": int(instance_number),
                "dcm_filepath": None,
                "box": None,
                "label": None,
                "pred_cls": pred_class,
                "score": None,
                "mask_filepath": filepath,
                "mask": None
            }

            outputs.append(out)

    return outputs


def smooth_contour(contour):
    resX, resY = regularize_Bsplines(contour, 3)
    return np.array([resX, resY]).T

def load_articulator_array(filepath):
    """
    Loads the target array with the proper orientation (right to left)
    """

    target_array = np.load(filepath)

    # All the countors should be oriented from right to left. If it is the opposite,
    # we flip the array.
    if target_array[0][0] < target_array[-1][0]:
        target_array = np.flip(target_array, axis=0)

    return target_array.copy()


def process_out(output_item, datadir, save_to):
    subject = output_item["subject"]
    sequence = output_item["sequence"]
    instance_number = output_item["instance_number"]
    pred_class = output_item["pred_cls"]

    outputs_dir = os.path.join(save_to, "inference_contours", subject, sequence)
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    if "mask_filepath" in output_item:
        mask_filepath = output_item["mask_filepath"]
        mask_img = Image.open(mask_filepath).convert("L")
        mask = np.array(mask_img) / 255.
    else:
        mask = output_item["mask"]

    gravity_curve_filepath = os.path.join(
        datadir,
        subject,
        sequence,
        "inference_contours",
        f"{'%04d' % instance_number}_upper-incisor.npy"
    )

    if os.path.isfile(gravity_curve_filepath):
        gravity_curve = load_articulator_array(gravity_curve_filepath)[:-10]
    else:
        gravity_curve = None

    post_proc_cfg = deepcopy(POST_PROCESSING[pred_class])
    if pred_class == TONGUE and not output_item["is_dental"]:
        # Deactivate gravity algorithm for non-dental articulations
        post_proc_cfg["G"] = 0
        post_proc_cfg["delta"] = 0

    contour = calculate_contour(pred_class, mask, gravity_curve=gravity_curve, cfg=post_proc_cfg)
    if len(contour) > 0:
        contour = smooth_contour(contour)

        npy_filepath = os.path.join(outputs_dir, f"{'%04d' % instance_number}_{pred_class}.npy")
        with open(npy_filepath, "wb") as f:
            np.save(f, contour)

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "maskrcnn"

    collate_fn = getattr(InferenceVocalTractMaskRCNNDataset, f"{model_name}_collate_fn")

    dataset = InferenceVocalTractMaskRCNNDataset(
        cfg["datadir"],
        cfg["sequences"],
        cfg["classes"],
        size=cfg["size"],
        mode=cfg["mode"],
        image_folder=cfg["image_folder"],
        image_ext=["image_ext"],
        include_bkg=(model_name == "maskrcnn")
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        worker_init_fn=set_seeds,
        collate_fn=collate_fn
    )

    inference_directory = cfg.get("inference_dir")

    if not os.path.exists(cfg["save_to"]):
        os.mkdir(cfg["save_to"])

    # densenet = dental_articulation.load_model(device.type)
    # progress_bar = tqdm(dataloader, desc=f"Dental articulation")

    # make_key = lambda d: "_".join([
    #     d["subject"],
    #     d["sequence"],
    #     "%04d" % d["instance_number"]
    # ])

    # is_dental_map = {}
    # for batch_info, batch_inputs, _ in progress_bar:
    #     batch_outputs = dental_articulation.discriminate_dental_articulations(
    #         batch_inputs, device=device.type, model=densenet
    #     )
    #     for info, output in zip(batch_info, batch_outputs):
    #         is_dental = bool(output.item())
    #         is_dental_map[make_key(info)] = is_dental

    if inference_directory is None:
        state_dict = cfg.get("state_dict_fpath")
        mask_rcnn = border_segmentation.load_model(device.type, state_dict_filepath=state_dict)

        class_map = {i: c for c, i in dataset.classes_dict.items()}
        border_seg_outputs = run_border_segmentation_inference(
            model=mask_rcnn,
            dataloader=dataloader,
            device=device,
            class_map=class_map,
            outputs_dir=os.path.join(cfg["save_to"], "inference")
        )
    else:
        border_seg_outputs = load_outputs_from_directory(
            inference_directory, cfg["sequences"], cfg["classes"]
        )

    for output_item in tqdm(border_seg_outputs, desc="Calculating contours"):
        # is_dental = is_dental_map[make_key(output_item)]
        is_dental = False
        output_item["is_dental"] = is_dental
        process_out(output_item, cfg["datadir"], cfg["save_to"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="cfg_filepath")
    args = parser.parse_args()

    with open(args.cfg_filepath) as f:
        cfg = yaml.safe_load(f.read())

    main(cfg)
