import pdb

import cv2
import funcy
import numpy as np
import os
import torch

from copy import deepcopy
from PIL import Image, ImageDraw
from tqdm import tqdm

from connect_points import (connect_with_active_contours,
                            connect_points_graph_based,
                            connect_with_skeleton,
                            draw_contour, rescale_contour)
from dataset import VocalTractMaskRCNNDataset
from metrics import evaluate_model
from settings import *


def save_image_with_contour(img, filepath, contour, target=None):
    mask_contour = np.array(img)
    if target is not None:
        mask_contour[:, :, 0][target > 200] = 0
        mask_contour[:, :, 1][target > 200] = 0
        mask_contour[:, :, 2][target > 200] = 255
    mask_contour = draw_contour(mask_contour, contour, color=(255, 0, 0))

    mask_contour_img = Image.fromarray(mask_contour)
    mask_contour_img.save(filepath)


def calculate_contour(pred_class, mask, box):
    if pred_class not in POST_PROCESSING:
        raise KeyError(
            f"Class '{pred_class}' does not have post-processing parameters configured"
        )

    post_processing = POST_PROCESSING[pred_class]
    post_processing_method = post_processing["method"]
    alpha = post_processing["alpha"]
    beta = post_processing["beta"]
    gamma = post_processing["gamma"]
    upscale = post_processing["upscale"]
    threshold = post_processing["threshold"]

    contour = []
    threshold_tmp = threshold

    # If we upscale before post-processing, we need to define a function to rescale the generated
    # contour. Else, we use an identity function as a placeholder.
    if upscale is not None:
        s_out, _ = mask.shape
        s_in = upscale
        mask = cv2.resize(mask, (upscale, upscale), interpolation=cv2.INTER_CUBIC)
        rescale_contour_fn = funcy.partial(rescale_contour, s_in=s_in, s_out=s_out)
    else:
        rescale_contour_fn = lambda x: x

    while len(contour) < 10:
        mask_thr = mask.copy()
        mask_thr[mask_thr <= threshold_tmp] = 0.
        mask_thr[mask_thr > threshold_tmp] = 1.

        if post_processing_method == GRAPH_BASED:
            contour, _, _ = connect_points_graph_based(mask_thr, 3, alpha, beta, gamma)
        elif post_processing_method == ACTIVE_CONTOURS:
            contour = connect_with_active_contours(mask_thr, pred_class, alpha, beta, gamma)
        elif post_processing_method == SKELETON:
            contour = connect_with_skeleton(mask_thr)
        else:
            raise ValueError(f"Unavailable post-processing method '{post_processing}'")

        threshold_tmp = threshold_tmp - 0.05
        if threshold_tmp < 0.0:
            break

    contour = rescale_contour_fn(contour)
    return contour


def draw_bbox(mask, bbox, text=None):
    x0, y0, x1, y1 = bbox

    mask_img = mask.convert("RGB")
    draw = ImageDraw.Draw(mask_img)
    draw.line([
        (x0, y0),
        (x1, y0),
        (x1, y1),
        (x0, y1),
        (x0, y0)
    ], fill=(255, 0, 0), width=2)

    if text is not None:
        draw.text((10, 10), text, (0, 0, 255))

    return mask_img


def run_test(epoch, model, dataloader, outputs_dir, class_map, threshold=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_outputs_dir = os.path.join(outputs_dir, "inference", str(epoch))
    if not os.path.exists(epoch_outputs_dir):
        os.makedirs(epoch_outputs_dir)

    model.eval()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - inference")

    return_outputs = []
    for i, (info, inputs, targets_dict) in enumerate(progress_bar):
        inputs = inputs.to(device)
        targets_dict = [{
            k: v.to(device) for k, v in d.items()
        } for d in targets_dict]

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            for j, (im_info, im_outputs, targets_dict) in enumerate(zip(info, outputs, targets_dict)):
                targets = targets_dict["masks"]
                bg, llip, sp, tongue, ulip = targets
                targets = {
                    "lower-lip": llip,
                    "soft-palate": sp,
                    "tongue": tongue,
                    "upper-lip": ulip
                }

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
                        detected.append(art + (targets[c],))

                for box, label, score, mask, target in detected:
                    mask_arr = mask.squeeze(dim=0).cpu().numpy()
                    if threshold is not None:
                        mask_arr[mask_arr > threshold] = 1.
                        mask_arr[mask_arr <= threshold] = 0.
                    mask_arr = mask_arr * 255
                    mask_arr = mask_arr.astype(np.uint8)

                    mask_img = Image.fromarray(mask_arr)
                    # mask_img = draw_bbox(mask_img, box, "%.4f" % score)

                    mask_dirname = os.path.join(
                        epoch_outputs_dir,
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
                        "mask": mask.squeeze(dim=0).cpu().numpy(),
                        "target": target.squeeze(dim=0).cpu().numpy()
                    })
                    return_outputs.append(im_outputs_with_info)

    return return_outputs


def run_evaluation(outputs, classes, save_to=None, load_fn=None):
    pred_classes = []
    targets = []
    contours = []
    for out in outputs:
        target = out["target"] * 255
        box = out["box"]
        mask = out["mask"]
        pred_class = out['pred_cls']

        contour = calculate_contour(pred_class, mask, box)
        zeros = np.zeros_like(mask)
        clean_contour = draw_contour(zeros, contour, color=(255, 255, 255))

        if save_to is not None:
            if load_fn is None:
                raise ValueError("If 'save_to' is passed, 'load' cannot be None")

            outputs_dir = os.path.join(
                save_to, "inference_contours", out["subject"], out["sequence"]
            )
            if not os.path.exists(outputs_dir):
                os.makedirs(outputs_dir)

            img = load_fn(out["dcm_filepath"])
            img_filepath = os.path.join(
                outputs_dir,
                f"{'%04d' % out['instance_number']}_{pred_class}.png"
            )
            save_image_with_contour(img, img_filepath, contour, target)

            npy_filepath = os.path.join(
                outputs_dir,
                f"{'%04d' % out['instance_number']}_{pred_class}.npy"
            )
            with open(npy_filepath, "wb") as f:
                np.save(f, contour)

        targets.append(target)
        contours.append(clean_contour)
        pred_classes.append(pred_class)

    results = {}
    zipped = list(zip(pred_classes, targets, contours))
    for cls_ in classes:
        filtered = funcy.lfilter(lambda t: t[0] == cls_, zipped)

        cls_targets = [t[1] for t in filtered]
        cls_contours = [t[2] for t in filtered]
        cls_mean, cls_std = evaluate_model(cls_targets, cls_contours)

        results[cls_] = (cls_mean, cls_std)

    return results
