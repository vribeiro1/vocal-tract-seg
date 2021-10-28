import pdb

import funcy
import numpy as np
import os
import torch

from copy import deepcopy
from PIL import Image, ImageDraw
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import skeletonize
from tqdm import tqdm
from vt_tracker import EPIGLOTTIS, SOFT_PALATE
from vt_tracker.metrics import p2cp_mean
from vt_tracker.postprocessing.calculate_contours import calculate_contour

from connect_points.active_contours import get_soft_palate_extremities
from helpers import draw_contour


def save_image_with_contour(img, filepath, contour, target=None):
    mask_contour = np.array(img)
    if target is not None:
        mask_contour[:, :, 0][target > 200] = 0
        mask_contour[:, :, 1][target > 200] = 0
        mask_contour[:, :, 2][target > 200] = 255
    mask_contour = draw_contour(mask_contour, contour, color=(255, 0, 0))

    mask_contour_img = Image.fromarray(mask_contour)
    mask_contour_img.save(filepath)


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


def evaluate_model(targets, contours):
    """
    Calculates the point-to-closest-point distance (in px) between two curves drawn in an image.
    """
    targets = np.squeeze(targets)

    p2cps = []
    for target, predicted in zip(targets, contours):
        x_targets, y_targets = np.where(target == 255)
        target_points = list(zip(x_targets, y_targets))

        x_preds, y_preds = np.where(predicted == 255)
        preds_points = list(zip(x_preds, y_preds))

        mean_p2cp = p2cp_mean(target_points, preds_points)
        p2cps.append(mean_p2cp)

    mean = np.mean(p2cps)
    sigma = np.std(p2cps)
    median = np.median(p2cps)

    return mean, sigma, median


def run_test(epoch, model, dataloader, outputs_dir, class_map, threshold=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_outputs_dir = os.path.join(outputs_dir, "inference", str(epoch))
    if not os.path.exists(epoch_outputs_dir):
        os.makedirs(epoch_outputs_dir)

    model.eval()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - inference")

    return_outputs = []
    for _, (info, inputs, targets_dict) in enumerate(progress_bar):
        inputs = inputs.to(device)
        targets_dict = [{
            k: v.to(device) for k, v in d.items()
        } for d in targets_dict]

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            for _, (im_info, im_outputs, targets_dict) in enumerate(zip(info, outputs, targets_dict)):
                targets = targets_dict["masks"]

                zipped = list(zip(
                    im_outputs["boxes"],
                    im_outputs["labels"],
                    im_outputs["scores"],
                    im_outputs["masks"]
                ))

                detected = []
                for c_idx, _ in class_map.items():
                    art_list = funcy.lfilter(lambda t: t[1].item() == c_idx, zipped)
                    if len(art_list) > 0:
                        art = max(art_list, key=lambda t: t[2])
                        detected.append(art + (targets[c_idx],))

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


def soft_palate_mask_to_center_line(mask_):
    mask = mask_.copy()

    ext1, ext2, ext3 = get_soft_palate_extremities(mask)
    x1, y1 = ext1
    x2, y2 = ext2

    mask[y2:y1+1, x1] = mask.max()
    filled_mask = binary_fill_holes(mask).astype(np.uint8)
    center_line = skeletonize(filled_mask) * 255

    return center_line


def run_evaluation(outputs, classes, save_to=None, load_fn=None):
    pred_classes = []
    targets = []
    contours = []
    for out in outputs:
        target = out["target"] * 255
        mask = out["mask"]
        pred_class = out["pred_cls"]

        if pred_class in [EPIGLOTTIS, SOFT_PALATE]:
            target = soft_palate_mask_to_center_line(target)

        contour = calculate_contour(pred_class, mask)

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
        cls_mean, cls_std, cls_median = evaluate_model(cls_targets, cls_contours)

        results[cls_] = (cls_mean, cls_std, cls_median)

    return results
