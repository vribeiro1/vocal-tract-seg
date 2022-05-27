import funcy
import numpy as np
import os
import torch

from copy import deepcopy
from PIL import Image, ImageDraw
from roifile import roiread
from tqdm import tqdm
from vt_tools.bs_regularization import regularize_Bsplines
from vt_tools.metrics import p2cp_mean
from vt_tracker.postprocessing.calculate_contours import calculate_contour

from dataset import VocalTractMaskRCNNDataset
from helpers import draw_contour


def load_articulator_array(filepath, norm_value=None):
    """
    Loads the target array with the proper orientation (right to left)

    Args:
    filepath (str): Path to the articulator array
    """
    articul_array = np.load(filepath)
    n_rows, _ = articul_array.shape
    if n_rows == 2:
        articul_array = articul_array.T

    # All the countors should be oriented from right to left. If it is the opposite,
    # we flip the array.
    if articul_array[0][0] < articul_array[-1][0]:
        articul_array = np.flip(articul_array, axis=0)

    if norm_value is not None:
        articul_array = articul_array.copy() / norm_value

    return articul_array



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


def evaluate_model(pred_classes, target_filepaths, pred_filepaths):
    results = []
    for pred_class, target_filepath, pred_filepath in zip(pred_classes, target_filepaths, pred_filepaths):
        target_array = roiread(target_filepath).coordinates()
        reg_x, reg_y = regularize_Bsplines(target_array.T, degree=2)
        reg_target_array = np.array([reg_x, reg_y]).T
        pred_array = load_articulator_array(pred_filepath)
        p2cp = p2cp_mean(pred_array, reg_target_array)

        if pred_class in VocalTractMaskRCNNDataset.closed_articulators:
            # Calculate the Jaccard Index
            jacc = 0
        else:
            jacc = np.nan

        basename = os.path.basename(target_filepath)
        dirname = os.path.dirname(os.path.dirname(target_filepath))
        sequence = os.path.basename(dirname)
        subject = os.path.basename(os.path.dirname(dirname))
        filename, _ = basename.split(".")
        frame, _ = filename.split("_")

        results.append(dict(
            subject=subject,
            sequence=sequence,
            frame=frame,
            pred_class=pred_class,
            target_filepath=target_filepath,
            pred_filepath=pred_filepath,
            p2cp_distance=p2cp,
            jaccard_index=jacc
        ))

    return results


def run_maskrcnn_test(epoch, model, dataloader, outputs_dir, class_map, threshold=None, device=None):
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


def run_deeplabv3_test(epoch, model, dataloader, outputs_dir, class_map, threshold=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_outputs_dir = os.path.join(outputs_dir, "inference", str(epoch))
    if not os.path.exists(epoch_outputs_dir):
        os.makedirs(epoch_outputs_dir)

    model.eval()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - inference")

    return_outputs = []
    for _, (info, inputs, targets) in enumerate(progress_bar):
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)["out"]
            outputs = torch.sigmoid(outputs)

            for _, (im_info, im_outputs, im_targets) in enumerate(zip(info, outputs, targets)):
                # detected: List of (bbox, label, score, output_mask, target_mask)
                detected = [(
                    (0, 0, 0, 0), label, 1.0, im_outputs[label], im_targets[label]
                ) for label, _ in class_map.items()]

                for box, label, score, mask, target in detected:
                    mask_arr = mask.squeeze(dim=0).cpu().numpy()
                    if threshold is not None:
                        mask_arr[mask_arr > threshold] = 1.
                        mask_arr[mask_arr <= threshold] = 0.
                    mask_arr = mask_arr * 255
                    mask_arr = mask_arr.astype(np.uint8)

                    mask_img = Image.fromarray(mask_arr)

                    mask_dirname = os.path.join(
                        epoch_outputs_dir,
                        im_info["subject"],
                        im_info["sequence"]
                    )
                    if not os.path.exists(mask_dirname):
                        os.makedirs(mask_dirname)

                    pred_cls = class_map[label]
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


def run_evaluation(outputs, save_to, load_fn):
    pred_classes = []
    target_filepaths = []
    pred_filepaths = []
    for out in outputs:
        subject = out["subject"]
        sequence = out["sequence"]
        input_img_filepath = out["img_filepath"]

        target = out["target"] * 255
        mask = out["mask"]
        pred_class = out["pred_cls"]

        frame = "%04d" % out["instance_number"]
        fname = f"{frame}_{pred_class}"

        outputs_dir = os.path.join(save_to, "inference_contours", subject, sequence)
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)

        contour = calculate_contour(pred_class, mask)

        img = load_fn(input_img_filepath)
        save_filepath = os.path.join(outputs_dir,f"{fname}.png")
        save_image_with_contour(img, save_filepath, contour, target)

        npy_filepath = os.path.join(outputs_dir, f"{fname}.npy")
        with open(npy_filepath, "wb") as f:
            np.save(f, contour)

        dirname = os.path.dirname(os.path.dirname(input_img_filepath))
        target_filepath = os.path.join(dirname, "contours", f"{fname}.roi")

        pred_classes.append(pred_class)
        target_filepaths.append(target_filepath)
        pred_filepaths.append(npy_filepath)

    results = evaluate_model(pred_classes, target_filepaths, pred_filepaths)

    return results


test_runners = {
    "maskrcnn": run_maskrcnn_test,
    "deeplabv3": run_deeplabv3_test
}
