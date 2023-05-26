import cv2
import funcy
import numpy as np
import os
import pydicom
import roifile

from glob import glob
from scipy import interpolate
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import regionprops
from vt_tools.bs_regularization import regularize_Bsplines
from vt_tracker.visualization import uint16_to_uint8

from helpers import sequences_from_dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def remove_duplicates(iterable):
    seen = set()
    return [item for item in iterable if not (item in seen or seen.add(item))]


def roi_to_target_tensor(roi, size):
    points = roi.coordinates()
    x = points[:, 0]
    y = points[:, 1]

    tck, _ = interpolate.splprep([x, y], s=0)
    unew = np.arange(0, 1, 0.001)
    out_y, out_x = interpolate.splev(unew, tck)

    res_x = funcy.lmap(lambda v: int(round(v)), out_x)
    res_y = funcy.lmap(lambda v: int(round(v)), out_y)

    target_arr = np.zeros(size)
    target_arr[res_x, res_y] = 1.0

    return target_arr


def assert_expression(expression, exception=Exception, message=""):
    if not expression:
        raise exception(message)


def moving_average(arr, window_size):
    ma_arr = []
    for i in range(len(arr)):
        lwindow = window_size // 2
        rwindow = window_size // 2
        if window_size % 2 != 0:
            rwindow += 1

        min_idx = max(0, i - lwindow)
        max_idx = min(len(arr), i + rwindow)

        window = arr[min_idx:max_idx]
        ma_arr.append(np.mean(window))

    return np.array(ma_arr)


def subtraction(arr1, arr2, reduction=None):
    assert_expression(arr1.shape == arr2.shape, ValueError, "arr1 and arr2 should have the same shape")

    if reduction is None:
        reduction = lambda arr: arr

    sub = np.abs(arr1 - arr2)
    return reduction(sub)


def load_input_image(filepath):
    _, ext = os.path.basename(filepath).split(".")
    if ext == "dcm":
        pixel_arr = pydicom.dcmread(filepath).pixel_array
    elif ext == "npy":
        pixel_arr = np.load(filepath)
    else:
        raise Exception()

    uint8_pixel_arr = uint16_to_uint8(pixel_arr, norm_hist=False)
    return uint8_pixel_arr / 255


def load_segmentation_mask(filepath, size):
    filename, _ = os.path.basename(filepath).split(".")
    roi = roifile.roiread(filepath)
    mask = binary_fill_holes(roi_to_target_tensor(roi, size)).astype(np.uint8)

    props = regionprops(mask)
    y0, x0, y1, x1 = props[0]["bbox"]

    return mask, (x0, y0, x1, y1)


def get_sequence_reference_mask(
    database,
    datadir,
    subject,
    incisor,
    margin=1
):
    ref_mask_filepath = funcy.first(
        glob(os.path.join(
            BASE_DIR,
            "data",
            "ref_masks",
            database,
            subject,
            f"S*_{incisor}.roi"
        ))
    )
    ref_sequence, _ = os.path.basename(ref_mask_filepath).split("_")
    ref_filepath = os.path.join(datadir, subject, ref_sequence, "NPY_MR", "0001.npy")

    ref_arr = load_input_image(ref_filepath)
    _, bbox = load_segmentation_mask(ref_mask_filepath, ref_arr.shape)

    x0, y0, x1, y1 = bbox
    mask = ref_arr[y0 - margin:y1 + margin, x0 - margin:x1 + margin]
    return mask


def draw_incisor(e0, control_params):
    x0, y0 = e0

    keypoints = []
    for xi, yi in control_params:
        x = x0 + xi
        y = y0 + yi
        keypoints.append((x, y))

    keypoints = np.array(keypoints)
    reg_x, reg_y = regularize_Bsplines(keypoints, degree=2)
    reg_keypoints = np.array([reg_x, reg_y]).T

    return reg_keypoints
