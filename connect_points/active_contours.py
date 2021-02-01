import pdb

import numpy as np

from skimage.measure import regionprops
from skimage.segmentation import active_contour

from .graph_based import find_contour_points, detect_tails


def get_box_from_mask_tensor(mask_, margin=0):
    mask = mask_.copy()
    mask_np = mask.astype(np.uint8)

    props = regionprops(mask_np)
    y0, x0, y1, x1 = props[0]["bbox"]
    return [x0 - margin, y0 - margin, x1 + margin, y1 + margin]


def get_general_extremities(mask_arr):
    contour, c = find_contour_points(mask_arr)
    ext1, ext2, _ = detect_tails(c, contour)

    box = get_box_from_mask_tensor(mask_arr)
    x0, y0, x1, y1 = box
    ext3 = (x0, y0)

    return ext1, ext2, ext3


def get_soft_palate_extremities(mask_arr, min_diff=4):
    h, w = mask_arr.shape
    box = get_box_from_mask_tensor(mask_arr)
    x0, y0, x1, y1 = box

    x0_arr = mask_arr[:, x0]
    indices = np.where(x0_arr == 1)[0]
    y_min = indices.min()
    y_max = indices.max()

    diff = y_max - y_min
    if diff < min_diff:
        y_min -= diff - min_diff
        y_max += diff - min_diff

    ext1 = (x0, y_min)
    ext2 = (x0, y_max)
    ext3 = (x1, y1)

    return ext1, ext2, ext3


def get_extremities(art, mask_arr):
    if art == "soft-palate":
        return get_soft_palate_extremities(mask_arr)
    else:
        return get_general_extremities(mask_arr)


def get_circle(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    x12 = x1 - x2
    y12 = y1 - y2

    x13 = x1 - x3
    y13 = y1 - y3

    x21 = x2 - x1
    y21 = y2 - y1

    x31 = x3 - x1
    y31 = y3 - y1

    sx13 = x1 ** 2 - x3 ** 2
    sy13 = y1 ** 2 - y3 ** 2

    sx21 = x2 ** 2 - x1 ** 2
    sy21 = y2 ** 2 - y1 ** 2

    f = (sx13 * x12 + sy13 * x12 + sx21 * x13 + sy21 * x13) / (2 * (y31 * x12 - y21 * x13))
    g = (sx13 * y12 + sy13 * y12 + sx21 * y13 + sy21 * y13) / (2 * (x31 * y12 - x21 * y13))
    c = - x1 ** 2 - y1 ** 2 - 2 * g * x1 - 2 * f * y1

    xc = -g
    yc = -f
    r = np.sqrt(xc ** 2 + yc **2 - c)

    return (xc, yc), r


def connect_with_active_contours(mask_, mask_thr_, art, alpha, beta, gamma, n_samples=400, max_iter=2500):
    mask = mask_.copy()
    mask_thr = mask_thr_.copy()

    ext1, ext2, ext3 = get_extremities(art, mask_thr)
    (x_c, y_c), radius = get_circle(ext1, ext2, ext3)

    x1, y1 = ext1
    sin1 = (y1 - y_c) / radius
    cos1 = (x1 - x_c) / radius
    theta1 = np.arctan2(sin1, cos1)

    x2, y2, = ext2
    sin2 = (y2 - y_c) / radius
    cos2 = (x2 - x_c) / radius
    theta2 = np.arctan2(sin2, cos2)

    if art != "soft-palate":
        theta1 += 2 * np.pi

    s = np.linspace(theta2, theta1, n_samples)
    r = y_c + radius * np.sin(s)
    c = x_c + radius * np.cos(s)
    init = np.array([r, c]).T

    snake = active_contour(
        mask,
        init,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        max_iterations=max_iter,
        boundary_condition="fixed"
    )

    contour = np.zeros_like(snake)
    contour[:, 0] = snake[:, 1]
    contour[:, 1] = snake[:, 0]

    if np.isnan(contour).any():
        pdb.set_trace()

    return contour
