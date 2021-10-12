import pdb

import cv2
import funcy
import numpy as np

from copy import deepcopy
from scipy.ndimage import binary_fill_holes
from scipy.signal import convolve2d
from skimage.measure import regionprops, find_contours
from skimage.morphology import skeletonize

from .active_contours import (
    connect_with_active_contours,
    get_open_initial_curve,
    get_closed_initial_curve
)
from .cfg import *
from .graph_based import (
    connect_points_graph_based,
    detect_tails,
    detect_extremities_on_axis,
    find_contour_points
)
from .skeleton import (
    connect_with_skeleton,
    skeleton_sort
)


def rescale_contour(contour, s_in, s_out):
    r = s_out / s_in
    return np.array(contour) * r


def threshold_array(arr_, threshold, high=1., low=0.):
    arr = arr_.copy()
    arr[arr <= threshold] = low
    arr[arr > threshold] = high

    return arr


def contour_bbox_area(contour):
    x0 = contour[:, 0].min()
    y0 = contour[:, 1].min()
    x1 = contour[:, 0].max()
    y1 = contour[:, 1].max()

    return (x1 - x0) * (y1 - y0)


def calculate_contours_with_graph(
    mask, threshold, r, alpha, beta, gamma, delta, articulator, G, gravity_curve, **kwargs
):
    if articulator != TONGUE:
        mask_thr = threshold_array(mask, threshold)

        if articulator == SOFT_PALATE:
            mask_thr = binary_fill_holes(mask_thr)

        mask_thr = skeletonize(mask_thr).astype(np.uint8)
    else:
        mask_thr = mask.copy()
        mask_thr[mask_thr <= threshold] = 0.

    contour_points, c = find_contour_points(mask_thr)
    if articulator in (SOFT_PALATE, VOCAL_FOLDS):
        source, sink = detect_extremities_on_axis(contour_points, axis=0)
    elif articulator in (PHARYNX, EPIGLOTTIS):
        source, sink = detect_extremities_on_axis(contour_points, axis=1)
    else:
        source, sink, _ = detect_tails(c, contour_points)

    if source is None or sink is None:
        return []

    contour, _, _ = connect_points_graph_based(
        mask_thr,
        contour_points,
        r, alpha, beta, gamma, delta,
        tails=(source, sink),
        G=G,
        gravity_curve=gravity_curve
    )

    return contour


def calculate_contours_with_active_contours(
    mask, threshold, alpha, beta, gamma, articulator, **kwargs
):
    mask_thr = threshold_array(mask, threshold).copy()

    if articulator in (UPPER_LIP, LOWER_LIP, TONGUE, SOFT_PALATE):
        init = get_open_initial_curve(mask_thr, articulator)
        boundary_condition = "fixed"
    elif articulator in (ARYTENOID_MUSCLE, HYOID_BONE, THYROID_CARTILAGE, VOCAL_FOLDS):
        init = get_closed_initial_curve(mask_thr)
        boundary_condition = "periodic"
    else:
        raise NotImplemented

    if init is None:
        return []

    contour = connect_with_active_contours(
        mask_thr, init, alpha, beta, gamma,
        max_iter=500, boundary_condition=boundary_condition
    )

    return contour


def calculate_contours_with_skeleton(mask, threshold, **kwargs):
    mask_thr = threshold_array(mask, threshold)
    contour = connect_with_skeleton(mask_thr)

    return contour


def calculate_contours_with_border_method(mask, threshold, **kwargs):
    mask_thr = threshold_array(mask, threshold)
    props = regionprops(mask_thr.astype(np.uint8))
    y0, x0, y1, x1 = props[0]["bbox"]

    # The convolutional kernel counts the number of white pixels in the neighboorhood.
    conv_kernel = np.array([
        [1., 1., 1.],
        [1., 0., 0.],
        [1., 1., 1.]
    ])

    margin = 3
    region_mask_arr = mask_thr[y0 - margin:y1 + margin, x0 - margin:x1 + margin]

    region_conv_mask_arr = convolve2d(conv_kernel, region_mask_arr)
    region_conv_mask_arr = region_conv_mask_arr[1:-1, 1:-1]

    conv_mask_arr = np.zeros_like(mask_thr)
    conv_mask_arr[y0 - margin:y1 + margin, x0 - margin: x1 + margin] = region_conv_mask_arr

    border_mask_arr = conv_mask_arr.copy()

    # If convolution value is maximum, all neighboors are white, on the other side, if the
    # convolution value is minimum, all neighbors are black. In both cases, the pixel is not on
    # the border. Thus, we set them as zero. Pixels with an intermediate convolution value
    # are set to one.

    max_val = conv_kernel.sum()
    border_mask_arr[border_mask_arr == max_val] = 0
    border_mask_arr[(border_mask_arr > 0) & (border_mask_arr < max_val)] = 1

    skeleton = skeletonize(border_mask_arr)
    y, x = np.where(skeleton == True)
    unsorted_points = list(zip(x, y))

    if len(unsorted_points) == 0:
        return []

    origin = unsorted_points[0]
    contour = np.array(skeleton_sort(unsorted_points, origin))

    return contour


def calculate_contours_with_skimage(mask, threshold, **kwargs):
    mask_thr = threshold_array(mask, threshold)
    contours = sorted(find_contours(mask_thr), key=contour_bbox_area, reverse=True)

    if len(contours) == 0:
        return []

    contour = np.flip(contours[0])
    return contour


def upscale_mask(mask_, upscale):
    mask = mask_.copy()

    if upscale is None:
        return mask, lambda x: x

    s_out, _ = mask.shape
    s_in = upscale
    mask = cv2.resize(mask, (upscale, upscale), interpolation=cv2.INTER_CUBIC)
    rescale_contour_fn = funcy.partial(rescale_contour, s_in=s_in, s_out=s_out)

    return mask, rescale_contour_fn


def _calculate_contour_threshold_loop(
    post_processing_fn, mask, threshold, articulator, alpha, beta, gamma,
    G=0.0, gravity_curve=None
):
    min_length = 10
    min_threshold = 0.00005
    contour = []

    while len(contour) < min_length:
        contour = post_processing_fn(
            mask=mask,
            threshold=threshold,
            articulator=articulator,
            r=3,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            G=G,
            gravity_curve=gravity_curve
        )

        threshold = threshold * 0.9
        if threshold < min_threshold:
            break

    return contour


def calculate_contour(articulator, mask, gravity_curve=None):
    if articulator not in POST_PROCESSING:
        raise KeyError(
            f"Class '{articulator}' does not have post-processing parameters configured"
        )

    post_proc = POST_PROCESSING[articulator]
    if post_proc.method not in METHODS:
        raise ValueError(f"Unavailable post-processing method '{post_proc.method}'")

    max_upscale_iter = post_proc.max_upscale_iter
    for i in range(1, max_upscale_iter + 1):
        upscale = i * post_proc.upscale

        # If we upscale before post-processing, we need to define a function to rescale the
        # generated contour. Else, we use an identity function as a placeholder.
        new_mask, rescale_contour_fn = upscale_mask(mask, upscale)

        post_processing_fn = METHODS[post_proc.method]
        contour = _calculate_contour_threshold_loop(
            post_processing_fn,
            new_mask,
            post_proc.threshold,
            articulator,
            post_proc.alpha,
            post_proc.beta,
            post_proc.gamma,
            post_proc.G,
            gravity_curve
        )

        if len(contour) > 10:
            break

    if np.isnan(contour).any():
        raise Exception("contour has nan")
    contour = rescale_contour_fn(contour)

    return contour


METHODS = {
    GRAPH_BASED: calculate_contours_with_graph,
    ACTIVE_CONTOURS: calculate_contours_with_active_contours,
    SKELETON: calculate_contours_with_skeleton,
    BORDER_METHOD: calculate_contours_with_border_method,
    SKIMAGE: calculate_contours_with_skimage
}
