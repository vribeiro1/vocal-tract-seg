import cv2
import funcy
import numpy as np

from scipy.ndimage import binary_fill_holes

from .active_contours import connect_with_active_contours
from .cfg import *
from .graph_based import connect_points_graph_based
from .skeleton import connect_with_skeleton


def rescale_contour(contour, s_in, s_out):
    r = s_out / s_in
    return np.array(contour) * r


def threshold_array(arr_, threshold, high=1., low=0.):
    arr = arr_.copy()
    arr[arr <= threshold] = low
    arr[arr > threshold] = high

    return arr


def calculate_contours_with_graph(mask, threshold, r, alpha, beta, gamma, pred_class, **kwargs):
    mask_thr = threshold_array(mask, threshold)

    if pred_class == SOFT_PALATE:
        mask_thr = binary_fill_holes(mask_thr)

    contour, _, _ = connect_points_graph_based(mask_thr, r, alpha, beta, gamma)

    return contour


def calculate_contours_with_active_contours(mask, threshold, alpha, beta, gamma, pred_class, **kwargs):
    mask_ = mask.copy()
    mask_thr = threshold_array(mask, threshold)
    contour = connect_with_active_contours(mask, mask_thr, pred_class, alpha, beta, gamma, max_iter=500)

    return contour


def calculate_contours_with_skeleton(mask, threshold, **kwargs):
    mask_thr = threshold_array(mask, threshold)
    contour = connect_with_skeleton(mask_thr)

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


def _calculate_contour_threshold_loop(post_processing_fn, mask, threshold, pred_class, alpha, beta, gamma):
    min_length = 10
    min_threshold = 0.00005
    contour = []

    while len(contour) < min_length:
        contour = post_processing_fn(
            mask=mask,
            threshold=threshold,
            pred_class=pred_class,
            r=3,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )

        threshold = threshold * 0.9
        if threshold < min_threshold:
            break

    return contour


def calculate_contour(pred_class, mask):
    if pred_class not in POST_PROCESSING:
        raise KeyError(
            f"Class '{pred_class}' does not have post-processing parameters configured"
        )

    post_proc = POST_PROCESSING[pred_class]
    if post_proc.method not in METHODS:
        raise ValueError(f"Unavailable post-processing method '{post_proc.method}'")

    max_upscale_iter = post_proc.max_upscale_iter
    for i in range(1, max_upscale_iter + 1):
        upscale = i * post_proc.upscale

        # If we upscale before post-processing, we need to define a function to rescale the generated
        # contour. Else, we use an identity function as a placeholder.
        new_mask, rescale_contour_fn = upscale_mask(mask, upscale)

        post_processing_fn = METHODS[post_proc.method]
        contour = _calculate_contour_threshold_loop(
            post_processing_fn,
            new_mask,
            post_proc.threshold,
            pred_class,
            post_proc.alpha,
            post_proc.beta,
            post_proc.gamma
        )

        if len(contour) > 10:
            break

    contour = rescale_contour_fn(contour)
    if np.isnan(contour).any():
        raise Exception("contour has nan")

    return contour


METHODS = {
    GRAPH_BASED: calculate_contours_with_graph,
    ACTIVE_CONTOURS: calculate_contours_with_active_contours,
    SKELETON: calculate_contours_with_skeleton
}
