import numpy as np
import cv2
import funcy

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


def calculate_contours_with_graph(mask, threshold, r, alpha, beta, gamma, **kwargs):
    mask_thr = threshold_array(mask, threshold)
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


def calculate_contour(pred_class, mask):
    if pred_class not in POST_PROCESSING:
        raise KeyError(
            f"Class '{pred_class}' does not have post-processing parameters configured"
        )

    post_proc = POST_PROCESSING[pred_class]
    # If we upscale before post-processing, we need to define a function to rescale the generated
    # contour. Else, we use an identity function as a placeholder.
    if post_proc.upscale is not None:
        s_out, _ = mask.shape
        s_in = post_proc.upscale
        mask = cv2.resize(mask, (post_proc.upscale, post_proc.upscale), interpolation=cv2.INTER_CUBIC)
        rescale_contour_fn = funcy.partial(rescale_contour, s_in=s_in, s_out=s_out)
    else:
        rescale_contour_fn = lambda x: x

    contour = []
    threshold_tmp = post_proc.threshold
    while len(contour) < 10:
        if post_proc.method not in METHODS:
            raise ValueError(f"Unavailable post-processing method '{post_proc.method}'")

        post_processing_fn = METHODS[post_proc.method]
        contour = post_processing_fn(
            mask=mask,
            threshold=threshold_tmp,
            pred_class=pred_class,
            r=3,
            alpha=post_proc.alpha,
            beta=post_proc.beta,
            gamma=post_proc.gamma
        )

        threshold_tmp = threshold_tmp - 0.05
        if threshold_tmp < 0.0:
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
