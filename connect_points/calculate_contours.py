import cv2
import funcy
import numpy as np

from scipy.ndimage import binary_fill_holes
from skimage.morphology import skeletonize

from .active_contours import connect_with_active_contours, get_open_initial_curve, get_closed_initial_curve
from .cfg import *
from .graph_based import connect_points_graph_based, detect_tails, detect_extremities_on_axis, find_contour_points
from .skeleton import connect_with_skeleton


def rescale_contour(contour, s_in, s_out):
    r = s_out / s_in
    return np.array(contour) * r


def threshold_array(arr_, threshold, high=1., low=0.):
    arr = arr_.copy()
    arr[arr <= threshold] = low
    arr[arr > threshold] = high

    return arr


def calculate_contours_with_graph(mask, threshold, r, alpha, beta, gamma, articulator, **kwargs):
    mask_thr = threshold_array(mask, threshold)

    if articulator == SOFT_PALATE:
        mask_thr = binary_fill_holes(mask_thr)
    mask_thr = skeletonize(mask_thr).astype(np.uint8)

    contour, c = find_contour_points(mask_thr)
    if articulator in (SOFT_PALATE,):
        source, sink = detect_extremities_on_axis(mask_thr, axis=0)
    elif articulator in (PHARYNX,):
        source, sink = detect_extremities_on_axis(mask_thr, axis=1)
    else:
        source, sink, _ = detect_tails(c, contour)

    if source is None or sink is None:
        return []

    contour, _, _ = connect_points_graph_based(mask_thr, contour, r, alpha, beta, gamma, tails=(source, sink))

    return contour


def calculate_contours_with_active_contours(mask, threshold, alpha, beta, gamma, articulator, **kwargs):
    mask_ = mask.copy()
    mask_thr = threshold_array(mask, threshold).copy()

    if articulator in (UPPER_LIP, LOWER_LIP, TONGUE, SOFT_PALATE):
        init = get_open_initial_curve(mask_thr, articulator)
        boundary_condition = "fixed"
    elif articulator in (ARYTENOID_MUSCLE, HYOID_BONE, THYROID_CARTILAGE, VOCAL_FOLDS):
        init = get_closed_initial_curve(mask_thr)
        boundary_condition = "free"
    else:
        raise NotImplemented

    if init is None:
        return []

    contour = connect_with_active_contours(
        mask, init, alpha, beta, gamma,
        max_iter=500, boundary_condition=boundary_condition
    )

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


def _calculate_contour_threshold_loop(post_processing_fn, mask, threshold, articulator, alpha, beta, gamma):
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
            gamma=gamma
        )

        threshold = threshold * 0.9
        if threshold < min_threshold:
            break

    return contour


def calculate_contour(articulator, mask):
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

        # If we upscale before post-processing, we need to define a function to rescale the generated
        # contour. Else, we use an identity function as a placeholder.
        new_mask, rescale_contour_fn = upscale_mask(mask, upscale)

        post_processing_fn = METHODS[post_proc.method]
        contour = _calculate_contour_threshold_loop(
            post_processing_fn,
            new_mask,
            post_proc.threshold,
            articulator,
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
