import numpy as np

from copy import deepcopy
from scipy.spatial.distance import euclidean
from skimage.morphology import skeletonize


def _remove_duplicates(iterable):
    seen = set()
    return [item for item in iterable if not (item in seen or seen.add(item))]


def skeleton_sort(points_, origin, max_dist=5):
    points = _remove_duplicates(deepcopy(points_))
    curr_p = origin

    sorted_points = []
    while len(points) > 0:
        next_p = min(points, key=lambda p: euclidean(p, curr_p))

        dist = euclidean(next_p, curr_p)
        if dist < max_dist or curr_p == origin:
            sorted_points.append(next_p)
            curr_p = next_p

        points.remove(next_p)

    return sorted_points


def connect_with_skeleton(mask):
    im_H, im_W = mask.shape

    skeleton = skeletonize(mask)
    y, x = np.where(skeleton == True)
    unsorted_points = list(zip(x, y))
    sorted_points = skeleton_sort(unsorted_points, (0, im_W))

    return np.array(sorted_points)
