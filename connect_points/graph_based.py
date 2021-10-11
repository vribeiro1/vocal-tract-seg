import pdb

import cv2
import funcy
import heapq
import math
import numpy as np
import statistics

from collections import defaultdict
from numba import jit


@jit(nopython=True)
def euclidean(u, v):
    x_u, y_u = u
    x_v, y_v = v

    return np.sqrt((x_u - x_v) ** 2 + (y_u - y_v) ** 2)


def point_deserialize(pt_str):
    x, y = pt_str.split()
    return int(x), int(y)


def point_serialize(pt):
    return " ".join(map(str, pt))


def uint16_to_uint8(image):
    max_val = np.amax(image)
    image = image.astype(float) * 255. / max_val
    image = image.astype(np.uint8)
    image = cv2.equalizeHist(image)
    image = image.astype(np.uint8)
    return image


def shortest_path(edges, source, sink):
    """
    This function is from https://gist.github.com/hanfang/89d38425699484cd3da80ca086d78129
    """
    # create a weighted DAG - {node:[(cost,neighbour), ...]}
    graph = defaultdict(list)

    for pt_left, pt_right, cost in edges:
        graph[pt_left].append((cost, pt_right))

    # create a priority queue and hash set to store visited nodes
    queue = [(0, source, [])]
    heapq.heapify(queue)
    visited = set()

    # traverse graph with BFS
    while queue:
        (cost, node, path) = heapq.heappop(queue)

        # visit the node if it was not visited before
        if node not in visited:
            visited.add(node)
            path = path + [node]

            # hit the sink
            if node == sink:
                return cost, path

            # visit neighbours
            for c, neighbour in graph[node]:
                if neighbour not in visited:
                    heapq.heappush(queue, (cost + c, neighbour, path))

    return np.inf, []


# We take all the contour points around the center of mass, then we sort them and
# detect a maximal gap. We think that the points separated by this gap are the tails,
# but it can be not the case in many cases. Pay attention.
def detect_tails(point, other_points):
    if len(other_points) == 0:
        return (), (), None

    x, y = point
    angles_and_points = sorted([
        (
            math.atan2(other_y - y, other_x - x),
            (other_x, other_y)
        )
        for other_x, other_y in other_points
    ])

    max_gap = 0
    for i, (angle, point) in enumerate(angles_and_points):
        angle_m1, point_m1 = angles_and_points[i - 1]
        angle_distance = angle - angle_m1

        if angle_distance > max_gap:
            max_gap = angle_distance
            pt1 = point_m1
            pt2 = point

    if len(angles_and_points) < 2:
        return other_points

    if max_gap == 0:
        _, pt1 = angles_and_points[0]
        _, pt2 = angles_and_points[1]
        dist = int(euclidean(pt1, pt2))

        return pt1, pt2, dist

    dist_last, _ = angles_and_points[-1]
    dist_first, _ = angles_and_points[0]
    if (2 * math.pi - dist_last + dist_first) > max_gap:
        max_gap = 2 * math.pi - dist_last + dist_first
        _, pt1 = angles_and_points[-1]
        _, pt2 = angles_and_points[0]

    dist = int(euclidean(pt1, pt2))

    return pt1, pt2, dist


def detect_extremities_on_axis(contour_points, axis=0):
    """
    Detect the extremities of the contour probability map in a given axis.

    Args:
    contour_points (List[Tuple[int, int]]): List of points that belong to the contour.
    axis (int): 0 for the x-axis, 1 for the y-axis.
    """
    if len(contour_points) == 0:
        return None, None

    points = np.array(contour_points)
    pt1 = min(points, key=lambda p: p[axis])
    pt2 = max(points, key=lambda p: p[axis])

    return pt1, pt2


def construct_graph(contour, mask, r, alpha, beta, gamma, prev_contour=None):
    edges = []
    for pt in contour:
        other_points = []

        x, y = pt
        for ix in range(x - r, x + r):
            for iy in range(y - r, y + r):
                if ix == x and iy == y:
                    continue

                mask_H, mask_W = mask.shape
                if (
                    ix < 0 or
                    ix > mask_W - 1 or
                    iy < 0 or
                    iy > mask_H - 1
                ):
                    continue

                P = mask[iy][ix]
                if P:
                    other_points.append((ix, iy))

                    weight_to = euclidean((x, y), (ix, iy)) ** 4
                    weight_intensity = 1 - P
                    distance_to_point = lambda other_pt: euclidean((ix, iy), other_pt)
                    weight_prev_contour = min(map(distance_to_point, prev_contour)) if prev_contour else 0

                    weight = alpha * weight_to + beta * weight_intensity + gamma * weight_prev_contour
                    edges.append((point_serialize((x, y)), point_serialize((ix, iy)), weight))

    return edges


def find_contour_points(img, thr_dist_factor=2.):
    locations = cv2.findNonZero(img)
    if locations is None:
        return [], ()

    # Construct a list with all points
    points = np.squeeze(locations, axis=1)

    # Center of mass
    cm_x, cm_y = points.sum(axis=0) / len(points)
    cm = cm_x, cm_y

    # Calculate all distances to the center of the mass and their median
    distances_to_CM = funcy.lmap(lambda pt: euclidean(pt, cm), points)
    median_dist = statistics.median(distances_to_CM)

    # If the point is not farther than the threshold, count that this is not an outlier
    contour = np.array([
        pt for pt, dist in zip(points, distances_to_CM)
        if dist < thr_dist_factor * median_dist
    ])

    # Center of mass
    cm_x, cm_y = contour.sum(axis=0) / len(points)
    cm = cm_x, cm_y

    return contour, cm


def connect_points_graph_based(mask, contour, r, alpha, beta, gamma, tails, prev_contour=None):
    if len(contour) == 0:
        return [], (), ()

    # Distance betweent the tails is decreased by 1, to not connect the
    # tails, but connect for sure all another points
    edges = construct_graph(contour, mask, r - 1, alpha, beta, gamma, prev_contour=prev_contour)

    source, sink = map(point_serialize, tails)
    cost, path = shortest_path(edges, source, sink)
    path = funcy.lmap(point_deserialize, path)

    if cost == np.inf:
        return [], (), ()

    return np.array(path), source, sink
