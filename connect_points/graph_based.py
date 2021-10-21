import pdb

import cv2
import funcy
import heapq
import math
import numpy as np

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


def gravity(m1, m2, d):
    eps = 1e-3
    return -(m1 * m2) / ((d + eps) ** 4)


def calculate_edge_weights(
    curr_pt, other_pt, center_of_mass, other_pt_prob, alpha, beta, gamma, delta, G,
    prev_contour=None, gravity_curve=None
):
    """
    Calculates the weight for each edge in the graph. The edges' weights have five components,
    which are:
    1) The squared distance between the current and the next points;
    2) The probability of the next point;
    3) The minimal distance between the previous frame contour (if provided) and the next point;
    4) The angle between the current point and the next point to avoid the algorithm following the
    backwards direction.
    5) The distance between a reference curve (gravity_curve) and the next point;

    Args:
    curr_pt (Tuple[int, int]): Current node in the graph.
    other_pt (Tuple[int, int]): Next node in the graph.
    center_of_mass (Tuple[int, int]): Center of mass of the graph.
    other_pt_prob (float): Probability of the next node.
    alpha (float): Weight of the (1) component.
    beta (float): Weight of the (2) component.
    gamma (float): Weight of the (3) component.
    delta (float): Weight of the (4) component.
    G (float): Weight of the (5) component.
    prev_contour (List): Contour of the previous frame.
    gravity_curve (np.ndarray): Contour of the reference curve of the gravity algorithm.
    """
    x, y = curr_pt
    ix, iy = other_pt
    cm_x, cm_y = center_of_mass

    weight_to = euclidean((x, y), (ix, iy)) ** 2
    weight_intensity = 1 - other_pt_prob

    # If the contour of the previous frame is provided, this weight penalizes the
    # algorithm for finding curves that are too distant from the previous, avoiding
    # big jumps.
    distance_to_point = lambda other_pt: euclidean((ix, iy), other_pt)
    weight_prev_contour = min(
        map(distance_to_point, prev_contour)
    ) if prev_contour else 0

    # We want to follow a proper direction in the search for the contour.
    # Thus, we penalize the algorithm for going backwards.
    angle_curr_rad = math.atan2(y - cm_y, x - cm_x) - (np.pi / 2)
    if angle_curr_rad < 0:
        angle_curr_rad = 2 * np.pi - angle_curr_rad

    angle_other_rad = math.atan2(iy - cm_y, x - cm_x) - (np.pi / 2)
    if angle_other_rad < 0:
        angle_other_rad = 2 * np.pi - angle_other_rad

    weight_angle = int(angle_other_rad < angle_curr_rad)

    # Gravity weight penalizes the algorithm for selecting points farther from the
    # reference curve, which is useful for adjusting the contact between the tongue
    # and the alveolar region.
    weight_gravity = 0.0
    if gravity_curve is not None:
        min_xg, min_yg = gravity_curve.min(axis=0)
        max_xg, max_yg = gravity_curve.max(axis=0)

        # To reduce the number of evaluated points, exclude points that are not in the region of
        # the gravity curve since their weights would be close to zero anyway.
        if (min_xg - 5) < ix < (max_xg + 5) and min_yg < iy < (max_yg + 5):
            m1 = m2 = 1.
            d = min([euclidean((ix, iy), (x_g, y_g)) for x_g, y_g in gravity_curve])
            weight_gravity = gravity(m1, m2, d)

    weight = (
        alpha * weight_to +
        beta * weight_intensity +
        gamma * weight_prev_contour +
        delta * weight_angle +
        G * weight_gravity
    )
    return weight


def construct_graph(
    contour, mask, r, alpha, beta, gamma, delta,
    prev_contour=None, G=0.0, gravity_curve=None
):
    edges = []

    # Center of mass
    cm_x, cm_y = contour.sum(axis=0) / len(contour)

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

                    weight = calculate_edge_weights(
                        (x, y), (ix, iy), (cm_x, cm_y), P,
                        alpha, beta, gamma, delta, G,
                        prev_contour, gravity_curve
                    )
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
    median_dist = np.median(distances_to_CM)

    # If the point is not farther than the threshold, count that this is not an outlier
    contour = np.array([
        pt for pt, dist in zip(points, distances_to_CM)
        if dist < thr_dist_factor * median_dist
    ])

    if len(contour) == 0:
        return [], ()

    # Center of mass
    cm_x, cm_y = contour.sum(axis=0) / len(points)
    cm = cm_x, cm_y

    return contour, cm


def connect_points_graph_based(
    mask, contour, r, alpha, beta, gamma, delta, tails,
    prev_contour=None, G=0.0, gravity_curve=None
):
    if prev_contour is None:
        prev_contour = []

    # Distance betweent the tails is decreased by 1, to not connect the
    # tails, but connect for sure all another points
    edges = construct_graph(
        contour,
        mask,
        r - 1,
        alpha,
        beta,
        gamma,
        delta,
        prev_contour=prev_contour,
        G=G,
        gravity_curve=gravity_curve
    )

    source, sink = map(point_serialize, tails)
    cost, path = shortest_path(edges, source, sink)
    path = funcy.lmap(point_deserialize, path)

    if cost == np.inf:
        return [], (), ()

    return np.array(path), source, sink
