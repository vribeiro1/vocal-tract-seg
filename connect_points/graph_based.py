import pdb

import cv2
import heapq
import math
import numpy as np
import statistics

from collections import defaultdict
from scipy.spatial.distance import euclidean
from skimage.morphology import skeletonize


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # 4-th power of distance, because linear distance will force the algorithm
    # to connect points which are far from each other, skiping another points
    def weightTo(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**2

    def distanceTo(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def __lt__(self, other):
        return True


def name_to_pt(name):
    coordList = name.split()
    return (int(coordList[0]), int(coordList[1]))


def pt_to_name(pt):
    return (str(pt[0]) + " " + str(pt[1]))


def calc_min_dist(pt, contour):
    l2 = 10000000
    for pt_cont in contour:
        l2_new = (pt.x - pt_cont[0])**2 + (pt.y - pt_cont[1])**2
        if l2_new < l2:
            l2 = l2_new
    return np.sqrt(l2)


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
    for l, r, c in edges:
        graph[l].append((c,r))
    # create a priority queue and hash set to store visited nodes
    queue, visited = [(0, source, [])], set()
    heapq.heapify(queue)
    # traverse graph with BFS
    while queue:
        (cost, node, path) = heapq.heappop(queue)
        # visit the node if it was not visited before
        if node not in visited:
            visited.add(node)
            path = path + [node]
            # hit the sink
            if node == sink:
                return (cost, path)
            # visit neighbours
            for c, neighbour in graph[node]:
                if neighbour not in visited:
                    heapq.heappush(queue, (cost+c, neighbour, path))
    return float("inf")


# We take all the contour points around the center of mass, then we sort them and
# detect a maximal gap. We think that the points separated by this gap are the tails,
# but it can be not the case in many cases. Pay attention.
def detect_tails(point, otherPoints):
    if len(otherPoints) == 0:
        return (), (), None

    anglesAndPoints = []
    for otherPoint in otherPoints:
        angle = math.atan2(otherPoint.y - point.y, otherPoint.x - point.x)
        anglesAndPoints.append([angle, otherPoint])

    anglesAndPoints.sort()
    maxGap = 0
    for i in range(1, len(anglesAndPoints)):
        angleDistance = anglesAndPoints[i][0] - anglesAndPoints[i - 1][0]
        if angleDistance > maxGap:
            maxGap = angleDistance
            pt1 = anglesAndPoints[i - 1][1]
            pt2 = anglesAndPoints[i][1]

    if len(anglesAndPoints) < 2:
      return otherPoints

    if maxGap == 0:
       return [anglesAndPoints[0][1], anglesAndPoints[1][1]]

    if (2 * math.pi - anglesAndPoints[-1][0] + anglesAndPoints[0][0]) > maxGap:
       maxGap = 2 * math.pi - anglesAndPoints[-1][0] + anglesAndPoints[0][0]
       pt1 = anglesAndPoints[-1][1]
       pt2 = anglesAndPoints[0][1]

    dist = int(pt1.distanceTo(pt2))

    return (pt1.x, pt1.y), (pt2.x, pt2.y), dist


def detect_extremities_on_axis(contour_points, axis=0):
    if len(contour_points) == 0:
        return None, None

    points = np.array([(p.x, p.y) for p in contour_points])
    pt1 = min(points, key=lambda p: p[axis])
    pt2 = max(points, key=lambda p: p[axis])

    return pt1, pt2


def gravity(G, m1, m2, r):
    return G * (m1 * m2) / (r ** 2)


def construct_graph(contour, mask, r, alpha, beta, gamma, prev_contour=None, G=0.0, gravity_curve=None):
    if prev_contour is None:
        prev_contour = []

    edges = []
    for pt in contour:
        otherPoints = []
        for ix in range(pt.x - r, pt.x + r):
            for iy in range(pt.y - r, pt.y + r):
                if (ix < 0 or iy < 0 or ix > mask.shape[1] - 1 or iy > mask.shape[0] - 1 or (ix == pt.x and iy == pt.y)):
                    continue
                if mask[iy][ix]:
                    P = mask[iy][ix]
                    if prev_contour == []:
                        R = 0
                    else:
                        R = calc_min_dist(Point(ix, iy), prev_contour)
                    otherPoints.append(Point(ix, iy))

                    if gravity_curve is not None:
                        m1 = 1.
                        m2 = 1.
                        r = min([euclidean((ix, iy), (x_g, y_g)) for x_g, y_g in gravity_curve])
                        weight_gravity = gravity(G, m1, m2, r)
                    else:
                        weight_gravity = 0.0

                    weight = alpha * pt.weightTo(Point(ix, iy)) + beta * (1 - P) + gamma * R + weight_gravity
                    edges.append([pt_to_name((pt.x, pt.y)), pt_to_name((ix, iy)), weight])
    return edges


def find_contour_points(img, threshDistFactor=2):
    locations = cv2.findNonZero(img)
    if locations is None:
        return [], []

    # Construct a list with all points
    points = []
    for i in range(0, locations.shape[0]):
        points.append(Point(locations[i][0][0], locations[i][0][1]))
    # Center of the mass
    cx = 0
    cy = 0
    for pt in points:
        cx += pt.x
        cy += pt.y
    cx /= len(points)
    cy /= len(points)
    c = Point(cx, cy)

    # Calculate all distances to the center of the mass and their median
    dist = []
    for pt in points:
        dist.append(pt.distanceTo(c))
    medDist = statistics.median(dist)

    # If the point is not farther than the threshold, count that this is not
    # an outlier
    contour = []
    for pt in points:
        if pt.distanceTo(c) < threshDistFactor * medDist:
            contour.append(pt)

    # Center of the mass
    cx = 0
    cy = 0
    for pt in contour:
        cx += pt.x
        cy += pt.y
    cx /= len(points)
    cy /= len(points)
    c = Point(cx, cy)

    return contour, c


def connect_points_graph_based(mask, contour, r, alpha, beta, gamma, tails, prev_contour=None, G=0.0, gravity_curve=None):
    if prev_contour is None:
        prev_contour = []

    if len(contour) == 0:
        return [], (), ()

    # Distance betweent the tails is decreased by 1, to not connect the
    # tails, but connect for sure all another points
    edges = construct_graph(
        contour,
        mask,
        r - 1,
        alpha,
        beta,
        gamma,
        prev_contour=prev_contour,
        G=G,
        gravity_curve=gravity_curve
    )

    source, sink = tails
    names = shortest_path(edges, pt_to_name(source), pt_to_name(sink))
    if type(names) is float:
        return [], [], []

    names = names[1]
    path = []
    for name in names:
        pt = name_to_pt(name)
        path.append(pt)

    return np.array(path), source, sink
