import cv2
import math
import numpy as np

from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial.distance import euclidean


def MSD(target, pred):
    curveHuman = cv2.findNonZero(255 - target)
    curveModel = cv2.findNonZero(255 - pred)

    if curveModel is None:
        return -1.

    MSD = 0
    for ptModel in curveModel:
        minDist = target.shape[0] * target.shape[1]
        for ptHuman in curveHuman:
            currDist = math.sqrt((ptHuman[0][0] - ptModel[0][0])**2 + (ptHuman[0][1] - ptModel[0][1])**2)
            if currDist < minDist:
                minDist = currDist
        MSD += minDist

    for ptHuman in curveHuman:
        minDist = target.shape[0] * target.shape[1]
        for ptModel in curveModel:
            currDist = math.sqrt((ptHuman[0][0] - ptModel[0][0])**2 + (ptHuman[0][1] - ptModel[0][1])**2)
            if currDist < minDist:
                minDist = currDist
        MSD += minDist

    return MSD / (len(curveHuman) + len(curveModel))


def MSD_fast(target, pred) :
    MSD = 0
    n_el = 0

    dist_transform = distance_transform_edt(pred)
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            if target[i][j] == 0:
                MSD += dist_transform[i][j]
                n_el += 1

    dist_transform = distance_transform_edt(target)
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if pred[i][j] == 0:
                MSD += dist_transform[i][j]
                n_el += 1

    return MSD / n_el


def p2cp(i, u, v):
    ui = u[i]
    ui2cp = min(euclidean(ui, vj) for vj in v)
    return ui2cp


def p2cp_mean(u_, v_):
    n = len(u_)
    m = len(v_)

    dist_mtx = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist_mtx[i][j] = euclidean(u_[i], v_[j])

    u2cv = dist_mtx.min(axis=1)
    v2cu = dist_mtx.min(axis=0)
    mean_p2cp = (sum(u2cv) + sum(v2cu)) / (n + m)

    return mean_p2cp


def evaluate_model(targets, contours):
    targets = np.squeeze(targets)

    p2cps = []
    for target, predicted in zip(targets, contours):
        x_targets, y_targets = np.where(target == 255)
        target_points = list(zip(x_targets, y_targets))

        x_preds, y_preds = np.where(predicted == 255)
        preds_points = list(zip(x_preds, y_preds))

        mean_p2cp = p2cp_mean(target_points, preds_points)
        p2cps.append(mean_p2cp)

    mean = np.mean(p2cps)
    sigma = np.std(p2cps)

    return mean, sigma
