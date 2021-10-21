import cv2
import numpy as np
import random
import torch


def set_seeds(worker_id):
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed + 1)
    random.seed(seed + 2)


def draw_contour(image, contour, color=(0, 0, 0)):
    for i in range(1, len(contour)):
        pt1 = (int(contour[i - 1][0]), int(contour[i - 1][1]))
        pt2 = (int(contour[i][0]), int(contour[i][1]))
        image = cv2.line(image, pt1, pt2, color, 1)
    return image
