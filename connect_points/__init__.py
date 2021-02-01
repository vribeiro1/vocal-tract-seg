import cv2

from .active_contours import *
from .calculate_contours import *
from .graph_based import *
from .skeleton import *


def draw_contour(image, contour, color=(0, 0, 0)):
    for i in range(1, len(contour)):
        pt1 = (int(contour[i - 1][0]), int(contour[i - 1][1]))
        pt2 = (int(contour[i][0]), int(contour[i][1]))
        image = cv2.line(image, pt1, pt2, color, 1)
    return image
