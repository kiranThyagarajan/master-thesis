import cv2
import numpy as np


def world_to_pixel(x, y, p, H=64, W=128):
    R = p["l1"] + p["l2"] # Max reach of the arm

    col = (x + R) / (2 * R) * (W - 1) # [-R, R] -Shift> [0, 2R] -Normalize> [0, 1] -Scale> [0, W-1]
    row = (1 - (y / R)) * (H - 1) # [0, R] -Normalize> [0, 1] -Invert> [1, 0] -Scale> [H-1, 0]

    return int(np.round(row)), int(np.round(col)) # pixel values are int


def render_markers(markers, p, H=64, W=128):
    """
    Render marker positions into a grayscale image.
    """
    image = np.zeros((H, W), dtype=np.uint8)

    for (x, y) in markers:
        row, col = world_to_pixel(x, y, p, H, W)
        cv2.circle(image, (col, row), radius=2, color=255, thickness=-1)

    return image
