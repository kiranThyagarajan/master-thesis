import cv2
import numpy as np


def world_to_pixel(x, y, p, H=64, W=128):
    R = p["l"]

    # x in [-R, R] maps to columns [0, W - 1]
    col = (x + R) / (2.0 * R) * (W - 1)

    # y in [0, R] maps to rows [H - 1, 0]
    row = (1.0 - y / R) * (H - 1)

    return int(np.round(row)), int(np.round(col))


def render_markers(markers, p, H=64, W=128):
    """
    Render marker positions into a grayscale image.
    """
    image = np.zeros((H, W), dtype=np.uint8)

    for x, y in markers:
        row, col = world_to_pixel(x, y, p, H, W)

        # Avoid passing an invalid pixel coordinate to OpenCV.
        if 0 <= row < H and 0 <= col < W:
            cv2.circle(
                image,
                (col, row),
                radius=2,
                color=255,
                thickness=-1,
            )

    return image