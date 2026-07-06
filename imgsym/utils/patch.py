import cv2
import numpy as np


def rotate_image(image, angle):
    """Rotate image by angle degrees around its center."""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
    return rotated


def origin_2_target(origin, target_start, target_end):
    """Map origin coordinates to target line coordinates."""
    angle = np.arctan2(target_end[1] - target_start[1],
                       target_end[0] - target_start[0])
    angle_deg = np.degrees(angle)
    return angle_deg


def pad_image(img, pad_size):
    """Pad image with reflection."""
    return cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size,
                               cv2.BORDER_REFLECT_101)


def calc_percentage_based_width(image, line, percentage=0.3):
    """Calculate patch width as percentage of image dimension."""
    h, w = image.shape[:2]
    return max(int(min(h, w) * percentage), 10)


def extract_patch(img, line, width=None, percentage=0.3):
    """Extract a symmetric patch around a symmetry axis line.

    Args:
        img: Input image (BGR or grayscale)
        line: Tuple of ((x1, y1), (x2, y2)) defining the symmetry axis
        width: Width of patch on each side of the axis. If None, calculated as percentage.
        percentage: Percentage of image size to use as width (default 0.3)

    Returns:
        Patch image aligned so the symmetry axis is vertical center
    """
    line_start, line_end = line

    if width is None:
        width = calc_percentage_based_width(img, line, percentage)

    # Calculate angle of the line
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Rotate image so line becomes vertical
    rotation_angle = -(angle - 90)  # Make line vertical
    h, w_img = img.shape[:2]
    center = (w_img / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # Calculate new bounding box
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w_img * cos_a)
    new_h = int(h * cos_a + w_img * sin_a)
    M[0, 2] += (new_w - w_img) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    # Transform line midpoint to rotated coordinates
    mid_x = (line_start[0] + line_end[0]) / 2
    mid_y = (line_start[1] + line_end[1]) / 2
    mid_rot = M @ np.array([mid_x, mid_y, 1])

    # Extract patch centered on the rotated line
    cx = int(mid_rot[0])
    rh, rw = rotated.shape[:2]

    x1 = max(0, cx - width)
    x2 = min(rw, cx + width)

    patch = rotated[:, x1:x2]

    return patch
