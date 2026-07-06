import math

import cv2
import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def optimal_font_dims(img, font_scale=1.5e-3, thickness_scale=3e-3):
    h, w = img.shape[:2]
    font_scale = min(w, h) * font_scale
    thickness = math.ceil(min(w, h) * thickness_scale)
    return font_scale, thickness


class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for MplColorHelper")
        self.cmap_name = cmap_name
        self.start_val = start_val
        self.stop_val = stop_val
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb_float(self, val):
        return self.scalarMap.to_rgba(val)

    def get_bgr(self, val):
        rgb = np.multiply(self.get_rgb_float(val)[:3], 255)
        return rgb[::-1]

    def get_bgr_abs(self, val):
        normalized_val = val / (self.stop_val - self.start_val)
        rgb = np.multiply(self.cmap(normalized_val)[:3], 255)
        return rgb[::-1]


def to_symmetry_lines(predictions):
    angles, midpoints, segment_lengths, _ = predictions
    axis = np.zeros((len(angles), 2, 2))

    for i in range(len(angles)):
        ag = angles[i]  # 0 - pi (radians)
        mp = midpoints[i]  # x, y
        sl = segment_lengths[i]
        # coordinate system of OpenCV (the y-axis increases downwards)
        p1 = mp + sl / 2 * np.asarray([math.cos(ag), -math.sin(ag)])
        p2 = mp - sl / 2 * np.asarray([math.cos(ag), -math.sin(ag)])
        axis[i, 0] = p1  # x, y
        axis[i, 1] = p2

    return axis


def visualize_predictions(img, predictions):
    """Draw detected symmetry lines on an image.

    Args:
        img: BGR image
        predictions: tuple of (angles, midpoints, segment_lengths, strengths)

    Returns:
        BGR image with symmetry lines drawn
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization")

    out = img.copy()

    sym_lines = to_symmetry_lines(predictions)
    midpoints = predictions[1]
    strengths = predictions[3]

    cmap = MplColorHelper("rainbow", 0, len(sym_lines))
    linewidth_max = np.mean(img.shape[:2]) * 0.005

    for i, (p1, p2) in enumerate(sym_lines):
        mp = midpoints[i]
        strength = strengths[i]

        bgr_color = cmap.get_bgr(i)
        col = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
        linewidth = int(max(linewidth_max - (i*linewidth_max/4), 2))
        font_scale, thickness = optimal_font_dims(out)

        out = cv2.line(out, (int(p1[0]), int(p1[1])), (int(
            p2[0]), int(p2[1])), col, linewidth)
        out = cv2.circle(out, (int(mp[0]), int(
            mp[1])), radius=5, color=col, thickness=-1)

        p = p1 if i % 2 == 0 else p2
        out = cv2.putText(out, f"{i+1}: {strength:.2f}", (int(p[0]), int(
            p[1])), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), int(thickness*2))
        out = cv2.putText(out, f"{i+1}: {strength:.2f}", (int(p[0]), int(
            p[1])), cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, thickness)

    return out
