"""imgsym - Image symmetry detection and measurement toolkit.

Provides methods for:
- Measuring symmetry scores along an axis (13+ methods)
- Detecting symmetry axes in images (4 methods) and binary shapes (1 method)
- Visualizing detected symmetry lines
"""

__version__ = "0.1.0"

from .scoring import calculate_symmetry_score, list_symmetry_methods, SymmetryCalculatorFactory

# Convenience aliases
score = calculate_symmetry_score
list_methods = list_symmetry_methods


def score_along_axis(image, axis, method="multi_scale_gradient", **kwargs):
    """Score symmetry along a specific axis.

    Args:
        image: Input image in BGR format (H x W x 3)
        axis: Tuple of ((x1, y1), (x2, y2)) defining the symmetry axis
        method: Scoring method name (see list_methods())
        **kwargs: Method-specific parameters

    Returns:
        float: Symmetry score (higher = more symmetric)
    """
    from .utils.patch import extract_patch

    patch = extract_patch(image, axis)
    return calculate_symmetry_score(patch, method=method, **kwargs)


def detect_axes(image, detector="xfeatures", max_axes=6, **kwargs):
    """Detect symmetry axes in a natural image.

    Args:
        image: Input image in BGR format (H x W x 3)
        detector: Detection method - "xfeatures", "nxc", or "wavelets"
        max_axes: Maximum number of axes to return
        **kwargs: Detector-specific parameters. nxc: boxSize, nBoxSamples,
            angleSet, multiprocessing. xfeatures / wavelets: debug.

    Returns:
        tuple: (angles, midpoints, segment_lengths, strengths)

    See also:
        detect_axes_from_mask() for binary shape inputs (r_lip).
    """
    if detector == "xfeatures":
        from .detection.xfeatures import calc_symmetry_lines
    elif detector == "nxc":
        from .detection.nxc import calc_symmetry_lines
    elif detector == "wavelets":
        from .detection.wavelets import calc_symmetry_lines
    else:
        raise ValueError(f"Unknown detector: {detector}. "
                         f"Available: xfeatures, nxc, wavelets. "
                         f"For binary shapes use detect_axes_from_mask().")

    return calc_symmetry_lines(image, maxNOutputs=max_axes, **kwargs)


def detect_axes_from_mask(mask, detector="r_lip", **kwargs):
    """Detect reflection-symmetry axes in a BINARY SHAPE mask.

    A different modality from detect_axes(): it operates on a binary shape rather
    than a natural image, and the detected axes pass through the shape centroid.
    The result uses the same 4-tuple contract as detect_axes(), so it flows into
    visualize() unchanged.

    Args:
        mask: Binary shape (H x W), nonzero = foreground
        detector: Mask detection method - "r_lip" (see list_mask_detectors())
        **kwargs: Detector-specific parameters. r_lip: signature ("LIP" or "R"),
            mode ("single" or "multi"), threshold, dist_metric

    Returns:
        tuple: (angles, midpoints, segment_lengths, measures)
    """
    if detector == "r_lip":
        return _detect_axes_r_lip(mask, **kwargs)
    raise ValueError(f"Unknown mask detector: {detector}. "
                     f"Available: {', '.join(list_mask_detectors())}. "
                     f"For natural images use detect_axes().")


def _detect_axes_r_lip(mask, signature="LIP", **kwargs):
    """r_lip (Nguyen et al., 2022): shape-signature reflection detection."""
    import numpy as np
    from .detection.r_lip import detect_reflection_sym, detect_centroid

    # Binarize to float {0.0, 1.0}: r_lip's radon profile is dtype-sensitive and
    # under-detects on uint8 {0, 1} masks (works on float / {0, 255}).
    mask = (np.asarray(mask) > 0).astype(np.float64)

    angles_deg, measures = detect_reflection_sym(mask, type_sig=signature, **kwargs)
    cy, cx = detect_centroid(mask)                      # (row, col) = (y, x)

    # r_lip axis at direction theta (deg) draws as [cos, -sin](deg2rad(theta+90))
    # through the centroid -- the same form to_symmetry_lines() expects in radians.
    angles = np.array([np.deg2rad(t + 90) % np.pi for t in angles_deg])
    midpoints = np.tile([cx, cy], (len(angles), 1)).astype(float)

    # per-axis segment length = extent of the shape projected onto that axis
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs - cx, ys - cy]).astype(float)   # centered (x, y-down)
    seg_lengths = np.zeros(len(angles))
    for i, ag in enumerate(angles):
        if pts.size:
            proj = pts @ np.asarray([np.cos(ag), -np.sin(ag)])
            seg_lengths[i] = float(proj.max() - proj.min())

    return angles, midpoints, seg_lengths, np.asarray(measures, dtype=float)


def get_axis(axes, index=0):
    """One detected axis as its segment endpoints — the form the scoring API uses.

    Args:
        axes: (angles, midpoints, seg_lengths, strengths) as returned by
            detect_axes() / detect_axes_from_mask()
        index: which axis (0 = strongest)

    Returns:
        ((x1, y1), (x2, y2)) — pass directly to score_along_axis().
    """
    import math

    angles, midpoints, seg_lengths, _ = axes
    cx, cy = float(midpoints[index][0]), float(midpoints[index][1])
    th, length = float(angles[index]), float(seg_lengths[index])
    dx, dy = math.cos(th) * length / 2.0, math.sin(th) * length / 2.0
    return (cx - dx, cy - dy), (cx + dx, cy + dy)


def list_detectors():
    """Get list of available image symmetry detectors (see detect_axes())."""
    return ["nxc", "wavelets", "xfeatures"]


def list_mask_detectors():
    """Get list of available binary-shape detectors (see detect_axes_from_mask())."""
    return ["r_lip"]


def visualize(image, predictions):
    """Draw detected symmetry lines on an image.

    Args:
        image: BGR image
        predictions: tuple of (angles, midpoints, segment_lengths, strengths)
            as returned by detect_axes() or detect_axes_from_mask()

    Returns:
        BGR image with symmetry lines drawn
    """
    from .utils.visualization import visualize_predictions
    return visualize_predictions(image, predictions)
