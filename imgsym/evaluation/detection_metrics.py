"""Standard symmetry-competition reflection-detection metrics.

Implements the reflection axis-matching protocol shared by the CVPR 2013 (Liu et
al., Sec 3.2) and ICCV 2017 (Funk et al., Sec 2.2.1) symmetry competitions, so our
detectors are comparable to published baselines (Loy-Eklundh, Atadjanov-Lee, ...).

A detected axis matches a ground-truth axis iff:
  * orientation difference < ``t1`` (default 10 deg), AND
  * distance from the DETECTED axis center to the GT line SEGMENT
    < ``t2 = t2_frac * min(l_det, l_GT)``  (default ``t2_frac`` 0.2; l = lengths).

Counting (per the competitions): multiple detections on one GT count as a single
true positive and the extras are NOT false positives; a detection matching no GT
is a false positive; an unmatched GT is a false negative. One detection matches at
most one GT (the nearest qualifying one). Detections are ranked by confidence and
the threshold is swept to trace a precision-recall curve; the summary score is the
maximum F-measure ``F = 2PR/(P+R)``. The perpendicular support WIDTH is not scored
(out of scope in both competitions).

Inputs are duck-typed ``Axis``-like objects exposing ``angle`` (radians),
``cx``/``cy``, ``along_extent``, and ``endpoints()`` -- see
:class:`imgsym.evaluation.extraction.Axis`.
"""
import numpy as np


def point_segment_distance(px, py, ax, ay, bx, by):
    """Euclidean distance from point ``(px, py)`` to segment ``(ax,ay)-(bx,by)``."""
    abx, aby = bx - ax, by - ay
    L2 = abx * abx + aby * aby
    if L2 <= 1e-12:
        return float(np.hypot(px - ax, py - ay))
    t = min(1.0, max(0.0, ((px - ax) * abx + (py - ay) * aby) / L2))
    return float(np.hypot(px - (ax + t * abx), py - (ay + t * aby)))


def axis_angle_diff(a, b):
    """Undirected angle difference in radians, in ``[0, pi/2]``."""
    d = abs(a - b) % np.pi
    return min(d, np.pi - d)


def axis_matches(det, gt, t1_deg=10.0, t2_frac=0.2):
    """Competition match test. Returns ``(matched, angle_deg, dist, t2)``."""
    da = axis_angle_diff(det.angle, gt.angle)
    (gx1, gy1), (gx2, gy2) = gt.endpoints()
    dist = point_segment_distance(det.cx, det.cy, gx1, gy1, gx2, gy2)
    seg_len = float(np.hypot(gx2 - gx1, gy2 - gy1))
    l_det = det.along_extent if det.along_extent else seg_len
    l_gt = gt.along_extent if gt.along_extent else seg_len
    t2 = t2_frac * min(l_det, l_gt)
    return (da < np.radians(t1_deg) and dist < t2), float(np.degrees(da)), dist, t2


def reflection_pr_f(per_image, t1_deg=10.0, t2_frac=0.2):
    """Precision-recall + max-F over a dataset.

    ``per_image``: list of ``(detections, gts)`` where ``detections`` is a list of
    ``(axis, confidence)`` and ``gts`` a list of ground-truth axes for that image.
    Sweeps the confidence threshold over every observed value (accepting
    ``conf >= threshold``) and returns a dict with ``best_f`` and the precision /
    recall / threshold at that operating point, plus the full ``curve``."""
    # Precompute each detection's nearest qualifying GT once (threshold-independent).
    pre = []
    for dets, gts in per_image:
        items = []
        for ax, conf in dets:
            best_gi = None
            best_d = np.inf
            for gi, gt in enumerate(gts):
                ok, _, d, _ = axis_matches(ax, gt, t1_deg, t2_frac)
                if ok and d < best_d:
                    best_gi, best_d = gi, d
            items.append((float(conf), best_gi))
        items.sort(key=lambda z: -z[0])          # high confidence first
        pre.append((items, len(gts)))

    confs = sorted({c for items, _ in pre for c, _ in items}, reverse=True)
    if not confs:
        return dict(best_f=0.0, precision=0.0, recall=0.0, threshold=None, curve=[])

    curve = []
    for thr in confs:
        TP = FP = FN = 0
        for items, n_gt in pre:
            matched = set()
            tp = fp = 0
            for conf, gi in items:
                if conf < thr:
                    break                         # rest are lower confidence
                if gi is None:
                    fp += 1
                elif gi not in matched:
                    matched.add(gi)
                    tp += 1
                # else: extra detection on an already-matched GT -> absorbed
            TP += tp
            FP += fp
            FN += n_gt - len(matched)
        P = TP / (TP + FP) if (TP + FP) else 1.0
        R = TP / (TP + FN) if (TP + FN) else 0.0
        F = 2 * P * R / (P + R) if (P + R) else 0.0
        curve.append((float(thr), P, R, F))

    best = max(curve, key=lambda z: z[3])
    return dict(best_f=best[3], precision=best[1], recall=best[2],
                threshold=best[0], curve=curve)
