"""Verify the competition reflection metric against the CVPR'13 / ICCV'17 spec.

Each test pins one rule to a hand-computable case: angle/distance thresholds, the
t2 = 0.2*min(l_det, l_GT) length scaling, the distance-to-SEGMENT (not line), the
many-to-one absorb rule, the no-match false positive, and the confidence-swept PR /
max-F. Includes the exact TP=1/FP=1/FN=1 example from CVPR'13 Fig 5(b).

Runnable directly (python tests/test_detection_metrics.py) or via pytest.
"""
import numpy as np

from imgsym.evaluation.detection_metrics import (axis_angle_diff, axis_matches,
                                                 point_segment_distance,
                                                 reflection_pr_f)
from imgsym.evaluation.extraction import Axis


def _axis(cx, cy, angle_rad, length):
    return Axis(cx=cx, cy=cy, angle=angle_rad, along_extent=length)


VERT = np.pi / 2.0   # a vertical axis: atan2(dy, dx) with dx=0, dy>0


# --------------------------------------------------------------------------- #
# geometry primitives
# --------------------------------------------------------------------------- #
def test_point_segment_distance_perpendicular():
    # (65,50) onto vertical segment x=50, y in [10,90] -> foot (50,50), d=15
    assert abs(point_segment_distance(65, 50, 50, 10, 50, 90) - 15.0) < 1e-9


def test_point_segment_distance_beyond_endpoint():
    # (50,100) is past the top endpoint (50,90) -> distance to endpoint = 10
    assert abs(point_segment_distance(50, 100, 50, 10, 50, 90) - 10.0) < 1e-9


def test_point_segment_distance_degenerate():
    assert abs(point_segment_distance(3, 4, 0, 0, 0, 0) - 5.0) < 1e-9


def test_angle_diff_is_undirected():
    assert abs(axis_angle_diff(0.0, np.pi)) < 1e-9            # pi == same axis
    assert abs(axis_angle_diff(0.1, np.pi - 0.1) - 0.2) < 1e-9  # wraps, 0.2 apart
    assert abs(axis_angle_diff(0.0, VERT) - VERT) < 1e-9     # orthogonal -> pi/2


# --------------------------------------------------------------------------- #
# match rule: angle, distance, t2 = 0.2*min(l_det, l_GT)
# --------------------------------------------------------------------------- #
def test_perfect_match():
    gt = Axis.from_segment(50, 10, 50, 90)
    ok, da, dist, t2 = axis_matches(_axis(50, 50, VERT, 80), gt)
    assert ok and da < 1e-9 and dist < 1e-9 and abs(t2 - 16.0) < 1e-9


def test_angle_threshold_is_10deg():
    gt = Axis.from_segment(50, 10, 50, 90)             # center on the segment -> dist 0
    assert axis_matches(_axis(50, 50, VERT + np.radians(9), 80), gt)[0]
    assert not axis_matches(_axis(50, 50, VERT + np.radians(11), 80), gt)[0]


def test_distance_threshold_strict():
    gt = Axis.from_segment(50, 10, 50, 90)             # len 80 -> t2 = 16
    assert axis_matches(_axis(65, 50, VERT, 80), gt)[0]        # dist 15 < 16
    assert not axis_matches(_axis(66, 50, VERT, 80), gt)[0]    # dist 16, not < 16


def test_t2_uses_min_of_lengths():
    # short detection (len 40) tightens t2 to 0.2*40 = 8, not 0.2*80
    gt = Axis.from_segment(50, 10, 50, 90)
    _, _, _, t2 = axis_matches(_axis(50, 50, VERT, 40), gt)
    assert abs(t2 - 8.0) < 1e-9
    assert axis_matches(_axis(57, 50, VERT, 40), gt)[0]        # dist 7 < 8
    assert not axis_matches(_axis(59, 50, VERT, 40), gt)[0]    # dist 9 > 8


def test_distance_is_to_segment_not_infinite_line():
    # detection collinear with GT's line but its center is far PAST the segment
    # end along the axis -> distance to segment is large -> no match.
    gt = Axis.from_segment(50, 10, 50, 30)             # short segment near top
    # center (50, 200): collinear (same x), but 170 px past the segment end (50,30)
    assert not axis_matches(_axis(50, 200, VERT, 80), gt)[0]


# --------------------------------------------------------------------------- #
# counting + PR/F over a dataset
# --------------------------------------------------------------------------- #
def test_perfect_dataset_f_is_one():
    gt = Axis.from_segment(50, 10, 50, 90)
    per_image = [([(_axis(50, 50, VERT, 80), 1.0)], [gt])]
    assert abs(reflection_pr_f(per_image)["best_f"] - 1.0) < 1e-9


def test_no_match_is_false_positive():
    gt = Axis.from_segment(50, 10, 50, 90)
    horiz = _axis(50, 50, 0.0, 80)                     # 90 deg off -> FP, GT -> FN
    res = reflection_pr_f([([(horiz, 1.0)], [gt])])
    assert res["best_f"] == 0.0


def test_many_detections_one_gt_is_one_tp_no_fp():
    # two valid detections on the single GT -> 1 TP, 0 FP (extras absorbed).
    gt = Axis.from_segment(50, 10, 50, 90)
    dets = [(_axis(50, 50, VERT, 80), 0.9), (_axis(52, 50, VERT, 80), 0.8)]
    res = reflection_pr_f([(dets, [gt])])
    # if the extra were wrongly counted FP, precision would drop to 0.5
    assert abs(res["best_f"] - 1.0) < 1e-9 and abs(res["precision"] - 1.0) < 1e-9


def test_cvpr_fig5b_tp1_fp1_fn1():
    # Two GTs; R1 hits GT1, R2 matches nothing, GT2 unmatched -> at the
    # all-accepted operating point: TP=1, FP=1, FN=1 -> P=0.5, R=0.5.
    gt1 = Axis.from_segment(30, 10, 30, 90)
    gt2 = Axis.from_segment(70, 10, 70, 90)
    r1 = (_axis(30, 50, VERT, 80), 0.9)               # matches gt1 only
    r2 = (_axis(50, 50, 0.0, 80), 0.8)                # horizontal -> matches none
    res = reflection_pr_f([([r1, r2], [gt1, gt2])])
    # the lowest-threshold (all-accepted) curve point must read P=0.5, R=0.5
    thr, P, R, F = min(res["curve"], key=lambda z: z[0])
    assert abs(P - 0.5) < 1e-9 and abs(R - 0.5) < 1e-9


def test_confidence_sweep_orders_detections():
    # Image A: correct det @0.90.  Image B: a wrong det @0.95.
    # Best threshold accepts both -> TP=1, FP=1, FN=1 -> F=0.5.
    gtA = Axis.from_segment(50, 10, 50, 90)
    gtB = Axis.from_segment(50, 10, 50, 90)
    a = ([(_axis(50, 50, VERT, 80), 0.90)], [gtA])
    b = ([(_axis(50, 50, 0.0, 80), 0.95)], [gtB])     # horizontal -> wrong
    res = reflection_pr_f([a, b])
    assert abs(res["best_f"] - 0.5) < 1e-9


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"FAIL {fn.__name__}: {exc!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {exc!r}")
    print(f"\n{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run() else 1)
