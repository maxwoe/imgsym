"""Self-contained unit tests for the imgsym evaluation harness.

No datasets required -- everything runs on synthetic images, so this is CI-safe.
Run with:  pytest tests/test_evaluation.py
"""
import numpy as np
import cv2
import pytest

from imgsym.evaluation import (
    Axis, extract, reflection_residual, make_fixed, perturb_axis,
    hard_negative_axes, discrimination_skill, discrimination_one_sided,
    discrimination_skill_ci, paired_skill_diff, separation_margin,
    perceptual_agreement_skill,
)
from imgsym.scoring.calculators import SymmetryCalculatorFactory as Factory
from imgsym.scoring.calculators import WeightedBinarySymmetryCalculator as WB


def _sym_about(cx, cy, ang, H=240, W=320):
    """Image exactly mirror-symmetric about the line (cx, cy, ang)."""
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    ct, st = np.cos(ang), np.sin(ang)
    perp = (xs - cx) * (-st) + (ys - cy) * ct
    along = (xs - cx) * ct + (ys - cy) * st
    g = np.clip(128 + 80 * np.cos(perp / 7) + 40 * np.cos(along / 5), 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


# --- extraction ----------------------------------------------------------- #
def test_extraction_true_axis_low_residual():
    ang = np.radians(35)
    img = _sym_about(170, 110, ang)
    sub = extract(img, Axis(170, 110, ang, along_extent=120),
                  policy=make_fixed(perp_frac=0.25, along_frac=0.25)).subimage
    assert reflection_residual(sub) < 0.05


def test_extraction_wrong_axis_higher_residual():
    ang = np.radians(35)
    img = _sym_about(170, 110, ang)
    pol = make_fixed(perp_frac=0.25, along_frac=0.25)
    r_true = reflection_residual(extract(img, Axis(170, 110, ang, along_extent=120), policy=pol).subimage)
    r_wrong = reflection_residual(extract(img, Axis(170, 110, ang + np.radians(15), along_extent=120), policy=pol).subimage)
    assert r_true < r_wrong


def test_output_size_preserves_symmetry():
    ang = np.radians(35)
    img = _sym_about(170, 110, ang)
    sub = extract(img, Axis(170, 110, ang, along_extent=120),
                  policy=make_fixed(perp_frac=0.25, along_frac=0.25), output_size=256).subimage
    assert sub.shape[:2] == (256, 256)
    assert reflection_residual(sub) < 0.05


def test_extract_even_width_centerline():
    img = _sym_about(160, 120, np.pi / 2)
    sub = extract(img, Axis(160, 120, np.pi / 2, along_extent=100),
                  policy=make_fixed(perp_frac=0.3, along_frac=0.3)).subimage
    assert sub.shape[1] % 2 == 0  # even width => axis exactly on the centerline


def test_perturb_axis_count_and_perpendicular():
    ax = Axis(100, 100, np.pi / 2, along_extent=80)  # vertical axis
    wrong = perturb_axis(ax, (200, 200))
    assert len(wrong) == 12                          # 3 shifts*2 + 3 angles*2
    for a in wrong[:6]:                              # shifts: move cx, keep cy
        assert abs(a.cy - 100) < 1e-6 and abs(a.cx - 100) > 1.0
    for a in wrong[6:]:                              # rotations: keep midpoint
        assert abs(a.cx - 100) < 1e-6 and abs(a.cy - 100) < 1e-6


# --- metrics -------------------------------------------------------------- #
def test_discrimination_skill_bounds_and_direction():
    rng = np.random.RandomState(0)
    perfect = [(10.0, list(rng.rand(12))) for _ in range(30)]
    chance = [(rng.rand(), list(rng.rand(12))) for _ in range(2000)]
    lower = [(-10.0, list(rng.rand(12))) for _ in range(30)]
    assert discrimination_skill(perfect) > 0.99
    assert abs(discrimination_skill(chance)) < 0.1
    assert discrimination_skill(lower) > 0.99   # lower-is-better auto-detected


def test_discrimination_one_sided_ties():
    assert discrimination_one_sided(1.0, [1.0, 1.0, 1.0]) == 0.5   # all ties
    assert discrimination_one_sided(2.0, [1.0, 1.0]) == 1.0
    assert discrimination_one_sided(0.0, [1.0, 1.0]) == 0.0
    assert discrimination_one_sided(0.0, [1.0], higher_is_better=False) == 1.0


def test_perceptual_agreement():
    assert perceptual_agreement_skill([[(0.9, 80), (0.1, 4)]] * 30) == 1.0
    assert perceptual_agreement_skill([[(0.1, 80), (0.9, 4)]] * 30) == -1.0
    assert perceptual_agreement_skill([[(0.5, 80), (0.5, 4)]] * 30) == 0.0


def test_discrimination_skill_ci():
    rng = np.random.RandomState(0)
    perfect = [(10.0, list(rng.rand(12))) for _ in range(40)]
    pt, lo, hi = discrimination_skill_ci(perfect, n_boot=200)
    assert pt > 0.99 and lo <= pt <= hi


def test_paired_skill_diff_detects_significant_difference():
    rng = np.random.RandomState(0)
    a = [(10.0, list(rng.rand(12))) for _ in range(50)]        # perfect
    b = [(rng.rand(), list(rng.rand(12))) for _ in range(50)]  # chance
    diff, lo, hi = paired_skill_diff(a, b, n_boot=200)
    assert diff > 0 and lo > 0    # CI excludes 0 => A significantly > B


def test_separation_margin():
    assert separation_margin(10.0, [0.0, 0.1, -0.1]) > 0.7   # far out => large margin
    assert separation_margin(0.05, [0.0, 0.1, 0.05, -0.05]) < 0.3  # inside cloud => small
    assert np.isnan(separation_margin(1.0, []))


def test_hard_negative_axes():
    axes = hard_negative_axes((200, 300))   # (H, W)
    assert len(axes) == 12
    assert all(isinstance(a, Axis) for a in axes)


# --- calculator changes --------------------------------------------------- #
def test_weighted_binary_axes_options():
    img = np.zeros((40, 40, 3), np.uint8)
    img[10:30, 5:35] = np.random.RandomState(1).randint(0, 255, (20, 30, 3))
    for axes in ("all", "v", "h"):
        assert 0 <= WB(axes=axes).calculate_score(img) <= 100
    with pytest.raises(ValueError):
        WB(axes="bad")


def test_hog_guard_small_image():
    small = np.zeros((30, 40, 3), np.uint8)   # below the 64x64 HOG window
    small[5:25, 5:35] = np.random.RandomState(1).randint(0, 255, (20, 30, 3))
    assert np.isfinite(Factory.create("hog").calculate_score(small))


def test_deep_features_default_stage():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    assert Factory.create("deep_features").feature_stage == 1


def test_local_global_heatmap_matches_score():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(7)
    half = rng.integers(0, 255, (90, 45, 3), np.uint8)
    img = np.concatenate([half, half[:, ::-1]], axis=1)    # perfectly mirror-symmetric
    calc = Factory.create("local_global")
    score, heat = calc.calculate_heatmap(img)
    assert heat.shape == img.shape[:2]
    assert np.isfinite(heat).any()
    assert abs(score - calc.calculate_score(img)) < 1e-9   # heatmap path = score path


def test_get_axis():
    import imgsym
    axes = (np.array([np.pi / 2]), np.array([[10.0, 20.0]]),
            np.array([8.0]), np.array([1.0]))
    (x1, y1), (x2, y2) = imgsym.get_axis(axes)
    assert abs(x1 - 10) < 1e-9 and abs(y1 - 16) < 1e-9
    assert abs(x2 - 10) < 1e-9 and abs(y2 - 24) < 1e-9


def test_detect_axes_from_mask_dispatch():
    import imgsym
    square = np.zeros((60, 60), np.uint8)
    square[15:45, 15:45] = 1
    axes = imgsym.detect_axes_from_mask(square, detector="r_lip")
    assert len(axes) == 4 and len(axes[0]) >= 1
    with pytest.raises(ValueError, match="r_lip"):
        imgsym.detect_axes_from_mask(square, detector="nxc")
