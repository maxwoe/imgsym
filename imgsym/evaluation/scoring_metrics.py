"""Evaluation metrics for the symmetry-scoring benchmark.

For each image with a ground-truth axis, :func:`perturb_axis` generates 'wrong'
axes (perpendicular shifts + rotations); a method scores the true vs wrong axes,
and :func:`discrimination_skill` turns those scores into a chance-anchored skill
(0 = chance, 1 = perfect), with :func:`discrimination_skill_ci` (bootstrap CI) and
:func:`paired_skill_diff` (paired significance test between two methods).
:func:`perceptual_agreement_skill` is the within-image perceptual metric (a
method's axis scores vs human vote counts).
"""

import numpy as np
from typing import Tuple, List

from .extraction import Axis


def perturb_axis(axis: Axis, image_shape: Tuple[int, int],
                 shifts: Tuple[float, ...] = (0.03, 0.05, 0.10),
                 angles: Tuple[float, ...] = (3, 5, 10)) -> List[Axis]:
    """Generate 'wrong' axes for discrimination: perpendicular shifts + rotations.

    Returns ``len(shifts)*2 + len(angles)*2`` perturbed :class:`Axis` objects
    (12 by default): perpendicular shifts of +/- each fraction of ``min(H, W)``
    along the axis normal, plus rotations of +/- each angle (degrees) about the
    axis midpoint. Shifting along the true normal (not horizontally, as the
    legacy generate_wrong_axes does) keeps it correct at any orientation. This is
    the perturbation used by the discrimination benchmark.
    """
    h, w = image_shape[:2]
    scale = min(h, w)
    nx, ny = -np.sin(axis.angle), np.cos(axis.angle)
    out: List[Axis] = []
    for f in shifts:
        for sgn in (1, -1):
            d = sgn * f * scale
            out.append(Axis(axis.cx + d * nx, axis.cy + d * ny, axis.angle,
                            axis.along_extent, axis.perp_extent))
    for a in angles:
        for sgn in (1, -1):
            out.append(Axis(axis.cx, axis.cy, axis.angle + np.radians(sgn * a),
                            axis.along_extent, axis.perp_extent))
    return out


def axis_is_near(a: "Axis", b: "Axis", image_shape: Tuple[int, int],
                 max_offset_frac: float = 0.05, max_angle_deg: float = 10.0) -> bool:
    """True if line ``a`` nearly coincides with line ``b`` (angle mod 180 within
    ``max_angle_deg`` AND normal offset within ``max_offset_frac`` of min(H, W)).
    Used to keep 'global' negatives away from the true axis."""
    h, w = image_shape[:2]
    d_ang = abs((np.degrees(a.angle - b.angle) + 90.0) % 180.0 - 90.0)
    if d_ang >= max_angle_deg:
        return False
    n = np.array([-np.sin(b.angle), np.cos(b.angle)])
    off = abs((np.array([a.cx - b.cx, a.cy - b.cy]) * n).sum()) / min(h, w)
    return off < max_offset_frac


def hard_negative_axes(image_shape: Tuple[int, int], seed: int = 0,
                       avoid: "Axis" = None) -> List[Axis]:
    """Structural ('global') wrong axes: cardinal lines, offsets, seeded-random.

    Unlike :func:`perturb_axis` (small *local* perturbations near the true axis --
    the genuinely hard negatives that discriminate methods), these are *gross*
    alternatives a method should reject easily, so they mostly test robustness
    rather than fine localization. Use them for a STRATIFIED local-vs-global
    analysis, reported separately -- not merged into the primary (local) skill,
    which they would inflate. Adapted from the OpenEvolve symmetry_score
    evaluator. Returns 12 axes (6 cardinal/offset + 6 seeded-random).

    ``avoid``: the TRUE axis, if given. Negatives that nearly coincide with it
    (see :func:`axis_is_near`) are excluded and random ones re-drawn -- on
    centered subjects the cardinal 'vertical center' line can BE the true axis,
    and scoring the truth as a negative punishes correct methods. Pass the true
    axis whenever it is known.
    """
    h, w = image_shape[:2]
    cx, cy = w / 2.0, h / 2.0
    vert = np.pi / 2.0
    cardinals = [
        Axis(cx, cy, vert),            # vertical center
        Axis(cx, cy, 0.0),             # horizontal center
        Axis(w * 0.35, cy, vert),      # vertical, left of center
        Axis(w * 0.65, cy, vert),      # vertical, right of center
        Axis(cx, h * 0.35, 0.0),       # horizontal, above center
        Axis(cx, h * 0.65, 0.0),       # horizontal, below center
    ]

    def ok(a):
        return avoid is None or not axis_is_near(a, avoid, image_shape)

    out = [a for a in cardinals if ok(a)]
    rng = np.random.RandomState(seed)
    tries = 0
    while len(out) < 12 and tries < 200:
        x1, y1 = rng.uniform(0, w - 1), rng.uniform(0, h - 1)
        x2, y2 = rng.uniform(0, w - 1), rng.uniform(0, h - 1)
        a = Axis.from_segment(x1, y1, x2, y2)
        if ok(a):
            out.append(a)
        tries += 1
    return out


def infer_direction(true_scores: list, wrong_scores: list) -> bool:
    """Decide globally whether higher scores mean more symmetric for a method.

    Returns True if the pooled mean true-axis score exceeds the pooled mean
    wrong-axis score. Direction must be decided once per method (not per image):
    picking the flattering direction per image (a max over both directions)
    inflates the chance baseline to ~0.77. Deciding globally keeps a 0.5 baseline
    and correctly handles lower-is-better measures (e.g. evolved scorers).
    """
    t = np.asarray(true_scores, dtype=float)
    w = np.asarray(wrong_scores, dtype=float)
    return float(np.nanmean(t)) >= float(np.nanmean(w))


def discrimination_one_sided(score_correct: float, wrong_scores: list,
                             higher_is_better: bool = True) -> float:
    """Fraction of wrong scores the correct score beats, in a FIXED direction.

    Ties count as 0.5 (chance for that pair), so saturating scores aren't
    penalised. Proper 0.5 chance baseline (a per-image max over both directions
    would be upward-biased to ~0.77). Pair with infer_direction to set
    higher_is_better once per method from pooled scores.
    """
    w = np.asarray(wrong_scores, dtype=float)
    w = w[np.isfinite(w)]
    if w.size == 0 or not np.isfinite(score_correct):
        return float("nan")
    diff = (score_correct - w) if higher_is_better else (w - score_correct)
    return float(np.mean((diff > 0).astype(float) + 0.5 * (diff == 0)))


def _skill_fixed(per_image: list, higher_is_better: bool) -> float:
    """Skill (2*AUC - 1) with a pre-decided direction; shared by the bootstrap
    helpers and discrimination_skill."""
    return 2.0 * float(np.nanmean([discrimination_one_sided(t, w, higher_is_better)
                                   for t, w in per_image])) - 1.0


def discrimination_skill(per_image: list, higher_is_better: bool = None) -> float:
    """Chance-anchored discrimination skill for one method across images.

    Args:
        per_image: list of ``(true_score, wrong_scores_list)`` tuples, one per
            image, for a single scoring method.
        higher_is_better: the method's score direction. None (default) infers it
            from ``per_image`` via :func:`infer_direction`. IMPORTANT: when
            evaluating a SLICE of a method's data (one perturbation level, one
            negative type, ...), pass the direction decided on the method's FULL
            standard pool -- re-inferring per slice silently flips the sign of
            near-chance slices (a per-slice max over directions).

    Returns:
        Skill in [-1, 1]: 0 = chance, 1 = the true axis beats every wrong axis on
        every image, < 0 = below chance. Direction is decided once per method
        (handles lower-is-better measures), then mapped to skill via 2*AUC - 1
        (= Somers' D / Gini).
    """
    if not per_image:
        return float("nan")
    if higher_is_better is None:
        higher_is_better = infer_direction([t for t, _ in per_image],
                                           [x for _, w in per_image for x in w])
    return _skill_fixed(per_image, higher_is_better)


def discrimination_skill_ci(per_image: list, n_boot: int = 1000, seed: int = 42
                            ) -> Tuple[float, float, float]:
    """Point estimate + 95% bootstrap CI ``(skill, lo, hi)`` for one method.

    Resamples images (the independent unit) with replacement. Direction is fixed
    once from the full sample so it cannot flip across resamples.
    """
    if not per_image:
        return float("nan"), float("nan"), float("nan")
    hib = infer_direction([t for t, _ in per_image],
                          [x for _, w in per_image for x in w])
    point = _skill_fixed(per_image, hib)
    rng = np.random.RandomState(seed)
    n = len(per_image)
    boot = [_skill_fixed([per_image[i] for i in rng.randint(0, n, n)], hib)
            for _ in range(n_boot)]
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_skill_diff(per_image_a: list, per_image_b: list,
                      n_boot: int = 1000, seed: int = 42
                      ) -> Tuple[float, float, float]:
    """Paired bootstrap of ``skill(A) - skill(B)`` over the SAME images.

    ``per_image_a`` and ``per_image_b`` must be aligned (entry i is the same
    image for both methods). Each bootstrap iteration draws one set of image
    indices and applies it to BOTH methods, preserving their correlation -- the
    correct test for "is A better than B" when both see identical images and
    perturbations. Returns ``(diff, lo, hi)``; if the CI excludes 0 the
    difference is significant at ~0.05. This is strictly more powerful than
    comparing the two methods' independent CIs for overlap.
    """
    assert len(per_image_a) == len(per_image_b), "methods must share aligned images"
    if not per_image_a:
        return float("nan"), float("nan"), float("nan")
    hib_a = infer_direction([t for t, _ in per_image_a],
                            [x for _, w in per_image_a for x in w])
    hib_b = infer_direction([t for t, _ in per_image_b],
                            [x for _, w in per_image_b for x in w])
    point = _skill_fixed(per_image_a, hib_a) - _skill_fixed(per_image_b, hib_b)
    rng = np.random.RandomState(seed)
    n = len(per_image_a)
    boot = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        boot.append(_skill_fixed([per_image_a[i] for i in idx], hib_a) -
                    _skill_fixed([per_image_b[i] for i in idx], hib_b))
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def separation_margin(score_correct: float, wrong_scores: list) -> float:
    """Per-image separation margin: a direction-agnostic Cohen's d, tanh-squashed.

    Effect size of how far the correct score sits from the wrong-score
    distribution: ``|score_correct - mean(wrong)| / std(wrong)``, squashed to
    [0, 1] via ``tanh(d / 3)`` (~0.76 at 3 sigma, ~0.96 at 6 sigma). Complements
    discrimination skill ("does correct beat wrong?", rank) with "by how much?"
    (magnitude). Borrowed from the OpenEvolve symmetry_score evaluator.
    """
    w = np.asarray(wrong_scores, dtype=float)
    w = w[np.isfinite(w)]
    if w.size == 0 or not np.isfinite(score_correct):
        return float("nan")
    sd = float(np.std(w))
    if sd < 1e-12:
        return 1.0 if abs(score_correct - float(np.mean(w))) > 1e-12 else 0.0
    return float(np.tanh(abs(score_correct - float(np.mean(w))) / sd / 3.0))


def separation_margin_mean(per_image: list) -> float:
    """Mean per-image :func:`separation_margin` over (true, wrongs) pairs."""
    vals = [separation_margin(t, w) for t, w in per_image]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def perceptual_agreement_skill(per_image: list, higher_is_better: bool = True) -> float:
    """Within-image agreement between a method's axis scores and human vote counts.

    Args:
        per_image: list over images; each entry a list of ``(score, num_labels)``
            for that image's annotated axes.
        higher_is_better: the method's symmetry direction (higher score = more
            symmetric). Take it from the method's discrimination direction, NOT
            from these data, to avoid circularity.

    Returns:
        Skill in [-1, 1]: 0 = chance, 1 = the method orders every within-image
        axis pair the way human vote counts do, < 0 = anti-correlated with
        perception. All within-image pairs with differing num_labels are pooled
        into a concordance AUC, mapped via 2*AUC - 1. Comparing axes only within
        the same image controls for per-image participant count and the salience
        (size/position) confound in num_labels.
    """
    concordant = 0.0
    total = 0
    for axes in per_image:
        n = len(axes)
        for i in range(n):
            si, li = axes[i]
            for j in range(i + 1, n):
                sj, lj = axes[j]
                if li == lj or not (np.isfinite(si) and np.isfinite(sj)):
                    continue
                total += 1
                ds = (si - sj) if higher_is_better else (sj - si)
                dl = li - lj
                if ds == 0:
                    concordant += 0.5            # score tie = chance for this pair
                elif (ds > 0) == (dl > 0):
                    concordant += 1.0
    if total == 0:
        return float("nan")
    return float(2.0 * (concordant / total) - 1.0)
