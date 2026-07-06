"""Qualitative figure for the scoring paper: the task, and what success looks like.

Top row: one NYU image with the annotated true axis plus examples of the
negative axes it must outrank (shifted / rotated / global), and the canonical
crop pair a scorer actually receives (I and M I).
Bottom row: four exemplar crops with the three leaders' verdicts (check = the
method ranks the true axis above ALL 12 standard perturbed negatives), chosen
live by re-scoring the first --limit NYU_s units: one everyone gets right, one
only DeepFeat gets, one only HOG gets, one everyone misses. Self-contained on
purpose -- no dependency on the master per_image_scores.json (whose legacy
records carry no image ids).

    conda run -n imgsym python scripts/make_task_figure.py [--limit 40]

Writes ../image-symmetry-scoring-paper/figures/fig_task_examples.pdf
"""
import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_discrimination import _configs, tagged_wrong_axes, DATA_ROOT, STD_TAGS

from imgsym.evaluation import extract
from imgsym.scoring.calculators import SymmetryCalculatorFactory as Factory

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "image-symmetry-scoring-paper", "figures")
METHODS = ["deep_features", "alexnet", "hog"]      # three leaders, leaderboard order
DISPLAY = {"deep_features": "DeepFeat", "hog": "HOG", "alexnet": "AlexNet-C2"}
OKABE = {"true": "#009E73", "shift": "#E69F00", "rot": "#0072B2", "global": "#999999"}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})


def axis_segment(ax_obj, length):
    """Endpoints of the axis line through (cx, cy) with direction (cos a, sin a)."""
    dx, dy = np.cos(ax_obj.angle), np.sin(ax_obj.angle)
    return ((ax_obj.cx - length * dx, ax_obj.cx + length * dx),
            (ax_obj.cy - length * dy, ax_obj.cy + length * dy))


def scene_view(ax_obj, sub):
    """Display the scene-consistent member of the pair {I, M I}.

    The extraction warp maps the axis normal to the crop's +x direction, so an
    upward-pointing axis (sin > 0) yields the left-right MIRRORED member (the
    clock-numerals test). Scores are invariant (every comparison is symmetric
    in the pair), but for display we show the member matching the scene."""
    return np.fliplr(sub) if np.sin(ax_obj.angle) > 0 else sub


def letterbox_square(im, pad=236):
    """Pad (never crop) the crop into a uniform square tile for display.

    The scorer receives the FULL canonical crop, so we must not trim it; we
    only add a neutral border to make every displayed tile the same square
    size (equal subplot boxes => equal verdict-text spacing) while showing the
    true crop, complete and at its true aspect ratio. Returns the padded image
    and the content rectangle (top, left, h, w) so the axis line can be drawn
    over the image region only."""
    h, w = im.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    out = np.full((s, s, im.shape[2]), pad, dtype=im.dtype)
    out[top:top + h, left:left + w] = im
    return out, (top, left, h, w)


def score_unit(img, unit, policy, calcs):
    """Per-method (fraction, (wins, total)) of perturbed negatives outranked.

    Raw scores are deliberately NOT reported: the 13 methods' score scales are
    arbitrary and mutually incomparable (paper Section on the metric), so the
    honest per-method quantity is the rank outcome wins/total."""
    rt = extract(img, unit.axis, policy=policy, min_support_frac=0.10)
    if rt.info.degenerate:
        return None, None, None
    true_sub = rt.subimage
    wsubs = [(tag, extract(img, wax, policy=policy, min_support_frac=0.10))
             for tag, wax in tagged_wrong_axes(unit.axis, img.shape, 1, use_global=False)
             if tag in STD_TAGS]
    fracs, counts = {}, {}
    for m in METHODS:
        t = float(calcs[m].calculate_score(true_sub))
        wins = tot = 0
        for tag, r in wsubs:
            if r.info.degenerate:
                continue
            s = float(calcs[m].calculate_score(r.subimage))
            if np.isfinite(s):
                tot += 1
                wins += int(t > s)
        fracs[m] = wins / tot if tot else np.nan
        counts[m] = (wins, tot)
    return true_sub, fracs, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40,
                    help="units scanned on NYU_s (task illustration + easy win)")
    ap.add_argument("--limit-hard", type=int, default=80,
                    help="units scanned on PIX2PER-nat (divergent exemplars)")
    args = ap.parse_args()

    print("loading calculators...", flush=True)
    calcs = {m: Factory.create(m) for m in METHODS}

    scored = []                                   # (unit, img, true_sub, fracs, ds)
    for ds_name, lim in [("NYU_s", args.limit), ("PIX2PER-nat", args.limit_hard)]:
        _, loader, policy = next(c for c in _configs(DATA_ROOT, "single", 10)
                                 if c[0] == ds_name)
        units = list(loader().dominant_axes())[:lim]
        for i, u in enumerate(units):
            img = cv2.imread(u.image_path)
            if img is None:
                continue
            true_sub, fracs, counts = score_unit(img, u, policy, calcs)
            if fracs is None:
                continue
            # i + 1 = the unit's global-negative seed in the harness (run_dataset
            # uses seed = idx + 1), so displayed globals match the run exactly.
            scored.append((u, img, true_sub, fracs, ds_name, counts, i + 1))
            if (i + 1) % 20 == 0:
                print(f"  [{ds_name}] scored {i + 1}/{len(units)}", flush=True)

    def ok(f):        # "gets it": true axis above ALL standard negatives
        return f >= 0.999

    def near_vertical(rec):
        # Crops are rotated so the axis is vertical; prefer axes that are
        # near-vertical AND point upward (mod 360), since a downward-pointing
        # vertical axis yields a crop rotated 180 degrees relative to the
        # scene (content looks upside down) and off-vertical axes add
        # constant-border fill.
        d = abs((np.rad2deg(rec[0].axis.angle) % 360.0) - 90.0)
        return d <= 25.0

    def pick(pred):
        for pool in ([r for r in scored if near_vertical(r)], scored):
            for rec in pool:
                if pred(rec[3]):
                    return rec
        return None

    exemplars = [
        ("all three succeed", pick(lambda f: all(ok(f[m]) for m in METHODS))),
        ("only DeepFeat", pick(lambda f: ok(f["deep_features"]) and f["hog"] <= 0.75)),
        ("only HOG", pick(lambda f: ok(f["hog"]) and f["deep_features"] <= 0.75)),
        ("all three miss", pick(lambda f: all(f[m] <= 0.6 for m in METHODS))),
    ]
    for label, rec in exemplars:
        print(f"  [{label}]", "-" if rec is None else
              dict({DISPLAY[m]: round(rec[3][m], 2) for m in METHODS}, ds=rec[4]),
              flush=True)
        if rec is not None:
            print(f"      image: {rec[0].image_path}", flush=True)

    illu = exemplars[0][1] or scored[0]           # task-illustration image
    u, img, true_sub = illu[0], illu[1], illu[2]
    negs = tagged_wrong_axes(u.axis, img.shape, illu[6], use_global=True)
    shift_ax = next(a for t, a in negs if t == "shift_0.05")
    rot_ax = next(a for t, a in negs if t == "rot_5")
    globals_ = [a for t, a in negs if t == "global"]
    os.makedirs(OUT, exist_ok=True)

    # Raw true-axis scores of the illustration crop (for the caption; each on
    # its own, method-specific scale -- the benchmark compares ranks, not values).
    raw = {DISPLAY[m]: float(calcs[m].calculate_score(true_sub)) for m in METHODS}
    print("  illustration s(I):", {k: round(v, 3) for k, v in raw.items()}, flush=True)

    # Most-diagonal global negative: cardinal center lines can accidentally
    # align with genuine secondary symmetries (e.g., a water reflection about a
    # horizontal line), which would make a confusing illustration.
    def diagness(a):
        d = np.rad2deg(a.angle) % 90.0
        return min(d, 90.0 - d)

    glob_ax = max(globals_, key=diagness)

    # ---------- One combined figure: image + axes -> crop pair, + exemplars ----------
    fig = plt.figure(figsize=(7.0, 4.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.16, wspace=0.10)

    # top left (2 cells): the input image with the true axis and one negative
    # example of each kind (shifted / rotated / global); referenced from the
    # negatives section of the paper.
    axImg = fig.add_subplot(gs[0, 0:2])
    axImg.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    L = max(img.shape[:2])
    for a_obj, key, style, lbl, lw in [
            (u.axis, "true", "-", "true axis", 1.8),
            (shift_ax, "shift", "--", "shifted 5%", 1.5),
            (rot_ax, "rot", ":", "rotated 5$^\\circ$", 1.5),
            (glob_ax, "global", "-.", "global negative", 1.5)]:
        (x0, x1), (y0, y1) = axis_segment(a_obj, L)
        axImg.plot([x0, x1], [y0, y1], style, color=OKABE[key], lw=lw, label=lbl)
    axImg.set_xlim(0, img.shape[1]); axImg.set_ylim(img.shape[0], 0)
    axImg.set_title("true axis and negative axes")
    axImg.legend(loc="lower right", fontsize=6, framealpha=0.85)
    axImg.axis("off")

    def show_tile(ax, crop_bgr, neg_scale=None):
        """Draw the FULL crop letterboxed into a uniform square tile, with the
        candidate axis marked over the image region only.

        neg_scale: min(H, W) of the ORIGINAL image. When given, also draw the
        12 standard perturbed negatives in the true-crop frame -- vertical
        lines at +-{3,5,10}% of neg_scale (same pixel scale as the crop) and
        lines through the axis center at +-{3,5,10} degrees -- clipped to the
        image region, so the wins/12 verdicts refer to visible lines."""
        from matplotlib.patches import Rectangle
        padded, (top, left, h, w) = letterbox_square(crop_bgr)
        ax.imshow(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
        # Freeze the imshow limits: a full-height axis line touching the image
        # edge would otherwise trigger a 5% autoscale margin, enlarging this
        # tile's box (aspect='equal') and misaligning it with the others.
        ax.autoscale(False)
        cx, cy = padded.shape[1] / 2.0, top + h / 2.0
        if neg_scale is not None:
            clip = Rectangle((left, top), w, h, transform=ax.transData)
            # Styled to match the top panel: shifted = dashed orange, rotated =
            # dotted blue. All 12 standard negatives are drawn so the wins/12
            # verdicts refer to visible lines.
            for f in (0.03, 0.05, 0.10):            # shifted negatives (dashed)
                for sgn in (1, -1):
                    ln, = ax.plot([cx + sgn * f * neg_scale] * 2, [top, top + h],
                                  "--", color=OKABE["shift"], lw=0.8, alpha=0.75)
                    ln.set_clip_path(clip)
            for a_deg in (3, 5, 10):                # rotated negatives (dotted)
                t = np.tan(np.radians(a_deg))
                for sgn in (1, -1):
                    ln, = ax.plot([cx - sgn * t * h / 2.0, cx + sgn * t * h / 2.0],
                                  [top + h, top], ":",
                                  color=OKABE["rot"], lw=0.9, alpha=0.75)
                    ln.set_clip_path(clip)
        ax.plot([cx, cx], [top, top + h], color=OKABE["true"], lw=1.6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#cccccc"); sp.set_linewidth(0.6)

    # --- top right (2 cells): canonical crop pair (scene-consistent member as
    #     I). Full crops, letterboxed to uniform square tiles (never cropped). ---
    show_sub = scene_view(u.axis, true_sub)
    for k, (m_img, ttl) in enumerate([(show_sub, "canonical crop $I$"),
                                      (np.fliplr(show_sub), "its mirror $M\\,I$")]):
        axB = fig.add_subplot(gs[0, 2 + k])
        show_tile(axB, m_img)
        axB.set_title(ttl)

    # --- bottom row: exemplar tiles with verdicts + outrank counts. Raw scores
    #     are deliberately not shown (arbitrary, mutually incomparable scales);
    #     wins/total over the 12 perturbed negatives is the honest quantity. ---
    for k, (label, rec) in enumerate(exemplars):
        axC = fig.add_subplot(gs[1, k])
        if rec is None:
            axC.axis("off")
            axC.set_title(f"({label}: none in sample)", fontsize=7)
            continue
        show_tile(axC, scene_view(rec[0].axis, rec[2]),
                  neg_scale=min(rec[1].shape[:2]))
        fracs, counts = rec[3], rec[5]
        for j, m in enumerate(METHODS):
            good = ok(fracs[m])
            wins, tot = counts[m]
            axC.text(0.5, -0.06 - 0.12 * j,
                     f"{DISPLAY[m]} {'✓' if good else '✗'} {wins}/{tot}",
                     color="#009E73" if good else "#D55E00", fontsize=6.8,
                     ha="center", va="top", transform=axC.transAxes)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fig_task_examples.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    print("wrote", os.path.normpath(path), flush=True)


if __name__ == "__main__":
    main()
