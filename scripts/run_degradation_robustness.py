"""Degradation robustness: discrimination skill under controlled image corruption.

Full-scale version (reviewer-requested, MDPI revision R1 comment 5): ALL 13 scoring
methods on ALL FOUR single-axis datasets -- the same data, negatives, and aggregation
as the headline leaderboard (Table 4), so the clean column reproduces the single-axis
benchmark. The degradation is applied to the FULL image before crop extraction
(simulating a corrupted input photograph; the reflection-exact pipeline downstream is
unchanged), then the standard protocol runs: true axis vs the STANDARD local negatives
(shifts 3/5/10% x2, rotations 3/5/10 deg x2).

Per-condition skill uses the score direction inferred ONCE per (method, dataset) from
the CLEAN pairs (re-inferring per corrupted slice would upward-bias near-chance
methods; see scoring_metrics.discrimination_skill's docstring).

All randomness (noise field, occluder placement) is seeded per image index, so every
method and every level sees identical corruptions.

Writes per-dataset rows to --out (CSV: method,dataset,family,level,skill,lo,hi,n).
Run split across two processes for wall-clock (deep_features dominates):

    python scripts/run_degradation_robustness.py --methods deep_features --out results/degradation_full_deep.csv
    python scripts/run_degradation_robustness.py --methods <the other 12> --out results/degradation_full_rest.csv
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np

from imgsym.scoring.calculators import SymmetryCalculatorFactory as Factory
from imgsym.evaluation import (extract, Axis, load_symcomp17_single, load_nyu_single,
                               load_pix2per)
from imgsym.evaluation.scoring_metrics import (infer_direction, _skill_fixed,
                                               discrimination_one_sided)

DATA_ROOT = os.environ.get("IMGSYM_DATA", "data/datasets")
METHODS = ["pixel_correlation", "sliding_window", "dct", "eros", "gradient",
           "multi_scale_gradient", "hog", "phog", "gabor", "alexnet",
           "deep_features", "weighted_binary", "local_global"]
METHOD_KWARGS = {"weighted_binary": {"axes": "v"}}   # as in run_discrimination.py

# Standard local negatives (= the headline-skill negatives of run_discrimination.py).
STD_SHIFTS = (0.03, 0.05, 0.1)
STD_ANGLES = (3.0, 5.0, 10.0)

# (family, level-label, level-value); "clean" baseline is prepended at runtime.
DEGRADATIONS = [
    ("blur", "1", 1.0), ("blur", "2", 2.0), ("blur", "4", 4.0),
    ("noise", "5", 5.0), ("noise", "10", 10.0), ("noise", "20", 20.0),
    ("jpeg", "50", 50), ("jpeg", "25", 25), ("jpeg", "10", 10),
    ("occl", "10", 0.10), ("occl", "20", 0.20), ("occl", "30", 0.30),
]


def degrade(img, family, value, idx):
    if family == "blur":
        return cv2.GaussianBlur(img, (0, 0), sigmaX=value, sigmaY=value)
    if family == "noise":
        rng = np.random.default_rng(1000 + idx)
        n = rng.normal(0.0, value, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    if family == "jpeg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(value)])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img
    if family == "occl":
        rng = np.random.default_rng(2000 + idx)
        h, w = img.shape[:2]
        side = max(2, int(round(value * min(h, w))))
        y0 = int(rng.integers(0, max(1, h - side)))
        x0 = int(rng.integers(0, max(1, w - side)))
        out = img.copy()
        out[y0:y0 + side, x0:x0 + side] = img.reshape(-1, 3).mean(axis=0).astype(np.uint8)
        return out
    raise ValueError(family)


def std_wrong_axes(axis, shape):
    h, w = shape[:2]
    scale = min(h, w)
    nx, ny = -np.sin(axis.angle), np.cos(axis.angle)
    out = []
    for f in STD_SHIFTS:
        for sgn in (1, -1):
            d = sgn * f * scale
            out.append(Axis(axis.cx + d * nx, axis.cy + d * ny, axis.angle,
                            axis.along_extent, axis.perp_extent))
    for a in STD_ANGLES:
        for sgn in (1, -1):
            out.append(Axis(axis.cx, axis.cy, axis.angle + np.radians(sgn * a),
                            axis.along_extent, axis.perp_extent))
    return out


def skill_ci_fixed(pairs, hib, n_boot=1000, seed=42):
    """Point + 95% CI with a FIXED direction (unit-level resampling, vectorized)."""
    u = np.array([discrimination_one_sided(t, w, hib) for t, w in pairs])
    point = 2.0 * float(np.nanmean(u)) - 1.0
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(u), size=(n_boot, len(u)))
    boots = 2.0 * np.nanmean(u[idx], axis=1) - 1.0
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N images per dataset")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--out", default="results/degradation_full.csv")
    ap.add_argument("--smoke", action="store_true", help="limit 3, fast methods only")
    args = ap.parse_args()
    methods = [m for m in args.methods.split(",") if m]
    limit = args.limit
    if args.smoke:
        limit, methods = 3, ["pixel_correlation", "hog", "weighted_binary"]

    calcs = {m: Factory.create(m, **METHOD_KWARGS.get(m, {})) for m in methods}
    conditions = [("clean", "0")] + [(f, lv) for f, lv, _ in DEGRADATIONS]
    datasets = [
        ("symComp17_s", load_symcomp17_single(
            os.path.join(DATA_ROOT, "symComp17", "reflection_training", "ref_s")), "min_edge"),
        ("NYU_s", load_nyu_single(
            os.path.join(DATA_ROOT, "sym_datasets", "NYU", "S")), "min_edge"),
        ("PIX2PER-nat", load_pix2per(
            os.path.join(DATA_ROOT, "PIX2PER Dataset"), subset="nat"), "bbox"),
        ("PIX2PER-art", load_pix2per(
            os.path.join(DATA_ROOT, "PIX2PER Dataset"), subset="art"), "bbox"),
    ]

    # pairs[(method, dataset, family, level)] = [(true, [wrongs]), ...]
    pairs = {(m, ds, f, lv): [] for m in methods for ds, _, _ in datasets
             for f, lv in conditions}
    t0 = time.time()
    for ds_name, ds, policy in datasets:
        units = list(ds.dominant_axes())
        if limit:
            units = units[:limit]
        for idx, u in enumerate(units):
            img = cv2.imread(u.image_path)
            if img is None:
                continue
            waxes = std_wrong_axes(u.axis, img.shape)
            for fam, lv, val in [("clean", "0", None)] + DEGRADATIONS:
                dimg = img if fam == "clean" else degrade(img, fam, val, idx)
                rt = extract(dimg, u.axis, policy=policy, min_support_frac=0.10)
                if rt.info.degenerate:
                    continue
                wsubs = [r.subimage for r in
                         (extract(dimg, wa, policy=policy, min_support_frac=0.10)
                          for wa in waxes) if not r.info.degenerate]
                if len(wsubs) < 6:
                    continue
                for m in methods:
                    try:
                        t = float(calcs[m].calculate_score(rt.subimage))
                        if not np.isfinite(t):
                            continue
                        ws = []
                        for s in wsubs:
                            try:
                                v = float(calcs[m].calculate_score(s))
                                if np.isfinite(v):
                                    ws.append(v)
                            except Exception:
                                pass
                        if ws:
                            pairs[(m, ds_name, fam, lv)].append((t, ws))
                    except Exception:
                        continue
            if (idx + 1) % 25 == 0:
                print(f"  [{ds_name}] {idx + 1}/{len(units)} images "
                      f"({time.time() - t0:.0f}s)", flush=True)
        print(f"[{ds_name}] done ({time.time() - t0:.0f}s)", flush=True)

    # Skill per (method, dataset, condition), direction FIXED from the clean pairs.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "dataset", "family", "level", "skill", "lo", "hi", "n_images"])
        for m in methods:
            for ds_name, _, _ in datasets:
                cp = pairs[(m, ds_name, "clean", "0")]
                if not cp:
                    continue
                hib = infer_direction([t for t, _ in cp],
                                      [x for _, ws in cp for x in ws])
                for f, lv in conditions:
                    p = pairs[(m, ds_name, f, lv)]
                    if not p:
                        continue
                    sk, lo, hi = skill_ci_fixed(p, hib)
                    w.writerow([m, ds_name, f, lv, f"{sk:.4f}", f"{lo:.4f}",
                                f"{hi:.4f}", len(p)])
    print(f"wrote {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
