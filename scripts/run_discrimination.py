"""Discrimination benchmark — score once, store raw, analyse many ways.

For each dataset, every method scores the true axis vs a rich *tagged* set of
wrong axes:
  * a fine->coarse LOCAL perturbation sweep (shifts 0.5%-10%, rotations 0.5-10 deg)
  * GLOBAL structural negatives (cardinal / offset / seeded-random)
All per-image scores are stored to ``results/per_image_scores.json`` so every
downstream view -- skill+CI, the sensitivity curve, local-vs-global, separation
margin, paired significance -- is computed post-hoc by ``analyze_discrimination.py``
without re-scoring.

Also writes ``results/discrimination.{csv,md}`` (headline skill on the STANDARD
local levels, for comparability) with bootstrap CIs and the separation margin.

    python scripts/run_discrimination.py [--limit N] [--no-global]

One process, models loaded once. Fixed seeds -> reproducible. Datasets whose
files are absent are skipped (reported).
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

from imgsym.scoring.calculators import SymmetryCalculatorFactory as Factory
from imgsym.evaluation import (extract, hard_negative_axes, discrimination_skill_ci,
                               separation_margin_mean, Axis, load_pix2per,
                               load_symcomp17_single, load_symcomp13_single,
                               load_nyu_single, load_symcomp13_multiple,
                               load_nyu_multiple, load_dendi_reflection)

DATA_ROOT = os.environ.get("IMGSYM_DATA", "data/datasets")
METHOD_KWARGS = {"weighted_binary": {"axes": "v"}}

# Full sweep (stored). Tags use {:g} so 0.1 -> "0.1", 10.0 -> "10".
SHIFTS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.1)     # fraction of min(H, W)
ANGLES = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0)          # degrees
# Headline skill/margin use these "standard" levels (matches the earlier run).
STD_SHIFTS = (0.03, 0.05, 0.1)
STD_ANGLES = (3.0, 5.0, 10.0)
STD_TAGS = ({f"shift_{f:g}" for f in STD_SHIFTS} | {f"rot_{a:g}" for a in STD_ANGLES})


def _configs(root, protocol, max_axes):
    """(name, loader, extent-policy) per dataset for the chosen protocol.

    single = one dominant axis per image (the headline benchmark);
    multi  = every annotated reflection axis, density-capped at ``max_axes``
    (symComp13/NYU multiple sets + DENDI + PIX2PER-all)."""
    if protocol == "single":
        # symComp13_s is EXCLUDED (2026-07-04): 25/35 of its images are the same
        # images as symComp17_s (content-hash verified, GT within ~1 degree), so
        # the two columns are not independent; symComp17_s is the competition
        # representative. symComp13_m below is the disjoint multi-axis subset.
        return [
            ("symComp17_s", lambda: load_symcomp17_single(
                os.path.join(root, "symComp17", "reflection_training", "ref_s")), "min_edge"),
            ("NYU_s", lambda: load_nyu_single(
                os.path.join(root, "sym_datasets", "NYU", "S")), "min_edge"),
            ("PIX2PER-art", lambda: load_pix2per(
                os.path.join(root, "PIX2PER Dataset"), subset="art"), "bbox"),
            ("PIX2PER-nat", lambda: load_pix2per(
                os.path.join(root, "PIX2PER Dataset"), subset="nat"), "bbox"),
        ]
    return [
        ("symComp13_m", lambda: load_symcomp13_multiple(
            os.path.join(root, "symComp13"), max_axes=max_axes), "min_edge"),
        ("NYU_m", lambda: load_nyu_multiple(
            os.path.join(root, "sym_datasets", "NYU", "M"), max_axes=max_axes), "min_edge"),
        ("DENDI", lambda: load_dendi_reflection(
            os.path.join(root, "dendi"), max_axes=max_axes), "min_edge"),
        ("PIX2PER-art", lambda: load_pix2per(
            os.path.join(root, "PIX2PER Dataset"), subset="art", max_axes=max_axes), "bbox"),
        ("PIX2PER-nat", lambda: load_pix2per(
            os.path.join(root, "PIX2PER Dataset"), subset="nat", max_axes=max_axes), "bbox"),
    ]


def tagged_wrong_axes(axis, shape, image_index, use_global):
    """[(tag, Axis), ...]: local sweep (shift_*/rot_*) then global.

    Global negatives pass ``avoid=axis`` so none of them near-coincides with the
    true axis (on centered subjects the cardinal center line can BE the truth,
    which would punish correct methods in the local-vs-global analysis)."""
    h, w = shape[:2]
    scale = min(h, w)
    nx, ny = -np.sin(axis.angle), np.cos(axis.angle)
    specs = []
    for f in SHIFTS:
        for sgn in (1, -1):
            d = sgn * f * scale
            specs.append((f"shift_{f:g}",
                          Axis(axis.cx + d * nx, axis.cy + d * ny, axis.angle,
                               axis.along_extent, axis.perp_extent)))
    for a in ANGLES:
        for sgn in (1, -1):
            specs.append((f"rot_{a:g}",
                          Axis(axis.cx, axis.cy, axis.angle + np.radians(sgn * a),
                               axis.along_extent, axis.perp_extent)))
    if use_global:
        for g in hard_negative_axes(shape, seed=image_index, avoid=axis):
            specs.append(("global", g))
    return specs


def _cap_dim(sub, cap=256):
    """Downscale a subimage so its longest side <= cap (aspect preserved). Bounds per-unit
    compute for the per-pixel classical methods on large crops -- DENDI's COCO images give
    crops ~7x larger than the single-axis sets, which the deep method (resizes to 224
    internally) shrugs off but sliding_window/phog/gabor do not. Applied to the MULTI protocol
    only (gated in run_dataset); resizing preserves the mirror-symmetry signal (cf. output_size)."""
    h, w = sub.shape[:2]
    if max(h, w) <= cap:
        return sub
    s = cap / float(max(h, w))
    return cv2.resize(sub, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA)


def run_dataset(name, dataset, policy, calcs, use_global, protocol, limit=None):
    """Return {method: [ {"true": float, "wrong": [[tag, score|None], ...]}, ... ]}.

    protocol="single" scores one dominant axis per image; "multi" scores EVERY
    annotated axis (one unit per (image, axis)) via ``all_axes``."""
    raw = {m: [] for m in calcs}
    # Crop cap is a tractability measure for the multi protocol's large in-the-wild crops
    # (DENDI); the single-axis benchmark stays at native resolution (its headline numbers).
    cap = _cap_dim if protocol == "multi" else (lambda s: s)
    used = skipped = 0
    method_skips = {m: 0 for m in calcs}   # units dropped per method (true-axis failures)
    t0 = time.time()
    units = list(dataset.all_axes() if protocol == "multi" else dataset.dominant_axes())
    if limit:
        units = units[:limit]
    for idx, u in enumerate(units):
        img = cv2.imread(u.image_path)
        if img is None:
            skipped += 1
            continue
        rt = extract(img, u.axis, policy=policy, min_support_frac=0.10)
        if rt.info.degenerate:
            skipped += 1
            continue
        true_sub = cap(rt.subimage)
        # Extract every wrong axis once (None = degenerate, keeps tag alignment).
        wsubs = []
        for tag, wax in tagged_wrong_axes(u.axis, img.shape, idx + 1, use_global):
            r = extract(img, wax, policy=policy, min_support_frac=0.10)
            wsubs.append((tag, None if r.info.degenerate else cap(r.subimage)))
        used += 1
        if used % 50 == 0:
            print(f"  [{name}] {used} units ({time.time() - t0:.0f}s)", flush=True)
        uid = f"{os.path.basename(u.image_path)}#{idx}"   # unit identity for paired joins
        for m, c in calcs.items():
            try:
                t = float(c.calculate_score(true_sub))
            except Exception:
                method_skips[m] += 1
                continue                          # method crashed on the true axis -> skip unit
            if not np.isfinite(t):
                method_skips[m] += 1
                continue
            wrong = []
            for tag, sub in wsubs:
                if sub is None:
                    wrong.append([tag, None])
                    continue
                try:
                    s = float(c.calculate_score(sub))
                    wrong.append([tag, s if np.isfinite(s) else None])
                except Exception:
                    wrong.append([tag, None])
            raw[m].append({"img": uid, "true": t, "wrong": wrong})
    per_m = {m: n for m, n in method_skips.items() if n}
    print(f"[{name}] used={used} skipped={skipped}"
          + (f" method_skips={per_m}" if per_m else "")
          + f" ({time.time()-t0:.0f}s)", flush=True)
    if per_m:
        print(f"[{name}] WARNING: method-specific unit drops above -- paired analyses "
              "must join on 'img', not position.", flush=True)
    return raw


def std_local_pairs(per_image):
    """(true, [wrong scores]) using the STANDARD local levels (drop None/global)."""
    out = []
    for rec in per_image:
        wrongs = [s for tag, s in rec["wrong"] if tag in STD_TAGS and s is not None]
        if wrongs:
            out.append((rec["true"], wrongs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N units per dataset")
    ap.add_argument("--no-global", action="store_true", help="skip global structural negatives")
    ap.add_argument("--protocol", choices=["single", "multi"], default="single",
                    help="single = one dominant axis/image; multi = every annotated axis")
    ap.add_argument("--max-axes", type=int, default=10,
                    help="multi: drop images with more than this many axes (density cap)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    use_global = not args.no_global
    tag = "" if args.protocol == "single" else "_multi"

    methods = Factory.list_methods()
    print("loading calculators...", flush=True)
    calcs = {m: Factory.create(m, **METHOD_KWARGS.get(m, {})) for m in methods}

    os.makedirs(args.out, exist_ok=True)
    meta = {"shifts": list(SHIFTS), "angles": list(ANGLES), "use_global": use_global,
            "std_tags": sorted(STD_TAGS), "protocol": args.protocol, "max_axes": args.max_axes}
    pij = os.path.join(args.out, f"per_image_scores{tag}.json")

    all_raw = {}
    for name, loader, policy in _configs(DATA_ROOT, args.protocol, args.max_axes):
        try:
            ds = loader()
        except (FileNotFoundError, OSError) as exc:
            print(f"[{name}] SKIP (data not found: {exc})", flush=True)
            continue
        if len(ds) == 0:
            print(f"[{name}] SKIP (no images)", flush=True)
            continue
        print(f"[{name}] {len(ds)} imgs, {sum(len(im.axes) for im in ds.images)} axes; "
              f"stats={ds.stats}", flush=True)
        all_raw[name] = run_dataset(name, ds, policy, calcs, use_global, args.protocol, args.limit)
        with open(pij, "w") as fh:                 # incremental save (crash-resilient)
            json.dump({"meta": meta, "data": all_raw}, fh)

    if not all_raw:
        print(f"No datasets found under {DATA_ROOT!r}.", file=sys.stderr)
        sys.exit(1)

    names = list(all_raw)
    skill = {n: {m: discrimination_skill_ci(std_local_pairs(all_raw[n][m])) for m in methods}
             for n in names}
    margin = {n: {m: separation_margin_mean(std_local_pairs(all_raw[n][m])) for m in methods}
              for n in names}

    with open(os.path.join(args.out, f"discrimination{tag}.csv"), "w", newline="") as fh:
        import csv
        wr = csv.writer(fh)
        wr.writerow(["method"] + [f"{n}_{s}" for n in names for s in ("skill", "lo", "hi", "margin")])
        for m in methods:
            row = [m]
            for n in names:
                pt, lo, hi = skill[n][m]
                row += [f"{pt:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{margin[n][m]:.4f}"]
            wr.writerow(row)

    def mean_skill(m):
        return float(np.mean([skill[n][m][0] for n in names]))

    order = sorted(methods, key=mean_skill, reverse=True)
    with open(os.path.join(args.out, f"discrimination{tag}.md"), "w") as fh:
        fh.write(f"# Discrimination skill ({args.protocol} protocol, standard local levels), "
                 "95% CI; (m)=separation margin\n\n")
        fh.write("| method | " + " | ".join(names) + " | mean skill |\n")
        fh.write("|" + "---|" * (len(names) + 2) + "\n")
        for m in order:
            cells = " | ".join(
                f"{skill[n][m][0]:+.2f} [{skill[n][m][1]:+.2f},{skill[n][m][2]:+.2f}] (m{margin[n][m]:.2f})"
                for n in names)
            fh.write(f"| {m} | {cells} | {mean_skill(m):+.2f} |\n")

    print(f"wrote {args.out}/per_image_scores{tag}.json, discrimination{tag}.csv, "
          f"discrimination{tag}.md", flush=True)


if __name__ == "__main__":
    main()
