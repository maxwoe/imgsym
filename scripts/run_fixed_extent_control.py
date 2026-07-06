"""Crop-size-confound control: re-run the single-axis benchmark with FIXED extents.

Threat to validity being tested: under the ``min_edge`` policy the perpendicular
half-width is the distance to the nearer image edge, so a SHIFTED wrong axis gets a
systematically narrower crop than the true axis (up to ~27% off-center). A scorer
sensitive to crop geometry could therefore gain shift-skill without measuring
symmetry. Control: a ``make_fixed`` policy gives the true axis and every wrong axis
of an image IDENTICAL crop dimensions (fractions of the short side), eliminating the
size cue; remaining skill must come from content. If the method ranking and the
deep-vs-hog story hold under fixed extents, the benchmark's conclusions do not rest
on the confound.

    python scripts/run_fixed_extent_control.py [--limit N]

Writes results/fixed_extent_control.{csv,md} (skill under fixed extents vs the
benchmark's min_edge/bbox numbers from results/discrimination.csv).
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_discrimination import (_configs, run_dataset, std_local_pairs,
                                METHOD_KWARGS, DATA_ROOT)

from imgsym.scoring.calculators import SymmetryCalculatorFactory as Factory
from imgsym.evaluation import discrimination_skill_ci
from imgsym.evaluation.extraction import make_fixed

# Thesis-critical methods + the most size-suspect cheap ones.
METHODS = ("deep_features", "hog", "alexnet", "gabor", "sliding_window",
           "pixel_correlation")
# Half-extents as fractions of min(H, W): every crop of an image is the same
# 0.5 x 0.7 short-side window regardless of axis position.
POLICY = make_fixed(perp_frac=0.25, along_frac=0.35)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print("loading calculators...", flush=True)
    calcs = {m: Factory.create(m, **METHOD_KWARGS.get(m, {})) for m in METHODS}

    skills = {}
    for name, loader, _default_policy in _configs(DATA_ROOT, "single", max_axes=10):
        try:
            ds = loader()
        except (FileNotFoundError, OSError) as exc:
            print(f"[{name}] SKIP ({exc})", flush=True)
            continue
        print(f"[{name}] {len(ds)} imgs (fixed extents)", flush=True)
        raw = run_dataset(name, ds, POLICY, calcs, use_global=False,
                          protocol="single", limit=args.limit)
        skills[name] = {m: discrimination_skill_ci(std_local_pairs(raw[m]))
                        for m in METHODS}

    # benchmark (min_edge/bbox) numbers for comparison
    bench = {}
    rows = list(csv.reader(open("results/discrimination.csv")))
    dss = [c[:-6] for c in rows[0][1:] if c.endswith("_skill")]
    for r in rows[1:]:
        bench[r[0]] = {d: float(r[1 + 4 * i]) for i, d in enumerate(dss)}

    names = list(skills)
    os.makedirs("results", exist_ok=True)
    with open("results/fixed_extent_control.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method"] + [f"{n}_{s}" for n in names
                                 for s in ("fixed", "lo", "hi", "min_edge")])
        for m in METHODS:
            row = [m]
            for n in names:
                pt, lo, hi = skills[n][m]
                row += [f"{pt:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{bench[m][n]:.4f}"]
            w.writerow(row)

    with open("results/fixed_extent_control.md", "w") as fh:
        fh.write("# Fixed-extent control (crop-size confound)\n\n")
        fh.write("Same benchmark, extents fixed to 0.25/0.35 of the short side -- true and\n"
                 "wrong crops of an image are IDENTICAL in size. fixed = this control;\n"
                 "bench = the published min_edge/bbox skill.\n\n")
        fh.write("| method | " + " | ".join(f"{n} fixed (bench)" for n in names)
                 + " | mean fixed (bench) |\n")
        fh.write("|" + "---|" * (len(names) + 2) + "\n")
        for m in METHODS:
            cells = " | ".join(f"{skills[n][m][0]:+.2f} ({bench[m][n]:+.2f})"
                               for n in names)
            mf = np.mean([skills[n][m][0] for n in names])
            mb = np.mean([bench[m][n] for n in names])
            fh.write(f"| {m} | {cells} | {mf:+.2f} ({mb:+.2f}) |\n")

    for m in METHODS:
        mf = np.mean([skills[n][m][0] for n in names])
        mb = np.mean([bench[m][n] for n in names])
        print(f"  {m:<20} fixed {mf:+.3f}   bench {mb:+.3f}   delta {mf-mb:+.3f}")
    print("wrote results/fixed_extent_control.{csv,md}", flush=True)


if __name__ == "__main__":
    main()
