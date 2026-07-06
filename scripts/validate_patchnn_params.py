"""PatchNN (local_global) parameter robustness: our params vs Hogeweg's.

Our benchmark scores local_global at OUR parameters (patch m=31, position weight
w=31, stride k=4), which are neither Hogeweg et al.'s default (m=9, w=10) nor
their grid optimum (m=15, w=17.7 at kappa=16). A reviewer could ask whether the
method's mid-pack rank is parameter-driven. This scores the SAME single-axis
benchmark at their optimum and their default (both at k=4 -- our stride matches
their kappa=16 condition; their default's full sampling would be ~16x the patches
and intractable under brute-force NN, so the fixed-sampling comparison is the
fair one) and compares skills + the local-vs-global sanity check against the
stored m=31/w=31 results.

    python scripts/validate_patchnn_params.py [--limit N]

Writes results/patchnn_param_check.{csv,md}.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_discrimination import _configs, run_dataset, std_local_pairs, DATA_ROOT

from imgsym.scoring.calculators import LocalGlobalSymmetryCalculator
from imgsym.evaluation import discrimination_skill_ci, discrimination_skill

CONFIGS = {
    "m31_w31_ours": dict(m=31, k=4, w=31.0),
    "m15_w17.7_optimum": dict(m=15, k=4, w=17.7),
    "m9_w10_default": dict(m=9, k=4, w=10.0),
}


def _slice(per_image, pred):
    out = []
    for rec in per_image:
        wr = [s for tag, s in rec["wrong"] if pred(tag) and s is not None]
        if wr:
            out.append((rec["true"], wr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    calcs = {name: LocalGlobalSymmetryCalculator(**kw) for name, kw in CONFIGS.items()}

    raw_all = {name: {} for name in CONFIGS}
    for ds_name, loader, policy in _configs(DATA_ROOT, "single", max_axes=10):
        try:
            ds = loader()
        except (FileNotFoundError, OSError) as exc:
            print(f"[{ds_name}] SKIP ({exc})", flush=True)
            continue
        print(f"[{ds_name}] {len(ds)} imgs", flush=True)
        raw = run_dataset(ds_name, ds, policy, calcs, use_global=True,
                          protocol="single", limit=args.limit)
        for name in CONFIGS:
            raw_all[name][ds_name] = raw[name]

    names = list(next(iter(raw_all.values())).keys())
    rows = []
    for cfg in CONFIGS:
        per_ds = {n: discrimination_skill_ci(std_local_pairs(raw_all[cfg][n]))
                  for n in names}
        pooled_loc, pooled_glob = [], []
        for n in names:
            pooled_loc += _slice(raw_all[cfg][n], lambda t: t.startswith(("shift_", "rot_")))
            pooled_glob += _slice(raw_all[cfg][n], lambda t: t == "global")
        rows.append((cfg, per_ds,
                     float(np.mean([per_ds[n][0] for n in names])),
                     discrimination_skill(pooled_loc), discrimination_skill(pooled_glob)))

    os.makedirs("results", exist_ok=True)
    with open("results/patchnn_param_check.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config"] + [f"{n}_skill" for n in names]
                   + ["mean_skill", "pooled_local", "pooled_global"])
        for cfg, per_ds, mean_sk, loc, glob in rows:
            w.writerow([cfg] + [f"{per_ds[n][0]:.4f}" for n in names]
                       + [f"{mean_sk:.4f}", f"{loc:.4f}", f"{glob:.4f}"])

    with open("results/patchnn_param_check.md", "w") as fh:
        fh.write("# PatchNN (local_global) parameter check: ours vs Hogeweg's\n\n")
        fh.write("All at stride k=4 (their kappa=16 condition); global negatives use the\n"
                 "coincidence-filtered generator, comparable to the corrected analysis.\n\n")
        fh.write("| config | " + " | ".join(names) + " | mean | local | global |\n")
        fh.write("|" + "---|" * (len(names) + 4) + "\n")
        for cfg, per_ds, mean_sk, loc, glob in rows:
            cells = " | ".join(f"{per_ds[n][0]:+.2f}" for n in names)
            fh.write(f"| {cfg} | {cells} | {mean_sk:+.2f} | {loc:+.2f} | {glob:+.2f} |\n")

    for cfg, per_ds, mean_sk, loc, glob in rows:
        print(f"  {cfg:<20} mean {mean_sk:+.3f}   local {loc:+.3f}   global {glob:+.3f}")
    print("wrote results/patchnn_param_check.{csv,md}", flush=True)


if __name__ == "__main__":
    main()
