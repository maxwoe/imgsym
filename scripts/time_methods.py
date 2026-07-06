"""Per-method scoring cost for the speed-vs-accuracy frontier.

Times ``calculate_score`` for all 13 methods on true-axis crops drawn evenly from the five
single-axis benchmark sets (the same native-resolution crops the skill numbers come from, so
speed and accuracy describe the same operating point). CPU, single process, method-by-method,
2 warmup calls per method (excludes model load / lazy init); the MEDIAN per-crop time is the
headline (robust to OS scheduling stalls).

    python scripts/time_methods.py [--per-dataset 12]

Writes results/method_timings.csv (method, n_crops, median_s, mean_s, p25_s, p75_s).
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_discrimination import _configs, METHOD_KWARGS, DATA_ROOT

from imgsym.scoring.calculators import SymmetryCalculatorFactory as Factory
from imgsym.evaluation import extract


def collect_crops(per_dataset):
    """Native-resolution true-axis crops, ``per_dataset`` from each single-axis set."""
    crops = []
    for name, loader, policy in _configs(DATA_ROOT, "single", max_axes=10):
        try:
            ds = loader()
        except (FileNotFoundError, OSError):
            continue
        n = 0
        for u in ds.dominant_axes():
            img = cv2.imread(u.image_path)
            if img is None:
                continue
            rt = extract(img, u.axis, policy=policy, min_support_frac=0.10)
            if rt.info.degenerate:
                continue
            crops.append(rt.subimage)
            n += 1
            if n >= per_dataset:
                break
        print(f"  [{name}] {n} crops", flush=True)
    return crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=12)
    args = ap.parse_args()

    print("collecting crops...", flush=True)
    crops = collect_crops(args.per_dataset)
    dims = [max(c.shape[:2]) for c in crops]
    print(f"{len(crops)} crops (max-dim median {int(np.median(dims))}, "
          f"range {min(dims)}-{max(dims)})\n", flush=True)

    print("loading calculators...", flush=True)
    methods = Factory.list_methods()
    calcs = {m: Factory.create(m, **METHOD_KWARGS.get(m, {})) for m in methods}

    rows = []
    for m, c in calcs.items():
        for w in crops[:2]:                        # warmup: lazy init / first-call caches
            try:
                c.calculate_score(w)
            except Exception:
                pass
        times = []
        for sub in crops:
            t0 = time.perf_counter()
            try:
                c.calculate_score(sub)
            except Exception:
                continue
            times.append(time.perf_counter() - t0)
        t = np.array(times)
        rows.append((m, len(t), float(np.median(t)), float(t.mean()),
                     float(np.percentile(t, 25)), float(np.percentile(t, 75))))
        print(f"  {m:<22} median {np.median(t)*1e3:8.1f} ms  "
              f"[{np.percentile(t,25)*1e3:.1f}, {np.percentile(t,75)*1e3:.1f}]  n={len(t)}",
              flush=True)

    os.makedirs("results", exist_ok=True)
    with open("results/method_timings.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n_crops", "median_s", "mean_s", "p25_s", "p75_s"])
        for r in rows:
            w.writerow([r[0], r[1]] + [f"{v:.6f}" for v in r[2:]])
    print("\nwrote results/method_timings.csv", flush=True)


if __name__ == "__main__":
    main()
