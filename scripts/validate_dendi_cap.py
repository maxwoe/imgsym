"""Validate the 256px crop cap on DENDI: same subset, native vs capped, per-method skill.

Background: the cap (``run_discrimination.py:_cap_dim``) downscales any crop whose longest
side exceeds 256px. It is a *tractability* measure for DENDI's large COCO images -- crops up
to ~830px stall the per-pixel classical scorers -- and is applied to the multi protocol only.
This script checks the cap is *scientifically neutral*: it scores a fixed DENDI subset BOTH
at native resolution and capped, with the identical benchmark pipeline, and compares per-method
discrimination skill and ranking. If skills barely move and the ranking (especially
deep_features ~ hog) is preserved, the cap changes runtime, not conclusions -- and we report it.

    python scripts/validate_dendi_cap.py [--images 40]

Writes results/dendi_cap_validation.{md,csv}. Self-contained; reuses the exact perturbation
model, standard local levels, and cap function from run_discrimination.py (imported), so the
only variable between the two passes is the cap.
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_discrimination import (tagged_wrong_axes, std_local_pairs, _cap_dim,
                                METHOD_KWARGS, DATA_ROOT)

from imgsym.scoring.calculators import SymmetryCalculatorFactory as Factory
from imgsym.evaluation import (extract, discrimination_skill_ci, paired_skill_diff,
                               load_dendi_reflection)


def score_subset(units, calcs, cap_on):
    """{method: [ {"true":.., "wrong":[[tag, score|None], ..]}, .. ]} for the subset.

    cap_on=True applies the 256px cap to every crop; False = native resolution. Extraction and
    the (local) perturbation set are identical across passes, so only resolution differs."""
    raw = {m: [] for m in calcs}
    t0 = time.time()
    for idx, u in enumerate(units):
        img = cv2.imread(u.image_path)
        if img is None:
            continue
        rt = extract(img, u.axis, policy="min_edge", min_support_frac=0.10)
        if rt.info.degenerate:
            continue
        true_sub = _cap_dim(rt.subimage) if cap_on else rt.subimage
        wsubs = []
        for tag, wax in tagged_wrong_axes(u.axis, img.shape, idx + 1, use_global=False):
            r = extract(img, wax, policy="min_edge", min_support_frac=0.10)
            sub = None if r.info.degenerate else (_cap_dim(r.subimage) if cap_on else r.subimage)
            wsubs.append((tag, sub))
        for m, c in calcs.items():
            try:
                t = float(c.calculate_score(true_sub))
            except Exception:
                continue
            if not np.isfinite(t):
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
            raw[m].append({"img": idx, "true": t, "wrong": wrong})
        if (idx + 1) % 10 == 0:
            print(f"  [{'capped' if cap_on else 'native'}] {idx + 1}/{len(units)} units "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=40, help="DENDI images (max_axes=10 cap)")
    args = ap.parse_args()

    methods = Factory.list_methods()
    print("loading calculators...", flush=True)
    calcs = {m: Factory.create(m, **METHOD_KWARGS.get(m, {})) for m in methods}

    ds = load_dendi_reflection(os.path.join(DATA_ROOT, "dendi"), max_axes=10)
    units = list(ds.all_axes())
    # Fix the subset by image so native & capped see byte-identical units.
    seen = set()
    for u in units:
        if u.image_path not in seen:
            seen.add(u.image_path)
        if len(seen) >= args.images:
            break
    sub_units = [u for u in units if u.image_path in seen]
    print(f"DENDI subset: {len(seen)} images, {len(sub_units)} axes\n", flush=True)

    results, raws = {}, {}
    for label, cap_on in [("native", False), ("capped", True)]:
        print(f"scoring {label}...", flush=True)
        t0 = time.time()
        raws[label] = score_subset(sub_units, calcs, cap_on)
        results[label] = {m: discrimination_skill_ci(std_local_pairs(raws[label][m]))
                          for m in methods}
        print(f"  {label} done ({time.time() - t0:.0f}s)\n", flush=True)

    # Paired capped-minus-native diff on the units both passes scored (stronger
    # than comparing two independent CIs on the same data).
    paired = {}
    for m in methods:
        by_img = {r["img"]: r for r in raws["native"][m]}
        na, ca = [], []
        for r in raws["capped"][m]:
            if r["img"] in by_img:
                na.append(by_img[r["img"]])
                ca.append(r)
        paired[m] = paired_skill_diff(std_local_pairs(ca), std_local_pairs(na)) \
            if len(std_local_pairs(ca)) == len(std_local_pairs(na)) else None

    order = sorted(methods, key=lambda m: results["native"][m][0], reverse=True)

    os.makedirs("results", exist_ok=True)
    with open("results/dendi_cap_validation.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "native_skill", "native_lo", "native_hi",
                    "capped_skill", "capped_lo", "capped_hi", "delta",
                    "paired_diff", "paired_lo", "paired_hi"])
        for m in order:
            ns, nlo, nhi = results["native"][m]
            cs, clo, chi = results["capped"][m]
            pd_ = paired[m] or (float("nan"),) * 3
            w.writerow([m, f"{ns:.4f}", f"{nlo:.4f}", f"{nhi:.4f}",
                        f"{cs:.4f}", f"{clo:.4f}", f"{chi:.4f}", f"{cs - ns:+.4f}",
                        f"{pd_[0]:+.4f}", f"{pd_[1]:+.4f}", f"{pd_[2]:+.4f}"])

    with open("results/dendi_cap_validation.md", "w") as fh:
        fh.write(f"# DENDI crop-cap validation ({len(seen)} images, {len(sub_units)} axes)\n\n")
        fh.write("Same subset scored at native resolution vs the 256px cap; discrimination "
                 "skill ($2\\cdot$AUC$-1$), 95% CI. delta = capped $-$ native.\n\n")
        fh.write("| method | native | capped | delta | paired capped-native [95% CI] |\n"
                 "|---|---|---|---|---|\n")
        for m in order:
            ns, nlo, nhi = results["native"][m]
            cs, clo, chi = results["capped"][m]
            pd_ = paired[m]
            pcell = (f"{pd_[0]:+.3f} [{pd_[1]:+.3f},{pd_[2]:+.3f}]"
                     + (" sig" if (pd_[1] > 0 or pd_[2] < 0) else " n.s.")) if pd_ else "n/a"
            fh.write(f"| {m} | {ns:+.3f} [{nlo:+.3f},{nhi:+.3f}] "
                     f"| {cs:+.3f} [{clo:+.3f},{chi:+.3f}] | {cs - ns:+.3f} | {pcell} |\n")

    def rank(label):
        return sorted(methods, key=lambda m: results[label][m][0], reverse=True)
    rn, rc = rank("native"), rank("capped")
    deltas = [abs(results["capped"][m][0] - results["native"][m][0]) for m in methods]
    print("native top-4:", " > ".join(f"{m} {results['native'][m][0]:.2f}" for m in rn[:4]))
    print("capped top-4:", " > ".join(f"{m} {results['capped'][m][0]:.2f}" for m in rc[:4]))
    print(f"max |delta| = {max(deltas):.3f}, mean |delta| = {np.mean(deltas):.3f}")
    print(f"top-4 ranking preserved: {rn[:4] == rc[:4]}")
    print("wrote results/dendi_cap_validation.{md,csv}", flush=True)


if __name__ == "__main__":
    main()
