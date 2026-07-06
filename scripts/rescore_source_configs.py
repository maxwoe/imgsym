"""Rescore alexnet + local_global at their sources' recommended configurations.

Policy (2026-07-03): every reimplemented method runs at the source's
recommended configuration at its reported optimum. This moves
  - alexnet:      conv1 @ 17x17  ->  conv2 @ 11x11  (Brachmann & Redies' stated
                  general recommendation, at their reported conv2 optimum), and
  - local_global: m=31/w=31      ->  m=15/w=17.7    (Hogeweg et al.'s grid
                  optimum), k=4 (their kappa=16 stride),
i.e. the new library defaults, and splices the rescored records into the master
per-image artifacts so every downstream analysis sees one consistent basis.

Consistency guards:
  - hard_negative_axes is called WITHOUT ``avoid=`` (legacy behavior) so the
    rescored methods score the IDENTICAL global-negative sets stored for the
    other 11 methods; coincident globals are excluded at analysis time via the
    replayed mask, as for all methods.
  - per-dataset record counts must match the stored records exactly (assert).

Rewrites discrimination{,_multi}.{csv,md} with the same code path as
run_discrimination.py.

    python scripts/rescore_source_configs.py --protocol single
    python scripts/rescore_source_configs.py --protocol multi
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_discrimination as rd
from run_discrimination import _configs, run_dataset, std_local_pairs, DATA_ROOT, METHOD_KWARGS

from imgsym.scoring.calculators import SymmetryCalculatorFactory as Factory
from imgsym.evaluation import discrimination_skill_ci, hard_negative_axes as _hna
from imgsym.evaluation import separation_margin_mean

RESCORE = ["alexnet", "local_global"]

# Legacy global-negative generation (no avoid=): reproduce the exact negative
# sets the stored 11 methods were scored on.
rd.hard_negative_axes = lambda shape, seed=0, avoid=None: _hna(shape, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["single", "multi"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    tag = "" if args.protocol == "single" else "_multi"
    pij = f"results/per_image_scores{tag}.json"

    blob = json.load(open(pij))
    meta = blob["meta"]
    use_global = bool(meta.get("use_global", args.protocol == "single"))
    max_axes = int(meta.get("max_axes", 10))

    print("loading calculators...", flush=True)
    calcs = {m: Factory.create(m, **METHOD_KWARGS.get(m, {})) for m in RESCORE}
    a, lg = calcs["alexnet"], calcs["local_global"]
    print(f"alexnet: layer={a.layer} patches={a.patches} | "
          f"local_global: m={lg.m} w={lg.w} k={lg.k} | use_global={use_global}", flush=True)
    assert (a.layer, a.patches) == (2, 11) and (lg.m, lg.w, lg.k) == (15, 17.7, 4), \
        "library defaults are not the source-recommended configs"

    for name, loader, policy in _configs(DATA_ROOT, args.protocol, max_axes):
        if name not in blob["data"]:
            continue
        ds = loader()
        print(f"[{name}] rescoring {RESCORE} ...", flush=True)
        raw = run_dataset(name, ds, policy, calcs, use_global, args.protocol, args.limit)
        for m in RESCORE:
            old, new = blob["data"][name][m], raw[m]
            assert len(old) == len(new), (name, m, len(old), len(new))
            blob["data"][name][m] = new
        with open(pij, "w") as fh:                 # incremental save
            json.dump(blob, fh)
        print(f"[{name}] spliced + saved", flush=True)

    # Rewrite the summary CSV/MD over ALL methods from the updated blob
    # (same logic as run_discrimination.main).
    data = blob["data"]
    names = list(data)
    methods = list(next(iter(data.values())).keys())
    skill = {n: {m: discrimination_skill_ci(std_local_pairs(data[n][m])) for m in methods}
             for n in names}
    margin = {n: {m: separation_margin_mean(std_local_pairs(data[n][m])) for m in methods}
              for n in names}

    with open(f"results/discrimination{tag}.csv", "w", newline="") as fh:
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
    with open(f"results/discrimination{tag}.md", "w") as fh:
        fh.write(f"# Discrimination skill ({args.protocol} protocol, standard local levels), "
                 "95% CI; (m)=separation margin\n\n")
        fh.write("| method | " + " | ".join(names) + " | mean skill |\n")
        fh.write("|" + "---|" * (len(names) + 2) + "\n")
        for m in order:
            cells = " | ".join(
                f"{skill[n][m][0]:+.2f} [{skill[n][m][1]:+.2f},{skill[n][m][2]:+.2f}] (m{margin[n][m]:.2f})"
                for n in names)
            fh.write(f"| {m} | {cells} | {mean_skill(m):+.2f} |\n")

    print(f"updated per_image_scores{tag}.json + discrimination{tag}.csv/md", flush=True)
    for m in RESCORE:
        print(f"  {m}: mean {mean_skill(m):+.3f}", flush=True)
    print("top of leaderboard:", " > ".join(f"{m} {mean_skill(m):+.3f}" for m in order[:4]), flush=True)


if __name__ == "__main__":
    main()
