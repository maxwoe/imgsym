"""AlexNet layer robustness: conv1 (our benchmark config) vs conv2 (the source's
general recommendation).

We benchmark Brachmann & Redies at conv1 (their layer-1 configuration at its
reported optimum, 17x17 grid) -- but their paper recommends conv2 for general
images. This scores the SAME single-axis benchmark at conv2 (torchvision AlexNet
features[0:5]: conv1-relu-pool-conv2-relu; the source's CaffeNet also has LRN,
absent from torchvision -- same gloss our conv1 config already makes) at two
pooling grids (17x17 as ours; 9x9 as a coarser-grid sensitivity, since their
per-layer grid optima differ), alongside a conv1 re-score under identical
conditions. Answers whether "hog ties AlexNet-C1" depends on the layer choice.

    python scripts/validate_alexnet_layer.py [--limit N]

Writes results/alexnet_layer_check.{csv,md}.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_discrimination import _configs, run_dataset, std_local_pairs, DATA_ROOT

from imgsym.scoring.calculators import AlexNetCalculator
from imgsym.evaluation import discrimination_skill_ci, discrimination_skill


class AlexNetConv2(AlexNetCalculator):
    """AlexNet scored at conv2 (features[0:5]); everything else identical."""

    def __init__(self, patches=17, target_size=512):
        super().__init__(patches=patches, target_size=target_size)
        from torchvision import models
        from torchvision.models.alexnet import AlexNet_Weights
        model = models.alexnet(weights=AlexNet_Weights.DEFAULT).eval()
        self.stack = model.features[0:5]        # conv1, relu, maxpool, conv2, relu

    def _get_features(self, bgr_image):
        x = self._preprocess(bgr_image)
        with self.torch.no_grad():
            return self.stack(x)


CONFIGS = {
    # layer=1 explicitly: the library default moved to the source's general
    # recommendation (conv2 @ 11x11) on 2026-07-03.
    "conv1_p17_ours": lambda: AlexNetCalculator(patches=17, layer=1),
    "conv2_p17": lambda: AlexNetConv2(patches=17),
    "conv2_p9": lambda: AlexNetConv2(patches=9),
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

    print("loading calculators...", flush=True)
    calcs = {name: make() for name, make in CONFIGS.items()}

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
        loc, glob = [], []
        for n in names:
            loc += _slice(raw_all[cfg][n], lambda t: t.startswith(("shift_", "rot_")))
            glob += _slice(raw_all[cfg][n], lambda t: t == "global")
        rows.append((cfg, per_ds,
                     float(np.mean([per_ds[n][0] for n in names])),
                     discrimination_skill(loc), discrimination_skill(glob)))

    os.makedirs("results", exist_ok=True)
    with open("results/alexnet_layer_check.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config"] + [f"{n}_skill" for n in names]
                   + ["mean_skill", "pooled_local", "pooled_global"])
        for cfg, per_ds, mean_sk, loc, glob in rows:
            w.writerow([cfg] + [f"{per_ds[n][0]:.4f}" for n in names]
                       + [f"{mean_sk:.4f}", f"{loc:.4f}", f"{glob:.4f}"])

    with open("results/alexnet_layer_check.md", "w") as fh:
        fh.write("# AlexNet layer check: conv1 (benchmark) vs conv2 (source recommendation)\n\n")
        fh.write("Same benchmark pipeline; conv2 = torchvision features[0:5]; global\n"
                 "negatives coincidence-filtered.\n\n")
        fh.write("| config | " + " | ".join(names) + " | mean | local | global |\n")
        fh.write("|" + "---|" * (len(names) + 4) + "\n")
        for cfg, per_ds, mean_sk, loc, glob in rows:
            cells = " | ".join(f"{per_ds[n][0]:+.2f}" for n in names)
            fh.write(f"| {cfg} | {cells} | {mean_sk:+.2f} | {loc:+.2f} | {glob:+.2f} |\n")

    for cfg, per_ds, mean_sk, loc, glob in rows:
        print(f"  {cfg:<16} mean {mean_sk:+.3f}   local {loc:+.3f}   global {glob:+.3f}")
    print("wrote results/alexnet_layer_check.{csv,md}", flush=True)


if __name__ == "__main__":
    main()
