"""Rebuild the global negatives of an EXISTING per_image_scores.json and flag
those that near-coincide with the true axis (no re-scoring needed).

Why: 15-41% of single-axis images have a near-centered, near-vertical true axis,
where the cardinal 'vertical center' global negative essentially IS the truth --
scoring it as a negative punishes correct methods in the local-vs-global table.
New runs exclude such negatives at generation time (``hard_negative_axes(avoid=)``)
but re-scoring the benchmark is expensive; this script instead REPLAYS the runner's
deterministic unit enumeration (same loaders, same order, same extraction gate,
same per-unit seed) to reconstruct each used unit's 12 global axes, and writes a
keep-mask aligned with the stored records:

    results/global_coincidence_mask.json
      {dataset: [[keep_bool x 12] per stored record, ...]}

analyze_discrimination.py picks the mask up automatically for section 2.

    python scripts/build_global_mask.py [results/per_image_scores.json]
"""
import json
import sys

import cv2

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from run_discrimination import _configs, DATA_ROOT

from imgsym.evaluation import extract, hard_negative_axes
from imgsym.evaluation.scoring_metrics import axis_is_near


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/per_image_scores.json"
    blob = json.load(open(path))
    if blob["meta"].get("protocol", "single") != "single" or not blob["meta"].get("use_global"):
        raise SystemExit("mask only applies to the single protocol with global negatives")
    data = blob["data"]

    masks = {}
    for name, loader, policy in _configs(DATA_ROOT, "single", max_axes=10):
        if name not in data:
            continue
        ds = loader()
        units = list(ds.dominant_axes())
        ds_masks = []
        for idx, u in enumerate(units):          # replay the runner's enumeration exactly
            img = cv2.imread(u.image_path)
            if img is None:
                continue
            rt = extract(img, u.axis, policy=policy, min_support_frac=0.10)
            if rt.info.degenerate:
                continue
            negs = hard_negative_axes(img.shape, seed=idx + 1)   # avoid=None = as stored
            ds_masks.append([not axis_is_near(g, u.axis, img.shape) for g in negs])
        n_rec = len(data[name][next(iter(data[name]))])
        if len(ds_masks) != n_rec:
            raise SystemExit(f"{name}: replay produced {len(ds_masks)} units, "
                             f"JSON has {n_rec} -- enumeration drifted, aborting")
        dropped = sum(12 - sum(m) for m in ds_masks)
        print(f"[{name}] {len(ds_masks)} units, {dropped}/{12*len(ds_masks)} "
              f"coincident negatives flagged", flush=True)
        masks[name] = ds_masks

    with open("results/global_coincidence_mask.json", "w") as fh:
        json.dump(masks, fh)
    print("wrote results/global_coincidence_mask.json")


if __name__ == "__main__":
    main()
