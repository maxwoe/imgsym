"""Post-hoc analyses of results/per_image_scores.json (no re-scoring).

Reads the raw per-image scores stored by run_discrimination.py and reports:
  1. Sensitivity curve  -- skill vs perturbation magnitude (where methods separate)
  2. Local vs global    -- precise localization vs rejecting gross alternatives
  3. Paired significance -- proper paired-bootstrap tests on close method pairs

Statistical hygiene:
  * Score DIRECTION is decided once per method from its standard-tag pool and held
    fixed for every slice -- re-inferring per slice silently flips the sign of
    near-chance cells (a per-slice max over directions).
  * Paired tests JOIN records on unit identity ("img", stored by the runner) when
    present; otherwise they require identical per-dataset counts for all methods.
  * If results/global_coincidence_mask.json exists (scripts/build_global_mask.py),
    global negatives that near-coincide with the true axis are EXCLUDED from the
    local-vs-global table (on centered subjects the cardinal center line can BE
    the truth; new runs already exclude them at generation time via ``avoid``).

    python scripts/analyze_discrimination.py [results/per_image_scores.json]
"""
import json
import os
import sys

import numpy as np

from imgsym.evaluation import discrimination_skill, paired_skill_diff
from imgsym.evaluation.scoring_metrics import infer_direction


def _pairs(per_image, pred):
    """(true, [wrong scores]) keeping wrongs whose tag passes pred(tag); drop None."""
    out = []
    for rec in per_image:
        wr = [s for tag, s in rec["wrong"] if pred(tag) and s is not None]
        if wr:
            out.append((rec["true"], wr))
    return out


def _pooled(data, method, pred):
    """Concatenate per-image (true, wrongs) across all datasets for one method."""
    out = []
    for ds in data:
        out += _pairs(data[ds][method], pred)
    return out


def _check_paired_alignment(data, a, b):
    """Methods must cover identical units for a positional paired test."""
    for ds in data:
        ra, rb = data[ds][a], data[ds][b]
        if len(ra) != len(rb):
            raise SystemExit(f"paired test invalid: {a}/{b} cover different units in {ds} "
                             f"({len(ra)} vs {len(rb)}); join on 'img' required")
        # Identity check needs ids on BOTH sides; legacy records (pre-rigor-pass
        # runs, and the 2026-07-03 splices into them) may lack 'img' -- for those
        # we fall back to the length check plus deterministic loader order.
        if ra and rb and "img" in ra[0] and "img" in rb[0]:
            ia, ib = [r["img"] for r in ra], [r["img"] for r in rb]
            if ia != ib:
                raise SystemExit(f"paired test invalid: {a}/{b} unit identity differs in {ds}")


def _global_pairs(per_image, mask):
    """Local-vs-global 'global' slice, excluding mask-flagged coincident negatives.

    mask[i] = list of 12 bools (keep) aligned with the unit's global entries, or
    None -> keep all."""
    out = []
    for i, rec in enumerate(per_image):
        keep = mask[i] if mask else None
        g = [(tag, s) for tag, s in rec["wrong"] if tag == "global"]
        wr = [s for j, (tag, s) in enumerate(g)
              if s is not None and (keep is None or (j < len(keep) and keep[j]))]
        if wr:
            out.append((rec["true"], wr))
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/per_image_scores.json"
    with open(path) as fh:
        blob = json.load(fh)
    meta, data = blob["meta"], blob["data"]
    methods = list(next(iter(data.values())).keys())
    shifts, angles = meta["shifts"], meta["angles"]
    std_tags = set(meta["std_tags"])

    # Direction per method, decided ONCE from the standard pool, fixed for all slices.
    direction, overall = {}, {}
    for m in methods:
        pool = _pooled(data, m, lambda t: t in std_tags)
        direction[m] = infer_direction([t for t, _ in pool],
                                       [x for _, w in pool for x in w])
        overall[m] = discrimination_skill(pool, higher_is_better=direction[m])
    methods = sorted(methods, key=lambda m: -overall[m])

    print("=== 1. SENSITIVITY: skill vs perturbation magnitude (pooled; 0=chance, 1=perfect) ===")
    print("    finer perturbations on the left -> where methods reveal their precision limit")
    print("    (direction fixed per method from the standard pool)\n")
    head = "method".ljust(20) + "".join(f"{f*100:g}%".rjust(7) for f in shifts) + "  ||" \
        + "".join(f"{a:g}d".rjust(7) for a in angles)
    print(head)
    for m in methods:
        row = m.ljust(20)
        for f in shifts:
            tag = f"shift_{f:g}"
            sk = discrimination_skill(_pooled(data, m, lambda t, x=tag: t == x),
                                      higher_is_better=direction[m])
            row += f"{sk:+.2f}".rjust(7)
        row += "  ||"
        for a in angles:
            tag = f"rot_{a:g}"
            sk = discrimination_skill(_pooled(data, m, lambda t, x=tag: t == x),
                                      higher_is_better=direction[m])
            row += f"{sk:+.2f}".rjust(7)
        print(row)

    if meta.get("use_global"):
        mask_path = "results/global_coincidence_mask.json"
        masks = json.load(open(mask_path)) if os.path.exists(mask_path) else None
        note = " (coincident negatives excluded)" if masks else ""
        print(f"\n=== 2. LOCAL vs GLOBAL skill (pooled){note} ===")
        print("    local = precise localization (hard);  global = reject gross alternatives (easy)\n")
        print("method".ljust(20) + "local".rjust(8) + "global".rjust(8))
        for m in methods:
            loc = discrimination_skill(
                _pooled(data, m, lambda t: t.startswith(("shift_", "rot_"))),
                higher_is_better=direction[m])
            gp = []
            for ds in data:
                gp += _global_pairs(data[ds][m], masks.get(ds) if masks else None)
            glob = discrimination_skill(gp, higher_is_better=direction[m])
            print(f"{m.ljust(20)}{loc:+8.2f}{glob:+8.2f}")
        if masks:
            # Count only datasets present in the data (the mask file may retain
            # datasets since dropped from the benchmark, e.g. symComp13_s).
            used = [masks[ds] for ds in data if ds in masks]
            dropped = sum(sum(1 for k in unit if not k) for ds in used for unit in ds)
            total = sum(len(unit) for ds in used for unit in ds)
            print(f"    ({dropped}/{total} global negatives excluded as near-coincident "
                  "with the true axis)")

    print("\n=== 3. PAIRED significance (pooled, standard local; CI excludes 0 => significant) ===")
    print("    paired bootstrap on the SAME images -- more powerful than CI overlap\n")
    ranked = methods  # already sorted best-first
    # Adjacent ladder pairs, plus the full triangle among the top three (the
    # skipped top-1 vs top-3 comparison is load-bearing for the manuscript).
    compares = [(ranked[i], ranked[i + 1]) for i in range(min(5, len(ranked) - 1))]
    if len(ranked) >= 3 and (ranked[0], ranked[2]) not in compares:
        compares.insert(1, (ranked[0], ranked[2]))
    for a, b in compares:
        _check_paired_alignment(data, a, b)
        d, lo, hi = paired_skill_diff(_pooled(data, a, lambda t: t in std_tags),
                                      _pooled(data, b, lambda t: t in std_tags))
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."
        print(f"  {a:20} - {b:20} = {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  {sig}")


if __name__ == "__main__":
    main()
