"""Free post-hoc analyses from results/per_image_scores.json (no re-scoring):

  1. Ranking stability -- is the method order consistent across datasets?
  2. Complementarity   -- which methods succeed on different images (ensemble potential)?
  3. Ensemble          -- does rank-combining methods beat the best single method?
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr

from imgsym.evaluation import (discrimination_skill, discrimination_one_sided,
                               infer_direction)

STD_TAGS = ({f"shift_{f:g}" for f in (0.03, 0.05, 0.1)} |
            {f"rot_{a:g}" for a in (3.0, 5.0, 10.0)})


def std_pairs(per_image):
    out = []
    for rec in per_image:
        w = [s for tag, s in rec["wrong"] if tag in STD_TAGS and s is not None]
        out.append((rec["true"], w) if w else None)
    return out


def pooled_pairs(data, m):
    out = []
    for ds in data:
        out += [p for p in std_pairs(data[ds][m]) if p is not None]
    return out


def per_image_disc(data, m):
    """Aligned per-image one-sided discrimination vector (NaN where no wrongs)."""
    hib = infer_direction(*_pool_scores(data, m))
    vals = []
    for ds in data:
        for p in std_pairs(data[ds][m]):
            vals.append(discrimination_one_sided(p[0], p[1], hib) if p else np.nan)
    return np.array(vals)


def _pool_scores(data, m):
    trues, wrongs = [], []
    for ds in data:
        for p in std_pairs(data[ds][m]):
            if p:
                trues.append(p[0])
                wrongs.extend(p[1])
    return trues, wrongs


def aligned_axes(data, ds, i, methods):
    recs = {m: data[ds][m][i] for m in methods}
    base = recs[methods[0]]["wrong"]
    js = [j for j, (tag, _) in enumerate(base) if tag in STD_TAGS]
    valid = [j for j in js if all(recs[m]["wrong"][j][1] is not None for m in methods)]
    if not valid:
        return None
    return {m: [recs[m]["true"]] + [recs[m]["wrong"][j][1] for j in valid] for m in methods}


def ensemble_skill(data, methods):
    per_image = []
    for ds in data:
        for i in range(len(data[ds][methods[0]])):
            ax = aligned_axes(data, ds, i, methods)
            if ax is None:
                continue
            n_ax = len(next(iter(ax.values())))
            ens = np.zeros(n_ax)
            for m in methods:
                sc = np.asarray(ax[m], float)
                ens += sc.argsort().argsort() / max(n_ax - 1, 1)   # within-image rank
            ens /= len(methods)
            per_image.append((float(ens[0]), [float(x) for x in ens[1:]]))
    return discrimination_skill(per_image)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/per_image_scores.json"
    data = json.load(open(path))["data"]
    names = list(data)
    methods = list(data[names[0]].keys())
    overall = {m: discrimination_skill(pooled_pairs(data, m)) for m in methods}
    methods = sorted(methods, key=lambda m: -overall[m])

    print("=== 1. RANKING STABILITY across datasets (Spearman of method-skill vectors) ===")
    ds_skill = {ds: np.array([discrimination_skill([p for p in std_pairs(data[ds][m]) if p])
                              for m in methods]) for ds in names}
    rhos = [spearmanr(ds_skill[a], ds_skill[b]).statistic
            for i, a in enumerate(names) for b in names[i + 1:]]
    print(f"    mean pairwise rank-corr = {np.mean(rhos):.3f} "
          f"(min {np.min(rhos):.3f})  -> {'very stable' if np.mean(rhos) > 0.85 else 'moderately stable'}\n")

    print("=== 2. COMPLEMENTARITY (per-image success correlation; low = complementary) ===")
    top = methods[:6]
    disc = {m: per_image_disc(data, m) for m in top}
    print("    " + "".join(m[:7].rjust(8) for m in top))
    for a in top:
        row = a[:18].ljust(20)
        for b in top:
            mask = np.isfinite(disc[a]) & np.isfinite(disc[b])
            r = np.corrcoef(disc[a][mask], disc[b][mask])[0, 1] if mask.sum() > 2 else np.nan
            row += f"{r:+.2f}".rjust(8)
        print(row)

    print("\n=== 3. ENSEMBLE (rank-combine; does it beat the best single?) ===")
    best = methods[0]
    base = overall[best]
    print(f"    best single: {best} = {base:+.3f}")
    combos = [
        methods[:2], methods[:3], [methods[0], methods[2]],          # deep+alex, +hog...
        ["deep_features", "hog", "gabor"], ["deep_features", "hog", "weighted_binary"],
        methods[:5], methods,
    ]
    seen = set()
    for combo in combos:
        combo = [m for m in combo if m in methods]
        key = tuple(combo)
        if len(combo) < 2 or key in seen:
            continue
        seen.add(key)
        sk = ensemble_skill(data, combo)
        flag = f"  (+{sk-base:.3f} vs best)" if sk > base else f"  ({sk-base:+.3f})"
        print(f"    {'+'.join(combo):50} = {sk:+.3f}{flag}")


if __name__ == "__main__":
    main()
