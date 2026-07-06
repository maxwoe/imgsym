"""Reviewer-response statistics on the EXISTING per-image scores (no re-scoring).

  1. Paired ladder with two-sided bootstrap p-values and Holm correction
     (the plain 95% CIs of analyze_discrimination.py do not control the
     family-wise error over the six ladder comparisons).
  2. TOST equivalence tests at a +-0.03 margin (the paper's own "small but
     significant" scale): alexnet vs hog (single, pooled) and deep vs hog on
     DENDI (multi) -- a tie claim needs equivalence, not non-significance.
  3. Cluster bootstrap for the multi protocol: resample IMAGES (with all
     their axes) instead of axes, removing the same-image pseudo-replication;
     reports per-dataset CI widening for the top methods.
  4. Held-out skills for the two benchmark-tuned methods, restricted to the
     datasets outside each method's tuning pool (from discrimination.csv).

    python scripts/analyze_claims_stats.py > results/claims_stats.txt
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_discrimination import _configs, DATA_ROOT

from imgsym.evaluation import extract
from imgsym.evaluation.scoring_metrics import _skill_fixed, infer_direction

N_BOOT = 2000
SEED = 42
MARGIN = 0.03          # TOST equivalence margin = the paper's "small" scale


def std_pairs(per_image, std_tags):
    out = []
    for rec in per_image:
        wr = [s for tag, s in rec["wrong"] if tag in std_tags and s is not None]
        if wr:
            out.append((rec["true"], wr))
    return out


def pooled(data, method, std_tags, datasets=None):
    out = []
    for ds in (datasets or data):
        out += std_pairs(data[ds][method], std_tags)
    return out


def paired_boot(pa, pb, n_boot=N_BOOT, seed=SEED):
    """(point, lo95, hi95, lo90, hi90, p_two_sided) for skill(A)-skill(B)."""
    hib_a = infer_direction([t for t, _ in pa], [x for _, w in pa for x in w])
    hib_b = infer_direction([t for t, _ in pb], [x for _, w in pb for x in w])
    point = _skill_fixed(pa, hib_a) - _skill_fixed(pb, hib_b)
    rng = np.random.RandomState(seed)
    n = len(pa)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        boot[b] = (_skill_fixed([pa[i] for i in idx], hib_a)
                   - _skill_fixed([pb[i] for i in idx], hib_b))
    # add-one smoothed two-sided bootstrap p-value
    p = 2.0 * min((np.sum(boot <= 0) + 1) / (n_boot + 1),
                  (np.sum(boot >= 0) + 1) / (n_boot + 1))
    return (point, np.percentile(boot, 2.5), np.percentile(boot, 97.5),
            np.percentile(boot, 5.0), np.percentile(boot, 95.0), min(p, 1.0))


def holm(pairs_p):
    """Holm-Bonferroni adjusted p-values (input: list of (label, p))."""
    m = len(pairs_p)
    order = sorted(range(m), key=lambda i: pairs_p[i][1])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pairs_p[i][1])
        adj[i] = min(1.0, running)
    return adj


def main():
    with open("results/per_image_scores.json") as fh:
        blob_s = json.load(fh)
    with open("results/per_image_scores_multi.json") as fh:
        blob_m = json.load(fh)
    std_s = set(blob_s["meta"]["std_tags"])
    std_m = set(blob_m["meta"]["std_tags"])
    ds_single, ds_multi = blob_s["data"], blob_m["data"]

    # ---- 1. Ladder with Holm correction (single, pooled) --------------------
    ladder = [("deep_features", "alexnet"), ("deep_features", "hog"),
              ("alexnet", "hog"), ("hog", "gabor"),
              ("gabor", "gradient"), ("gradient", "local_global")]
    print("=== 1. PAIRED LADDER: bootstrap p-values, Holm-corrected (m=6) ===\n")
    rows = []
    for a, b in ladder:
        pa, pb = pooled(ds_single, a, std_s), pooled(ds_single, b, std_s)
        rows.append((f"{a} - {b}", paired_boot(pa, pb)))
    adj = holm([(lbl, r[5]) for lbl, r in rows])
    for (lbl, (pt, lo, hi, _, _, p)), padj in zip(rows, adj):
        sig = "SIG" if padj < 0.05 else "n.s."
        print(f"  {lbl:34} = {pt:+.3f} [95% {lo:+.3f},{hi:+.3f}]  "
              f"p={p:.4f}  Holm p={padj:.4f}  -> {sig}")

    # ---- 2. TOST equivalence at +-MARGIN ------------------------------------
    print(f"\n=== 2. TOST EQUIVALENCE (margin +-{MARGIN}; 90% CI inside margin"
          " => equivalent at alpha=0.05) ===\n")
    for name, data, std, dss, a, b in [
            ("alexnet vs hog (single, pooled)", ds_single, std_s, None,
             "alexnet", "hog"),
            ("deep vs hog (DENDI only)", ds_multi, std_m, ["DENDI"],
             "deep_features", "hog")]:
        pa, pb = pooled(data, a, std, dss), pooled(data, b, std, dss)
        pt, lo, hi, lo90, hi90, p = paired_boot(pa, pb)
        eq = "EQUIVALENT" if (lo90 > -MARGIN and hi90 < MARGIN) else "NOT SHOWN"
        print(f"  {name:34} diff={pt:+.3f}  90% CI [{lo90:+.3f},{hi90:+.3f}]"
              f"  -> {eq}")

    # ---- 3. Cluster bootstrap for the multi protocol ------------------------
    # Rebuild each dataset's unit->image grouping by replaying loader order and
    # the extraction gate (no scoring); stored records are in this exact order.
    print("\n=== 3. MULTI PROTOCOL: axis-level vs image-cluster bootstrap CIs ===\n")
    top = ["deep_features", "hog", "alexnet"]
    rng = np.random.RandomState(SEED)
    for name, loader, policy in _configs(DATA_ROOT, "multi", 10):
        if name not in ds_multi:
            continue
        groups = []
        for u in loader().all_axes():
            img = cv2.imread(u.image_path)
            if img is None:
                continue
            rt = extract(img, u.axis, policy=policy, min_support_frac=0.10)
            if rt.info.degenerate:
                continue
            groups.append(u.image_path)
        n_rec = len(ds_multi[name][top[0]])
        if len(groups) != n_rec:
            print(f"  [{name}] ALIGNMENT MISMATCH replay={len(groups)} "
                  f"stored={n_rec} -- skipping cluster CI")
            continue
        uniq = {}
        for i, g in enumerate(groups):
            uniq.setdefault(g, []).append(i)
        clusters = list(uniq.values())
        line = f"  [{name}] images={len(clusters)} axes={n_rec}:"
        for m in top:
            prs = std_pairs(ds_multi[name][m], std_m)
            idx_prs = [i for i, rec in enumerate(ds_multi[name][m])
                       if any(tag in std_m and s is not None
                              for tag, s in rec["wrong"])]
            pos = {orig: k for k, orig in enumerate(idx_prs)}
            hib = infer_direction([t for t, _ in prs],
                                  [x for _, w in prs for x in w])
            boot_ax, boot_cl = np.empty(N_BOOT), np.empty(N_BOOT)
            n = len(prs)
            for b in range(N_BOOT):
                ai = rng.randint(0, n, n)
                boot_ax[b] = _skill_fixed([prs[i] for i in ai], hib)
                ci = rng.randint(0, len(clusters), len(clusters))
                sel = [pos[j] for c in ci for j in clusters[c] if j in pos]
                boot_cl[b] = _skill_fixed([prs[k] for k in sel], hib)
            wa = np.percentile(boot_ax, 97.5) - np.percentile(boot_ax, 2.5)
            wc = np.percentile(boot_cl, 97.5) - np.percentile(boot_cl, 2.5)
            line += (f"  {m.split('_')[0]}: axisCI={wa:.3f}"
                     f" clusterCI={wc:.3f} (x{wc / wa:.2f})")
        print(line)

    # ---- 4. Held-out skills for the two tuned methods -----------------------
    print("\n=== 4. TUNED METHODS ON DATASETS OUTSIDE THEIR TUNING POOLS ===\n")
    with open("results/discrimination.csv") as fh:
        rd = list(csv.DictReader(fh))
    tbl = {r["method"]: r for r in rd}

    def mean_over(m, sets):
        return float(np.mean([float(tbl[m][f"{d}_skill"]) for d in sets]))

    all_s = ["symComp17_s", "NYU_s", "PIX2PER-art", "PIX2PER-nat"]
    print("  HOG   tuning pool = symComp17 + PIX2PER-art")
    print(f"        held-out (NYU_s, PIX2PER-nat) mean = "
          f"{mean_over('hog', ['NYU_s', 'PIX2PER-nat']):+.3f}   "
          f"(pooled 4-set mean = {mean_over('hog', all_s):+.3f})")
    print("  DEEP  stage/backbone tuning subsets = NYU + symComp17 (easy), "
          "PIX2PER-art (hard)")
    print(f"        held-out (PIX2PER-nat) skill = "
          f"{mean_over('deep_features', ['PIX2PER-nat']):+.3f}   "
          f"(pooled 4-set mean = {mean_over('deep_features', all_s):+.3f})")
    print("  reference on the same held-out sets:")
    print(f"        alexnet on NYU_s+PIX2PER-nat = "
          f"{mean_over('alexnet', ['NYU_s', 'PIX2PER-nat']):+.3f}; "
          f"on PIX2PER-nat = {mean_over('alexnet', ['PIX2PER-nat']):+.3f}; "
          f"hog on PIX2PER-nat = {mean_over('hog', ['PIX2PER-nat']):+.3f}")


if __name__ == "__main__":
    main()
