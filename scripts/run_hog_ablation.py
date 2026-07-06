"""HOG parameter ablation for symmetry discrimination.

The hand-crafted analog of the network x stage ablation: HOG's cell size (spatial
resolution) and nbins (orientation resolution) play the role of feature "scale".
Sweeps those (+ window size, signed gradient) and reports discrimination skill, on
an easy+hard subset (symComp17 + PIX2PER-art) so configs separate. Standalone --
scores with a local hog_desc equivalent to HOGCalculator's; this sweep is what
JUSTIFIED the tuned library defaults (cell16/nbins18/signed=False), marked below.

    python scripts/run_hog_ablation.py [--limit 50]
"""
import argparse
import itertools
import time

import cv2
import numpy as np

from imgsym.evaluation import (extract, perturb_axis, discrimination_skill,
                               discrimination_skill_ci, load_symcomp17_single,
                               load_pix2per)

CELLS = (4, 8, 16, 32)  # spatial-resolution sweep (the mid-scale peak is at 16)
NBINS = (9, 18)         # orientation resolution
WINS = (64,)            # resize target
SIGNED = (False, True)  # unsigned vs signed gradient (mirror preserves unsigned)
DEFAULT = (16, 18, 64, False)   # tuned library default (pre-tuning default was 8/9/64/True)


def hog_desc(img, cell, nbins, win, signed):
    h, w = img.shape[:2]
    if w < win or h < win:
        s = max(win / w, win / h)
        img = cv2.resize(img, (max(win, int(round(w * s))), max(win, int(round(h * s)))),
                         interpolation=cv2.INTER_LINEAR)
    hog = cv2.HOGDescriptor(_winSize=(win, win), _blockSize=(2 * cell, 2 * cell),
                            _blockStride=(cell, cell), _cellSize=(cell, cell),
                            _nbins=nbins, _signedGradient=signed, _gammaCorrection=True)
    return hog.compute(img)


def hog_score(img, cfg):
    d1 = hog_desc(img, *cfg)
    d2 = hog_desc(np.fliplr(img), *cfg)
    n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    return float(np.dot(d1, d2) / (n1 * n2))


def build_cache(limit):
    cache = []
    cfgs = [
        (load_symcomp17_single("data/datasets/symComp17/reflection_training/ref_s"), "min_edge"),
        (load_pix2per("data/datasets/PIX2PER Dataset", subset="art"), "bbox"),
    ]
    for ds, policy in cfgs:
        for u in list(ds.dominant_axes())[:limit]:
            img = cv2.imread(u.image_path)
            if img is None:
                continue
            rt = extract(img, u.axis, policy=policy, min_support_frac=0.10)
            if rt.info.degenerate:
                continue
            wsubs = [r.subimage for r in
                     (extract(img, wa, policy=policy, min_support_frac=0.10)
                      for wa in perturb_axis(u.axis, img.shape))
                     if not r.info.degenerate]
            if len(wsubs) < 6:
                continue
            cache.append((rt.subimage, wsubs))
    return cache


def skill_for_cfg(cache, cfg):
    per_image = []
    for true_sub, wsubs in cache:
        t = hog_score(true_sub, cfg)
        w = [hog_score(s, cfg) for s in wsubs]
        per_image.append((t, w))
    return per_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50, help="images per dataset")
    args = ap.parse_args()

    print("building subimage cache...", flush=True)
    cache = build_cache(args.limit)
    print(f"cache: {len(cache)} images\n", flush=True)

    rows = []
    t0 = time.time()
    configs = list(itertools.product(CELLS, NBINS, WINS, SIGNED))
    for i, cfg in enumerate(configs):
        pi = skill_for_cfg(cache, cfg)
        sk = discrimination_skill(pi)
        rows.append((cfg, sk, pi))
        print(f"  [{i+1}/{len(configs)}] cell={cfg[0]} nbins={cfg[1]} signed={cfg[3]}  "
              f"skill={sk:+.3f}  ({time.time()-t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: -r[1])

    import csv
    import os
    os.makedirs("results", exist_ok=True)
    with open("results/hog_ablation.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "nbins", "win", "signed", "skill", "lo", "hi", "feat_dim", "n_images"])
        for cfg, sk, pi in rows:
            _, lo, hi = discrimination_skill_ci(pi)
            w.writerow([cfg[0], cfg[1], cfg[2], cfg[3], f"{sk:.4f}", f"{lo:.4f}", f"{hi:.4f}",
                        len(hog_desc(cache[0][0], *cfg)), len(cache)])
    print("wrote results/hog_ablation.csv", flush=True)

    print(f"{'cell':>5}{'nbins':>6}{'win':>5}{'signed':>8}{'skill':>9}   {'feat dim':>9}")
    for cfg, sk, _ in rows:
        cell, nbins, win, signed = cfg
        dim = len(hog_desc(cache[0][0], *cfg))
        tag = "  <- DEFAULT" if cfg == DEFAULT else ""
        print(f"{cell:>5}{nbins:>6}{win:>5}{str(signed):>8}{sk:>+9.3f}   {dim:>9}{tag}", flush=True)

    best_cfg, best_sk, best_pi = rows[0]
    _, lo, hi = discrimination_skill_ci(best_pi)
    def_pi = next(pi for cfg, _, pi in rows if cfg == DEFAULT)
    def_sk = discrimination_skill(def_pi)
    print(f"\nbest={best_cfg} skill={best_sk:+.3f} [{lo:+.3f},{hi:+.3f}]  "
          f"vs default skill={def_sk:+.3f}  (delta {best_sk-def_sk:+.3f})")
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
