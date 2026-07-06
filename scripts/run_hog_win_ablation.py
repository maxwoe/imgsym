"""HOG window-size x cell-size ablation for symmetry discrimination.

Companion to run_hog_ablation.py (which sweeps cell/nbins/signed at a FIXED
win=64): sweeps the resize-target window size jointly with the cell size to ask
whether the mid-scale peak is about the ABSOLUTE cell size or the cell:window
RATIO. Same cache, subset (symComp17 + PIX2PER-art), and protocol; unsigned
gradients, 18 bins throughout. (win=128, cell=4) is skipped -- at native
resolution its dense sliding-window descriptor is prohibitively large.

    python scripts/run_hog_win_ablation.py [--limit 50]
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_hog_ablation import build_cache, skill_for_cfg  # noqa: E402

from imgsym.evaluation import discrimination_skill, discrimination_skill_ci  # noqa: E402

WINS = (32, 64, 128)
CELLS = (4, 8, 16, 32, 64)
NBINS = 18
SIGNED = False
SKIP = {(128, 4)}  # (win, cell): dense-window descriptor too large at native res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50, help="images per dataset")
    args = ap.parse_args()

    print("building subimage cache...", flush=True)
    cache = build_cache(args.limit)
    print(f"cache: {len(cache)} images\n", flush=True)

    # cfg tuple order matches run_hog_ablation.hog_desc: (cell, nbins, win, signed)
    cfgs = [(c, NBINS, w, SIGNED) for w in WINS for c in CELLS
            if 2 * c <= w and (w, c) not in SKIP]

    rows = []
    t0 = time.time()
    for i, cfg in enumerate(cfgs):
        pi = skill_for_cfg(cache, cfg)
        sk = discrimination_skill(pi)
        _, lo, hi = discrimination_skill_ci(pi)
        rows.append((cfg, sk, lo, hi))
        print(f"  [{i + 1}/{len(cfgs)}] win={cfg[2]:>3} cell={cfg[0]:>2} "
              f"(ratio {cfg[2] // cfg[0]:>2})  skill={sk:+.3f} [{lo:+.3f},{hi:+.3f}]  "
              f"({time.time() - t0:.0f}s)", flush=True)

    os.makedirs("results", exist_ok=True)
    with open("results/hog_ablation_win.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["win", "cell", "nbins", "signed", "skill", "lo", "hi", "n_images"])
        for (cell, nbins, win, signed), sk, lo, hi in rows:
            w.writerow([win, cell, nbins, signed,
                        f"{sk:.4f}", f"{lo:.4f}", f"{hi:.4f}", len(cache)])
    print("wrote results/hog_ablation_win.csv", flush=True)

    print(f"\n{'win':>5}{'cell':>6}{'ratio':>7}{'skill':>9}")
    for (cell, nbins, win, signed), sk, lo, hi in sorted(rows, key=lambda r: -r[1]):
        print(f"{win:>5}{cell:>6}{win // cell:>7}{sk:>+9.3f}", flush=True)
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
