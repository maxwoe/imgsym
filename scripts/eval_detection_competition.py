"""Competition-protocol reflection-detection evaluation (CVPR'13 / ICCV'17).

Reports the standard max F-measure (with precision/recall at that operating point)
per detector per single-axis dataset, using the shared axis-match rule: angle
< 10 deg AND distance from the detected axis center to the GT line SEGMENT
< 0.2 * min(l_det, l_GT), with a confidence-swept PR curve. See
imgsym.evaluation.detection_metrics.

    python scripts/eval_detection_competition.py [--detectors hog,nxc-ncc]
                                                 [--datasets all] [--limit N]
"""
import argparse
import time

import cv2
import numpy as np

from imgsym.detection.nxc import calc_symmetry_lines as _nxc
from imgsym.detection.xfeatures import calc_symmetry_lines as _xfeatures
from imgsym.detection.wavelets import calc_symmetry_lines as _wavelets
from imgsym.evaluation import (load_nyu_single, load_symcomp13_single,
                               load_symcomp17_single)
from imgsym.evaluation.detection_metrics import reflection_pr_f
from imgsym.evaluation.extraction import Axis

DATASETS = {
    "symComp17_s": lambda: load_symcomp17_single("data/datasets/symComp17/reflection_training/ref_s"),
    "NYU_s": lambda: load_nyu_single("data/datasets/sym_datasets/NYU/S"),
    "symComp13_s": lambda: load_symcomp13_single("data/datasets/symComp13"),
}


def _to_axes(angs, mids, segs, strengths):
    return [(Axis(cx=float(mids[i][0]), cy=float(mids[i][1]),
                  angle=float(angs[i]), along_extent=float(segs[i])),
             float(strengths[i])) for i in range(len(angs))]


def detect_nxc(img):
    # Faithful Cicconet (MSR) settings = the nxc defaults (BoxSize=50, NumBoxSamples=100,
    # AngleSet=0:6:354 / 60 angles), matching symmetryViaRegistration2D.m. No throttle.
    # ~5 s/img (the earlier nBoxSamples=30 / 15deg was a speed hack that under-ran nxc).
    return _to_axes(*_nxc(img))


def detect_xfeatures(img):
    # Per-image normalized strengths (the default) score BETTER than raw here: the
    # raw Hough-peak weight scales with feature count, biasing cross-image ranking
    # (raw verified worse by -0.04..-0.07 F). Opposite of hog/nxc's bounded scores.
    return _to_axes(*_xfeatures(img))


def detect_wavelets(img):
    # Normalized strengths (default) > raw for the same feature-count reason (-0.13..-0.19 F raw).
    return _to_axes(*_wavelets(img))


DETECTORS = {"nxc-ncc": detect_nxc,
             "xfeatures": detect_xfeatures, "wavelets": detect_wavelets}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", default="nxc-ncc,xfeatures")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--limit", type=int, default=100000)
    args = ap.parse_args()

    dnames = list(DATASETS) if args.datasets == "all" else args.datasets.split(",")
    detectors = [d.strip() for d in args.detectors.split(",")]
    print(f"{'detector':13}{'dataset':14}{'n':>5}{'F':>8}{'prec':>7}{'rec':>7}{'s/img':>8}",
          flush=True)
    for dname in detectors:
        fn = DETECTORS[dname]
        for ds in dnames:
            units = list(DATASETS[ds]().dominant_axes())[:args.limit]
            per_image, t0, n = [], time.time(), 0
            for u in units:
                img = cv2.imread(u.image_path)
                if img is None:
                    continue
                n += 1
                per_image.append((fn(img), [u.axis]))
            res = reflection_pr_f(per_image)
            dt = (time.time() - t0) / max(n, 1)
            print(f"{dname:13}{ds:14}{n:>5}{res['best_f']:>8.3f}"
                  f"{res['precision']:>7.2f}{res['recall']:>7.2f}{dt:>8.2f}", flush=True)


if __name__ == "__main__":
    main()
