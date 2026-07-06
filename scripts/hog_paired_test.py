"""Paired test: new HOG defaults (cell16/nbins18/signed=False) vs old (cell8/nbins9/signed=True).

Scores both configs on the same extractions across the four single-axis datasets
(standard perturbations) and reports the per-dataset + pooled paired-bootstrap skill
difference. CI excluding 0 => the new defaults are significantly better. Pooled result:
+0.058 [+0.040,+0.077] SIGNIFICANT (symComp13_s excluded, per the 2026-07-04 overlap fix).
"""
import cv2
import numpy as np

from imgsym.scoring.calculators import HOGCalculator
from imgsym.evaluation import (extract, perturb_axis, discrimination_skill,
                               paired_skill_diff, load_symcomp17_single,
                               load_nyu_single, load_pix2per)

old = HOGCalculator(cell_size=8, nbins=9, signed_gradient=True)   # previous default
new = HOGCalculator()                                            # new default (16, 18, False)

# symComp13_s excluded: 25/35 of its images duplicate symComp17_s (2026-07-04 overlap fix).
ROOT = "data/datasets"
cfgs = [
    ("symComp17_s", load_symcomp17_single(f"{ROOT}/symComp17/reflection_training/ref_s"), "min_edge"),
    ("NYU_s", load_nyu_single(f"{ROOT}/sym_datasets/NYU/S"), "min_edge"),
    ("PIX2PER-art", load_pix2per(f"{ROOT}/PIX2PER Dataset", subset="art"), "bbox"),
    ("PIX2PER-nat", load_pix2per(f"{ROOT}/PIX2PER Dataset", subset="nat"), "bbox"),
]

all_old, all_new = [], []
print(f"{'dataset':14}{'old':>8}{'new':>8}   diff [95% CI]")
for name, ds, policy in cfgs:
    op, npi = [], []
    for u in ds.dominant_axes():
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
        op.append((float(old.calculate_score(rt.subimage)),
                   [float(old.calculate_score(s)) for s in wsubs]))
        npi.append((float(new.calculate_score(rt.subimage)),
                    [float(new.calculate_score(s)) for s in wsubs]))
    os_, ns_ = discrimination_skill(op), discrimination_skill(npi)
    d, lo, hi = paired_skill_diff(npi, op)
    sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."
    print(f"{name:14}{os_:>+8.3f}{ns_:>+8.3f}   {d:+.3f} [{lo:+.3f},{hi:+.3f}] {sig}", flush=True)
    all_old += op
    all_new += npi

os_, ns_ = discrimination_skill(all_old), discrimination_skill(all_new)
d, lo, hi = paired_skill_diff(all_new, all_old)
print(f"{'POOLED':14}{os_:>+8.3f}{ns_:>+8.3f}   {d:+.3f} [{lo:+.3f},{hi:+.3f}] "
      f"{'SIGNIFICANT' if (lo > 0 or hi < 0) else 'n.s.'}")
