"""Network x stage ablation for deep symmetry features.

Tests the 16 backbones from the image-rotation-angle-estimation paper (same set,
for cross-task comparability) as frozen feature extractors. For each backbone and
each of its feature stages, computes discrimination skill (true vs perturbed axes)
on a clean single-axis subset (NYU + symComp17). Asks: do hierarchical backbones
peak at an early/mid stage (like MambaOut stage 1), do the columnar ViTs behave
differently, and does symmetry agree with rotation on which backbone wins?

Writes results/network_stage_ablation.{json,md}. One backbone in memory at a time.

    python scripts/run_network_ablation.py [--limit 40]
"""
import argparse
import gc
import json
import os
import time

import cv2
import numpy as np
import torch
import timm
import timm.data

from imgsym.evaluation import (extract, perturb_axis, discrimination_skill,
                               discrimination_skill_ci, load_symcomp17_single,
                               load_nyu_single, load_pix2per)

# Exact pretrained tags from the rotation paper's architectures.py.
# vit_base_patch8 added 2026-07 (reviewer-requested finer-patch control: same AugReg
# weights family as the two patch-16 ViTs, isolating tokenization stride from hierarchy).
BACKBONES = [
    "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "vit_base_patch8_224.augreg_in21k_ft_in1k",
    "efficientvit_b0.r224_in1k",
    "efficientvit_b3.r224_in1k",
    "convnextv2_atto.fcmae_ft_in1k",
    "convnextv2_base.fcmae_ft_in22k_in1k",
    "efficientnetv2_rw_t.ra2_in1k",
    "efficientnetv2_rw_m.agc_in1k",
    "mambaout_tiny.in1k",
    "mambaout_base.in1k",
    "focalnet_tiny_lrf.ms_in1k",
    "focalnet_base_lrf.ms_in1k",
    "edgenext_xx_small.in1k",
    "edgenext_base.in21k_ft_in1k",
    "swin_tiny_patch4_window7_224",
    "swin_base_patch4_window7_224",
]


def build_subimage_cache(limit_per_ds, subset="easy"):
    """List of (true_sub, [wrong_subs]). subset='easy' = NYU+symComp17 (clean,
    for stage profiling); 'hard' = PIX2PER-art (to actually rank backbones, where
    the easy subset ceilings). Extraction is done once and reused per backbone."""
    cache = []
    if subset == "hard":
        cfgs = [(load_pix2per("data/datasets/PIX2PER Dataset", subset="art"), "bbox")]
    else:
        cfgs = [
            (load_symcomp17_single("data/datasets/symComp17/reflection_training/ref_s"), "min_edge"),
            (load_nyu_single("data/datasets/sym_datasets/NYU/S"), "min_edge"),
        ]
    for ds, policy in cfgs:
        for u in list(ds.dominant_axes())[:limit_per_ds]:
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


def make_preproc(name):
    tmp = timm.create_model(name, pretrained=False)
    cfg = timm.data.resolve_data_config({}, model=tmp)
    del tmp
    return (cfg["input_size"][1],
            np.array(cfg["mean"], np.float32),
            np.array(cfg["std"], np.float32))


def forward_stages(model, bgr, size, mean, std, device="cpu"):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (size, size)).astype(np.float32) / 255.0
    r = (r - mean) / std
    t = torch.from_numpy(r.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        fs = model(t)
    return [f.squeeze(0).abs().cpu().numpy() for f in fs]


def cmp_stage(a, b):
    return float(1.0 - np.sum(np.abs(a - b)) / (np.sum(np.maximum(a, b)) + 1e-8))


def score_backbone(name, cache, device="cpu"):
    model = timm.create_model(name, pretrained=True, features_only=True).eval().to(device)
    size, mean, std = make_preproc(name)
    per_stage = None
    for true_sub, wsubs in cache:
        tf = forward_stages(model, true_sub, size, mean, std, device)
        tff = forward_stages(model, np.fliplr(true_sub), size, mean, std, device)
        t_scores = [cmp_stage(a, b) for a, b in zip(tf, tff)]
        if per_stage is None:
            per_stage = [[] for _ in range(len(t_scores))]
        w_by_stage = [[] for _ in range(len(t_scores))]
        for ws in wsubs:
            wf = forward_stages(model, ws, size, mean, std, device)
            wff = forward_stages(model, np.fliplr(ws), size, mean, std, device)
            for si, (a, b) in enumerate(zip(wf, wff)):
                w_by_stage[si].append(cmp_stage(a, b))
        for si in range(len(t_scores)):
            per_stage[si].append((t_scores[si], w_by_stage[si]))
    del model
    gc.collect()
    skills = [discrimination_skill(ps) for ps in per_stage]
    best = int(np.argmax(skills))
    _, lo, hi = discrimination_skill_ci(per_stage[best])
    return {"n_stages": len(skills), "stage_skills": skills,
            "best_stage": best, "best_skill": skills[best], "best_ci": [lo, hi]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40, help="images per dataset")
    ap.add_argument("--subset", choices=["easy", "hard"], default="easy",
                    help="easy=NYU+symComp17 (stage profiling); hard=PIX2PER-art (ranking)")
    ap.add_argument("--device", default="cpu", help="torch device (cpu or cuda)")
    ap.add_argument("--only", default="",
                    help="comma-separated backbone tags: run only these (default all)")
    ap.add_argument("--out-suffix", default="",
                    help="append to output basename (avoids clobbering full-pool results)")
    args = ap.parse_args()
    out_base = f"results/network_stage_ablation_{args.subset}{args.out_suffix}"

    print(f"building subimage cache ({args.subset})...", flush=True)
    cache = build_subimage_cache(args.limit, args.subset)
    print(f"cache: {len(cache)} images\n", flush=True)

    pool = [b for b in args.only.split(",") if b] or BACKBONES
    results = {}
    for name in pool:
        t0 = time.time()
        try:
            results[name] = score_backbone(name, cache, args.device)
            v = results[name]
            print(f"[{name}] stages={v['n_stages']} "
                  f"skills={[round(s, 2) for s in v['stage_skills']]} "
                  f"best=stage{v['best_stage']} ({v['best_skill']:+.2f}) "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            results[name] = {"error": str(e)[:200]}
            print(f"[{name}] FAILED: {str(e)[:120]} ({time.time()-t0:.0f}s)", flush=True)
            gc.collect()

    os.makedirs("results", exist_ok=True)
    with open(out_base + ".json", "w") as fh:
        json.dump({"backbones": pool, "n_images": len(cache), "subset": args.subset,
                   "results": results}, fh, indent=2)

    ok = {k: v for k, v in results.items() if "stage_skills" in v}
    order = sorted(ok, key=lambda k: -ok[k]["best_skill"])
    with open(out_base + ".md", "w") as fh:
        fh.write(f"# Network x stage ablation ({args.subset}) -- discrimination skill "
                 f"(n={len(cache)})\n\n")
        fh.write("| backbone | stage skills (early -> late) | best | best skill [95% CI] |\n")
        fh.write("|---|---|---|---|\n")
        for k in order:
            v = ok[k]
            ss = " ".join(f"{s:+.2f}" for s in v["stage_skills"])
            fh.write(f"| {k} | {ss} | stage{v['best_stage']} | "
                     f"{v['best_skill']:+.2f} [{v['best_ci'][0]:+.2f},{v['best_ci'][1]:+.2f}] |\n")
        failed = [k for k in results if "error" in results[k]]
        if failed:
            fh.write(f"\n**Failed `features_only`:** {', '.join(failed)}\n")
    print(f"\nwrote {out_base}.json and {out_base}.md", flush=True)


if __name__ == "__main__":
    main()
