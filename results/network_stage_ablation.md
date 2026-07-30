# Network x stage ablation -- discrimination skill (n=80)

| backbone | stage skills (early -> late) | best | best skill [95% CI] |
|---|---|---|---|
| convnextv2_base.fcmae_ft_in22k_in1k | +0.90 +0.94 +0.79 +0.84 | stage1 | +0.94 [+0.90,+0.97] |
| mambaout_base.in1k | +0.90 +0.94 +0.88 +0.83 | stage1 | +0.94 [+0.90,+0.97] |
| convnextv2_atto.fcmae_ft_in1k | +0.90 +0.93 +0.89 +0.78 | stage1 | +0.93 [+0.90,+0.96] |
| efficientnetv2_rw_m.agc_in1k | +0.87 +0.91 +0.92 +0.93 +0.89 | stage3 | +0.93 [+0.90,+0.96] |
| swin_tiny_patch4_window7_224 | +0.85 +0.93 +0.64 +0.80 | stage1 | +0.93 [+0.90,+0.96] |
| focalnet_base_lrf.ms_in1k | +0.90 +0.93 +0.86 +0.74 | stage1 | +0.93 [+0.89,+0.97] |
| efficientvit_b0.r224_in1k | +0.90 +0.93 +0.93 +0.90 | stage1 | +0.93 [+0.89,+0.96] |
| efficientnetv2_rw_t.ra2_in1k | +0.81 +0.93 +0.93 +0.93 +0.84 | stage3 | +0.93 [+0.90,+0.96] |
| mambaout_tiny.in1k | +0.92 +0.93 +0.93 +0.86 | stage1 | +0.93 [+0.89,+0.97] |
| efficientvit_b3.r224_in1k | +0.91 +0.93 +0.93 +0.89 | stage1 | +0.93 [+0.89,+0.96] |
| swin_base_patch4_window7_224 | +0.82 +0.93 +0.75 +0.69 | stage1 | +0.93 [+0.90,+0.96] |
| focalnet_tiny_lrf.ms_in1k | +0.90 +0.93 +0.92 +0.84 | stage1 | +0.93 [+0.88,+0.96] |
| edgenext_base.in21k_ft_in1k | +0.91 +0.93 +0.91 +0.91 | stage1 | +0.93 [+0.89,+0.96] |
| edgenext_xx_small.in1k | +0.89 +0.92 +0.89 +0.88 | stage1 | +0.92 [+0.88,+0.95] |
| vit_base_patch16_224.augreg_in21k_ft_in1k | +0.92 +0.89 +0.89 | stage0 | +0.92 [+0.88,+0.95] |
| vit_base_patch8_224.augreg_in21k_ft_in1k | +0.92 +0.91 +0.92 | stage0 | +0.92 [+0.89,+0.95] |
| vit_tiny_patch16_224.augreg_in21k_ft_in1k | +0.86 +0.86 +0.85 | stage0 | +0.86 [+0.79,+0.93] |

vit_base_patch8 row: finer-patch stride control (reviewer-requested), run 2026-07-29 on an
NVIDIA A100 (GPU) in a clean env (torch 2.8.0+cu128, timm 1.0.19); pipeline validated by
reproducing the vit_base_patch16 CPU row to within +-0.01 per stage (see
network_stage_ablation_easy_sanity_b16.md, network_stage_ablation_easy_patch8.md).
