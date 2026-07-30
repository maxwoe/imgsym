# Network x stage ablation (hard) -- discrimination skill (n=40)

| backbone | stage skills (early -> late) | best | best skill [95% CI] |
|---|---|---|---|
| efficientnetv2_rw_t.ra2_in1k | +0.41 +0.56 +0.65 +0.73 +0.65 | stage3 | +0.73 [+0.62,+0.82] |
| convnextv2_atto.fcmae_ft_in1k | +0.60 +0.72 +0.64 +0.64 | stage1 | +0.72 [+0.62,+0.81] |
| efficientvit_b3.r224_in1k | +0.58 +0.63 +0.68 +0.72 | stage3 | +0.72 [+0.63,+0.81] |
| mambaout_tiny.in1k | +0.72 +0.68 +0.60 +0.59 | stage0 | +0.72 [+0.60,+0.82] |
| efficientvit_b0.r224_in1k | +0.48 +0.60 +0.69 +0.71 | stage3 | +0.71 [+0.60,+0.81] |
| focalnet_tiny_lrf.ms_in1k | +0.62 +0.69 +0.70 +0.57 | stage2 | +0.70 [+0.56,+0.82] |
| mambaout_base.in1k | +0.66 +0.68 +0.57 +0.61 | stage1 | +0.68 [+0.53,+0.80] |
| efficientnetv2_rw_m.agc_in1k | +0.48 +0.55 +0.65 +0.68 +0.68 | stage3 | +0.68 [+0.55,+0.80] |
| focalnet_base_lrf.ms_in1k | +0.66 +0.67 +0.50 +0.51 | stage1 | +0.67 [+0.55,+0.78] |
| swin_base_patch4_window7_224 | +0.61 +0.67 +0.41 +0.45 | stage1 | +0.67 [+0.54,+0.79] |
| convnextv2_base.fcmae_ft_in22k_in1k | +0.60 +0.67 +0.45 +0.60 | stage1 | +0.67 [+0.51,+0.80] |
| edgenext_xx_small.in1k | +0.60 +0.67 +0.66 +0.56 | stage1 | +0.67 [+0.53,+0.78] |
| swin_tiny_patch4_window7_224 | +0.60 +0.67 +0.46 +0.59 | stage1 | +0.67 [+0.54,+0.79] |
| vit_base_patch8_224.augreg_in21k_ft_in1k | +0.67 +0.66 +0.65 | stage0 | +0.67 [+0.55,+0.79] |
| edgenext_base.in21k_ft_in1k | +0.62 +0.59 +0.63 +0.64 | stage3 | +0.64 [+0.51,+0.76] |
| vit_base_patch16_224.augreg_in21k_ft_in1k | +0.61 +0.59 +0.53 | stage0 | +0.61 [+0.46,+0.74] |
| vit_tiny_patch16_224.augreg_in21k_ft_in1k | +0.39 +0.39 +0.39 | stage0 | +0.39 [+0.20,+0.57] |

vit_base_patch8 row: finer-patch stride control (reviewer-requested), run 2026-07-29 on an
NVIDIA A100 (GPU) in a clean env (torch 2.8.0+cu128, timm 1.0.19); pipeline validated by
reproducing the vit_base_patch16 CPU easy row to within +-0.01 per stage (see
network_stage_ablation_hard_patch8.md).
