# PatchNN (local_global) parameter check: ours vs Hogeweg's

All at stride k=4 (their kappa=16 condition); global negatives use the
coincidence-filtered generator, comparable to the corrected analysis.

| config | symComp17_s | NYU_s | PIX2PER-art | PIX2PER-nat | mean | local | global |
|---|---|---|---|---|---|---|---|
| m31_w31_ours | +0.71 | +0.79 | +0.51 | +0.56 | +0.64 | +0.44 | +0.24 |
| m15_w17.7_optimum | +0.70 | +0.78 | +0.51 | +0.59 | +0.65 | +0.44 | +0.29 |
| m9_w10_default | +0.65 | +0.76 | +0.50 | +0.58 | +0.62 | +0.42 | +0.28 |
