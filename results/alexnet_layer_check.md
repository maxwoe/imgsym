# AlexNet layer check: conv1 (benchmark) vs conv2 (source recommendation)

Same benchmark pipeline; conv2 = torchvision features[0:5]; global
negatives coincidence-filtered.

| config | symComp17_s | NYU_s | PIX2PER-art | PIX2PER-nat | mean | local | global |
|---|---|---|---|---|---|---|---|
| conv1_p17_ours | +0.84 | +0.90 | +0.67 | +0.75 | +0.79 | +0.58 | +0.80 |
| conv2_p17 | +0.85 | +0.93 | +0.71 | +0.78 | +0.81 | +0.60 | +0.91 |
| conv2_p9 | +0.86 | +0.90 | +0.71 | +0.77 | +0.81 | +0.59 | +0.91 |
