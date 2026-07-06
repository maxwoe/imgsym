# Fixed-extent control (crop-size confound)

Same benchmark, extents fixed to 0.25/0.35 of the short side -- true and
wrong crops of an image are IDENTICAL in size. fixed = this control;
bench = the published min_edge/bbox skill.

| method | symComp17_s fixed (bench) | NYU_s fixed (bench) | PIX2PER-art fixed (bench) | PIX2PER-nat fixed (bench) | mean fixed (bench) |
|---|---|---|---|---|---|
| deep_features | +0.87 (+0.87) | +0.91 (+0.94) | +0.68 (+0.74) | +0.72 (+0.77) | +0.79 (+0.83) |
| hog | +0.87 (+0.86) | +0.90 (+0.92) | +0.65 (+0.73) | +0.67 (+0.69) | +0.77 (+0.80) |
| alexnet | +0.88 (+0.86) | +0.93 (+0.91) | +0.66 (+0.71) | +0.72 (+0.77) | +0.80 (+0.81) |
| gabor | +0.83 (+0.76) | +0.86 (+0.83) | +0.56 (+0.61) | +0.61 (+0.64) | +0.72 (+0.71) |
| sliding_window | +0.73 (+0.71) | +0.82 (+0.77) | +0.47 (+0.55) | +0.47 (+0.54) | +0.62 (+0.65) |
| pixel_correlation | +0.68 (+0.69) | +0.80 (+0.77) | +0.47 (+0.55) | +0.45 (+0.54) | +0.60 (+0.64) |
