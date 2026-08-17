# imgsym

[![Paper](https://img.shields.io/badge/Paper-Symmetry%202026-004B87?logo=doi&logoColor=white)](https://doi.org/10.3390/sym18081355)
[![PyPI](https://img.shields.io/pypi/v/imgsym?logo=pypi&logoColor=white)](https://pypi.org/project/imgsym/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://github.com/maxwoe/imgsym/blob/main/LICENSE)

Detect symmetry axes in images and score mirror symmetry along them.
**imgsym** implements multiple algorithms from symmetry research — 4
**detectors** and 13 **scorers** — behind one small API, and benchmarks them
head-to-head.

## Installation

```bash
pip install imgsym
```

This covers the classical scoring methods and the `nxc` / `xfeatures` / `r_lip`
detectors. Some methods pull optional extras:

```bash
pip install "imgsym[deep]"      # alexnet, deep_features
pip install "imgsym[wavelets]"  # wavelets detector
pip install "imgsym[all]"       # everything (+ numba speedups)
```

## Quick start

```python
import cv2
import imgsym

# find symmetry axes: (angles, midpoints, lengths, strengths), strongest first
img = cv2.imread("assets/butterfly.jpg")
bf_axes = imgsym.detect_axes(img, detector="nxc")
vis1 = imgsym.visualize(img, bf_axes)

# score mirror symmetry along an axis; higher = more symmetric
axis = imgsym.get_axis(bf_axes, 0)                # strongest detected axis
imgsym.score_along_axis(img, axis, method="hog")  # 0.89

# binary shape masks have their own detector
mask = cv2.imread("assets/synthetic_star.png", cv2.IMREAD_GRAYSCALE)
star_axes = imgsym.detect_axes_from_mask(mask, detector="r_lip", mode="multi", threshold=0.9)
vis2 = imgsym.visualize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), star_axes)
```

<table>
  <tr>
    <td align="center"><img src="https://raw.githubusercontent.com/maxwoe/imgsym/main/assets/butterfly_axes_nxc.png" alt="Detected symmetry axis on a natural image" width="430"/></td>
    <td align="center"><img src="https://raw.githubusercontent.com/maxwoe/imgsym/main/assets/synthetic_star_axes_lip.png" alt="r_lip: five mirror axes of a star mask" width="280"/></td>
  </tr>
  <tr>
    <td align="center">natural image — <code>nxc</code></td>
    <td align="center">binary shape mask — <code>r_lip</code></td>
  </tr>
</table>

[`demo.ipynb`](https://github.com/maxwoe/imgsym/blob/main/demo.ipynb) reproduces every figure on this page and has more to
play with.

## Choosing a method

### Detectors

**Natural images** — `imgsym.detect_axes(image, detector=...)`

Detection accuracy is competition-protocol max-F (a detected axis counts if
its angle is within 10° of ground truth and its center lies on the true
segment), measured on three annotated benchmarks with
`scripts/eval_detection_competition.py`:

| detector | symComp17 | NYU | symComp13 | speed (s/img) | source |
|---|---|---|---|---|---|
| `nxc` | **0.67** | **0.81** | **0.80** | ~7 | Cicconet et al. (2017) |
| `xfeatures` | 0.59 | 0.74 | 0.57 | 0.05–0.2 | Loy & Eklundh (2006) |
| `wavelets` | 0.46 | 0.53 | 0.49 | ~0.15 | Elawady et al. (2017) |

`nxc` leads on every benchmark — use it when accuracy matters; `xfeatures` is
~25× faster and the solid middle ground; `wavelets` trails clearly.

**Binary shapes** — `imgsym.detect_axes_from_mask(mask, detector=...)`

A different input domain, so not comparable above:

| detector | speed (s/img) | source |
|---|---|---|
| `r_lip` | <0.1 | Nguyen et al. (2022) |

### Scorers

Numbers, not guesswork: **skill** is single-axis discrimination skill
(2·AUC−1; 1 = perfect, 0 = chance) averaged over the four datasets of our
benchmark (see [Benchmark](#benchmark)); **speed** is the median scoring time
per crop on CPU. Scores are comparable within a method, not across methods.

Three methods share the top and are statistically close — `deep_features`
(0.83), `alexnet` (0.81), `hog` (0.80). `hog` runs at 0.45 ms, ~20× faster
than `alexnet` and ~370× faster than `deep_features`, which makes it the
default choice; reach for `deep_features` when accuracy matters more than
speed. Below the top tier there is a clear gap to `gabor` and the remaining
classical measures.

| scoring method | skill | speed (ms) | source |
|---|---|---|---|
| `deep_features` | **0.83** | 169 | Brachmann & Redies (2016), generalized to modern backbones |
| `alexnet` | **0.81** | 9.9 | Brachmann & Redies (2016) |
| `hog` | **0.80** | 0.45 | Renero-C. et al. (2017) |
| `gabor` | 0.71 | 14.9 | Shaker & Monadjemi (2015) |
| `gradient` | 0.67 | 0.72 | Gnutti, Guerrini & Leonardi (2021) |
| `multi_scale_gradient` | 0.65 | 0.36 | multi-scale extension of the gradient measure |
| `local_global` | 0.65 | 7.8 | Hogeweg et al. (2017), "PatchNN" |
| `sliding_window` | 0.65 | 0.07 | windowed variant of pixel correlation |
| `pixel_correlation` | 0.64 | 0.06 | Mayer & Landwehr (2018) |
| `phog` | 0.63 | 0.85 | Renero-C. et al. (2017) |
| `dct` | 0.61 | 0.15 | Gunlu & Bilge (2009) |
| `weighted_binary` | 0.58 | 0.05 | Bauerly & Liu (2006) |
| `eros` | 0.27 | 1.1 | Smith & Jenkinson (1999) |

Full citations live in each method's docstring (`imgsym/scoring/calculators.py`).

## Where is it symmetric? (PatchNN heatmap)

The `local_global` scoring method matches every patch to its best mirror-twin
on the other side of the axis. `calculate_heatmap` returns the per-patch match
quality as a map about the vertical center column (`aligned` below is the
image rotated so the detected axis is that center column — see the notebook).

```python
calc = imgsym.SymmetryCalculatorFactory.create("local_global")
score, heatmap = calc.calculate_heatmap(aligned)   # (H, W) map of mirror-match quality
```

<img src="https://raw.githubusercontent.com/maxwoe/imgsym/main/assets/butterfly_local_global_heatmap.png" alt="PatchNN local-symmetry map" width="560"/>

Blue = high local symmetry, red = low: the mirrored wing bands read blue,
while the gravel background is genuinely asymmetric and reads warm.
(Butterfly photo from Wikimedia Commons.)

## Benchmark

The skill numbers above come from the symmetry-scoring benchmark that ships in
`scripts/`: every method ranks true symmetry axes against perturbed negatives
on four annotated datasets (~3900 axis units), with derived results
(leaderboards, ablations, timings) tracked under `results/`. The raw per-image
score JSONs are not tracked — they regenerate deterministically (fixed seeds)
from the datasets, whose sources and expected directory layout are documented
in [data/README.md](https://github.com/maxwoe/imgsym/blob/main/data/README.md):

```bash
python scripts/run_discrimination.py                    # single-axis protocol
python scripts/run_discrimination.py --protocol multi   # multi-axis protocol
```

The full protocol and analysis are described in the accompanying
[paper](https://doi.org/10.3390/sym18081355).

## License

MIT — see [LICENSE](https://github.com/maxwoe/imgsym/blob/main/LICENSE).

## Citation

If you use imgsym or the benchmark in your research, please cite:

> Woehrer, M. Classical Versus Deep Mirror-Symmetry Scoring: A Benchmark of
> Thirteen Methods. *Symmetry* **2026**, *18*, 1355.
> [doi:10.3390/sym18081355](https://doi.org/10.3390/sym18081355)

```bibtex
@Article{sym18081355,
  author       = {Woehrer, Maximilian},
  title        = {Classical Versus Deep Mirror-Symmetry Scoring: A Benchmark of Thirteen Methods},
  journal      = {Symmetry},
  volume       = {18},
  year         = {2026},
  number       = {8},
  article-number = {1355},
  url          = {https://www.mdpi.com/2073-8994/18/8/1355},
  issn         = {2073-8994},
  doi          = {10.3390/sym18081355}
}
```
