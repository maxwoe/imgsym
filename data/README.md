# Datasets

The benchmark datasets are **not redistributed** with this repository — obtain
them from their original sources (below) and place them under `data/datasets/`
in the layout shown here. `scripts/run_discrimination.py` reads the root from
the `IMGSYM_DATA` environment variable (default `data/datasets`); the other
scripts expect to run from the repository root.

Datasets that are missing are simply skipped by the benchmark runner.

## Expected layout

```
data/datasets/
├── symComp13/
│   └── reflection_training/
│       ├── single_training/                  # images
│       ├── singleGT_training/singleGT_training.mat
│       ├── multiple_training/                # images (multi-axis subset)
│       └── multipleGT_training/
├── symComp17/
│   └── reflection_training/
│       └── ref_s/                            # refs_###.jpg + label_refs.txt
├── sym_datasets/
│   └── NYU/
│       ├── S/                                # I###.png + I###.mat (single axis)
│       └── M/                                # multi-axis counterpart
├── dendi/
│   ├── symmetry/reflection/coco/<split>/     # <id>.<ext> + <id>.<ext>.json
│   └── reflection_split.pt
└── PIX2PER Dataset/
    ├── images/images_art/  images/images_nat/
    └── labels/art/         labels/nat/       # one <stem>.csv per image
```

## Sources

| dataset | source |
|---|---|
| **symComp13** | CVPR 2013 workshop *Symmetry Detection from Real World Images* (Liu et al.) — workshop page: <https://sites.psu.edu/lpac/cvpr-2013-workshop/>. The set also circulates as SDRW, mirrored together with LDRS via the OneDrive linked in the CLIPSym README.md. |
| **symComp17** | ICCV 2017 challenge *Detecting Symmetry in the Wild* (Funk et al., ICCVW 2017) — datasets page: <https://sites.google.com/view/symcomp17/challenges/datasets> (`reflection_training.zip`; Sym-COCO via Dropbox links there). |
| **NYU** | NYU Symmetry Database (Cicconet et al.) — our copy came as `sym_datasets.zip` via <https://github.com/timyoung2333/CLIPSym>. |
| **DENDI** | Seo et al., *Reflection and Rotation Symmetry Detection via Equivariant Learning* (EquiSym), CVPR 2022 — [OneDrive download](https://postechackr-my.sharepoint.com/:u:/g/personal/lastborn94_postech_ac_kr/ES2ftVVmTc5Du78EBgfTGy8BwygV_HRa5nWciYeq3cTvoQ?e=y9ETja), also linked from <https://github.com/ahyunSeo/EquiSym>. |
| **PIX2PER** | **Not publicly released.** We obtained it directly from the authors of the source study ([Vision Research, 2026](https://www.sciencedirect.com/science/article/pii/S0042698926000775)) on request; please contact them for access. |

The exact loader expectations (file formats, GT conventions, skip rules) are
documented in the docstrings of `imgsym/evaluation/datasets.py`.
