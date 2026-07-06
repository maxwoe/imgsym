"""Dataset annotation loaders -> canonical :class:`~imgsym.evaluation.extraction.Axis`.

Each symmetry dataset stores axes differently (CSV bounding boxes, line-segment
text, COCO polylines, MATLAB structs). These loaders parse them into a common
record so the rest of the evaluation harness never sees a dataset-specific
format. Every loader returns a :class:`Dataset` carrying both the parsed images
and an auditable ``stats`` dict (matched / skipped counts) so coverage is never
silently truncated.

Implemented (single-dominant-axis sources used by the scoring benchmark):

* :func:`load_pix2per`          -- PIX2PER CSV bounding boxes + per-axis
                                   ``num_labels`` (multi-axis, with strengths).
* :func:`load_symcomp17_single` -- ICCV'17 ``ref_s`` line-segment text.
* :func:`load_symcomp13_single` -- CVPR'13 / SDRW ``singleGT_training.mat``.
* :func:`load_nyu_single`       -- NYU ``NYU/S`` per-image ``.mat`` segments.

Multi-axis (every annotated axis; used by the per-axis ``all_axes`` protocol):

* :func:`load_symcomp13_multiple` -- CVPR'13 ``multipleGT_training.mat``.
* :func:`load_nyu_multiple`        -- NYU ``NYU/M`` per-image ``.mat`` segments.
* :func:`load_dendi_reflection`    -- DENDI (Seo et al. 2022) COCO/labelme JSON
                                      reflection polylines; dense, no dominance.

A uniform density cap (``max_axes``, default 10) drops regular tilings/rosettes,
where axis-level discrimination is ill-posed. Not yet implemented: LDRS.
"""

import csv
import glob
import os
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .extraction import Axis


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


@dataclass
class Unit:
    """One ``(image, axis)`` evaluation unit -- the atomic thing a scorer rates."""

    image_path: str
    axis: Axis
    strength: Optional[float]   # perceptual weight (PIX2PER num_labels); None if unknown
    image_id: str
    dataset: str


@dataclass
class AnnotatedImage:
    """All annotated axes for one image, plus optional per-axis strengths."""

    image_path: str
    image_id: str
    axes: List[Axis]
    strengths: Optional[List[float]] = None   # aligned to axes; None if dataset has none
    dataset: str = ""

    def dominant(self) -> Optional[Axis]:
        """Axis with the largest strength, or the sole axis when no strengths."""
        if not self.axes:
            return None
        if self.strengths:
            i = max(range(len(self.axes)), key=lambda k: self.strengths[k])
            return self.axes[i]
        return self.axes[0]

    def dominant_strength(self) -> Optional[float]:
        if not self.axes or not self.strengths:
            return None
        return max(self.strengths)


@dataclass
class Dataset:
    """A loaded dataset: parsed images + auditable load statistics."""

    name: str
    images: List[AnnotatedImage] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.images)

    def all_axes(self) -> Iterator[Unit]:
        """Yield one :class:`Unit` per annotated axis (perceptual protocol)."""
        for im in self.images:
            strengths = im.strengths or [None] * len(im.axes)
            for axis, s in zip(im.axes, strengths):
                yield Unit(im.image_path, axis, s, im.image_id, self.name)

    def dominant_axes(self) -> Iterator[Unit]:
        """Yield one :class:`Unit` per image, the dominant/single axis (discrimination)."""
        for im in self.images:
            ax = im.dominant()
            if ax is not None:
                yield Unit(im.image_path, ax, im.dominant_strength(), im.image_id, self.name)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _find_image(image_dir: str, stem: str) -> Optional[str]:
    for ext in _IMAGE_EXTS:
        p = os.path.join(image_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


# --------------------------------------------------------------------------- #
# PIX2PER  (perceptual primary)
# --------------------------------------------------------------------------- #
def load_pix2per(root: str, subset: str = "both", max_axes: int = 10**9) -> Dataset:
    """Load PIX2PER bounding-box annotations with per-axis perceptual strengths.

    Expects the layout::

        <root>/labels/<art|nat>/<stem>.csv      # rows: ,centerX,centerY,width_box,height_box,axis,rotation,num_labels
        <root>/images/images_<art|nat>/<stem>.<ext>

    Each CSV row is one annotated axis; ``num_labels`` (how many observers marked
    it) becomes the axis ``strength``. ``rotation`` is normalized mod 180 (axes
    are undirected). The dominant axis per image is the one with the most labels.

    Args:
        root: PIX2PER dataset root.
        subset: ``"art"``, ``"nat"`` or ``"both"``.
    """
    subsets = ["art", "nat"] if subset == "both" else [subset]
    images: List[AnnotatedImage] = []
    n_csv = n_matched = n_axes = skip_no_image = skip_empty = skip_badrow = 0

    for sub in subsets:
        label_dir = os.path.join(root, "labels", sub)
        image_dir = os.path.join(root, "images", f"images_{sub}")
        for csv_path in sorted(glob.glob(os.path.join(label_dir, "*.csv"))):
            n_csv += 1
            stem = os.path.splitext(os.path.basename(csv_path))[0]
            image_path = _find_image(image_dir, stem)
            if image_path is None:
                skip_no_image += 1
                continue
            axes: List[Axis] = []
            strengths: List[float] = []
            with open(csv_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        rot = float(row["rotation"]) % 180.0
                        axis = Axis.from_bbox(
                            float(row["centerX"]), float(row["centerY"]),
                            float(row["width_box"]), float(row["height_box"]),
                            row["axis"], rotation_deg=rot)
                        axes.append(axis)
                        strengths.append(float(row["num_labels"]))
                    except (KeyError, ValueError, TypeError):
                        skip_badrow += 1
            if not axes:
                skip_empty += 1
                continue
            images.append(AnnotatedImage(
                image_path=image_path, image_id=f"{sub}/{stem}",
                axes=axes, strengths=strengths, dataset="pix2per"))
            n_matched += 1
            n_axes += len(axes)

    n_dense = sum(1 for im in images if len(im.axes) > max_axes)
    images = [im for im in images if len(im.axes) <= max_axes]   # uniform density cap
    name = "pix2per" if subset == "both" else f"pix2per_{subset}"
    return Dataset(name=name, images=images, stats=dict(
        csv_files=n_csv, images_matched=len(images),
        total_axes=sum(len(im.axes) for im in images),
        skipped_no_image=skip_no_image, skipped_empty=skip_empty,
        skipped_bad_rows=skip_badrow, skipped_dense=n_dense, max_axes=max_axes))


# --------------------------------------------------------------------------- #
# symComp17 ref_s  (single-axis, ICCV'17)
# --------------------------------------------------------------------------- #
def load_symcomp17_single(ref_s_dir: str,
                          label_file: str = "label_refs.txt") -> Dataset:
    """Load the ICCV'17 ``ref_s`` single-axis set.

    ``label_refs.txt`` holds one ``x1,y1,x2,y2`` line per image, ordered to match
    the images sorted by filename (``refs_001.jpg`` -> line 1, ...). A line/image
    count mismatch is recorded in ``stats`` rather than guessed around.

    Note: the ``ref_m`` (multiple-axis) set ships no text GT in this distribution
    and is out of scope for single-axis scoring.
    """
    image_paths = sorted(
        p for p in glob.glob(os.path.join(ref_s_dir, "*"))
        if os.path.splitext(p)[1] in _IMAGE_EXTS)
    with open(os.path.join(ref_s_dir, label_file)) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    images: List[AnnotatedImage] = []
    n_pairs = min(len(image_paths), len(lines))
    skip_badrow = 0
    for image_path, line in zip(image_paths[:n_pairs], lines[:n_pairs]):
        parts = line.replace(",", " ").split()
        try:
            x1, y1, x2, y2 = (float(v) for v in parts[:4])
        except (ValueError, IndexError):
            skip_badrow += 1
            continue
        stem = os.path.splitext(os.path.basename(image_path))[0]
        images.append(AnnotatedImage(
            image_path=image_path, image_id=stem,
            axes=[Axis.from_segment(x1, y1, x2, y2)],
            strengths=None, dataset="symcomp17_s"))

    return Dataset(name="symcomp17_s", images=images, stats=dict(
        images_found=len(image_paths), label_lines=len(lines),
        images_matched=len(images), skipped_bad_rows=skip_badrow,
        count_mismatch=(len(image_paths) != len(lines))))


# --------------------------------------------------------------------------- #
# symComp13 single  (CVPR'13 / SDRW, single-axis training set)
# --------------------------------------------------------------------------- #
def load_symcomp13_single(root: str) -> Dataset:
    """Load the CVPR'13 single-axis set from ``singleGT_training.mat``.

    Expects the layout::

        <root>/reflection_training/singleGT_training/singleGT_training.mat
        <root>/reflection_training/single_training/<name>.jpg

    The ``.mat`` holds a ``gt`` struct array; each entry has ``.name`` (image
    file) and ``.ax = [x1, y1, x2, y2]``. Images are matched by name (the testing
    split ships no ground truth in this distribution, so only training is used).
    """
    import scipy.io as sio

    gt_path = os.path.join(root, "reflection_training", "singleGT_training",
                           "singleGT_training.mat")
    image_dir = os.path.join(root, "reflection_training", "single_training")
    gt = sio.loadmat(gt_path, squeeze_me=True, struct_as_record=False)["gt"]

    images: List[AnnotatedImage] = []
    skip_no_image = skip_bad = 0
    for entry in np.atleast_1d(gt):
        try:
            name = str(entry.name)
            ax = np.asarray(entry.ax, dtype=float).ravel()
            x1, y1, x2, y2 = ax[:4]
        except (AttributeError, ValueError):
            skip_bad += 1
            continue
        stem = os.path.splitext(name)[0]
        image_path = (os.path.join(image_dir, name)
                      if os.path.exists(os.path.join(image_dir, name))
                      else _find_image(image_dir, stem))
        if image_path is None:
            skip_no_image += 1
            continue
        images.append(AnnotatedImage(
            image_path=image_path, image_id=stem,
            axes=[Axis.from_segment(x1, y1, x2, y2)],
            strengths=None, dataset="symcomp13_s"))

    return Dataset(name="symcomp13_s", images=images, stats=dict(
        gt_entries=int(np.atleast_1d(gt).size), images_matched=len(images),
        skipped_no_image=skip_no_image, skipped_bad=skip_bad))


# --------------------------------------------------------------------------- #
# NYU single  (per-image .mat segments)
# --------------------------------------------------------------------------- #
def load_nyu_single(nyu_s_dir: str) -> Dataset:
    """Load the NYU single-axis set (the ``NYU/S`` folder).

    Each image ``I###.png`` has a paired ``I###.mat`` whose ``segments`` field is
    ``[[x1, y1], [x2, y2]]`` (one reflection axis per single-axis image).
    """
    import scipy.io as sio

    images: List[AnnotatedImage] = []
    skip_no_image = skip_bad = 0
    for mat_path in sorted(glob.glob(os.path.join(nyu_s_dir, "*.mat"))):
        stem = os.path.splitext(os.path.basename(mat_path))[0]
        image_path = _find_image(nyu_s_dir, stem)
        if image_path is None:
            skip_no_image += 1
            continue
        try:
            seg = np.asarray(sio.loadmat(mat_path, squeeze_me=True)["segments"],
                             dtype=float).reshape(-1, 2)
            (x1, y1), (x2, y2) = seg[0], seg[1]
        except (KeyError, ValueError, IndexError):
            skip_bad += 1
            continue
        images.append(AnnotatedImage(
            image_path=image_path, image_id=stem,
            axes=[Axis.from_segment(x1, y1, x2, y2)],
            strengths=None, dataset="nyu_s"))

    return Dataset(name="nyu_s", images=images, stats=dict(
        images_matched=len(images), skipped_no_image=skip_no_image,
        skipped_bad=skip_bad))


# --------------------------------------------------------------------------- #
# Multi-axis loaders (every annotated axis; for the per-axis protocol)
# --------------------------------------------------------------------------- #
def load_symcomp13_multiple(root: str, max_axes: int = 10) -> Dataset:
    """CVPR'13 multiple-axis set from ``multipleGT_training.mat``.

    Same layout as :func:`load_symcomp13_single` but the ``multipleGT_training``
    struct + ``multiple_training/`` images; each ``gt`` entry's ``.ax`` is an
    ``(n, 4)`` array of ``[x1, y1, x2, y2]`` rows (one per reflection axis). Images
    with more than ``max_axes`` axes are dropped (the uniform density cap)."""
    import scipy.io as sio

    gt_path = os.path.join(root, "reflection_training", "multipleGT_training",
                           "multipleGT_training.mat")
    image_dir = os.path.join(root, "reflection_training", "multiple_training")
    gt = sio.loadmat(gt_path, squeeze_me=True, struct_as_record=False)["gt"]

    images: List[AnnotatedImage] = []
    skip_no_image = skip_dense = skip_bad = 0
    for entry in np.atleast_1d(gt):
        try:
            name = str(entry.name)
            ax = np.asarray(entry.ax, dtype=float).reshape(-1, 4)
        except (AttributeError, ValueError):
            skip_bad += 1
            continue
        stem = os.path.splitext(name)[0]
        image_path = (os.path.join(image_dir, name)
                      if os.path.exists(os.path.join(image_dir, name))
                      else _find_image(image_dir, stem))
        if image_path is None:
            skip_no_image += 1
            continue
        if ax.shape[0] > max_axes:
            skip_dense += 1
            continue
        images.append(AnnotatedImage(
            image_path=image_path, image_id=stem,
            axes=[Axis.from_segment(*row) for row in ax],
            strengths=None, dataset="symcomp13_m"))

    return Dataset(name="symcomp13_m", images=images, stats=dict(
        gt_entries=int(np.atleast_1d(gt).size), images_matched=len(images),
        skipped_no_image=skip_no_image, skipped_dense=skip_dense,
        skipped_bad=skip_bad, max_axes=max_axes))


def load_nyu_multiple(nyu_m_dir: str, max_axes: int = 10) -> Dataset:
    """NYU multiple-axis set (the ``NYU/M`` folder).

    Each ``I###.mat`` ``segments`` field is an object array whose elements are the
    per-axis ``[[x1, y1], [x2, y2]]`` segments. Images with more than ``max_axes``
    axes are dropped (uniform density cap)."""
    import scipy.io as sio

    images: List[AnnotatedImage] = []
    skip_no_image = skip_dense = skip_bad = 0
    for mat_path in sorted(glob.glob(os.path.join(nyu_m_dir, "*.mat"))):
        stem = os.path.splitext(os.path.basename(mat_path))[0]
        image_path = _find_image(nyu_m_dir, stem)
        if image_path is None:
            skip_no_image += 1
            continue
        try:
            seg = np.asarray(sio.loadmat(mat_path, squeeze_me=True)["segments"])
            parts = list(np.atleast_1d(seg)) if seg.dtype == object else [seg]
            axes = []
            for s in parts:
                p = np.asarray(s, dtype=float).reshape(2, 2)
                axes.append(Axis.from_segment(p[0, 0], p[0, 1], p[1, 0], p[1, 1]))
        except (KeyError, ValueError, IndexError):
            skip_bad += 1
            continue
        if not axes:
            continue
        if len(axes) > max_axes:
            skip_dense += 1
            continue
        images.append(AnnotatedImage(
            image_path=image_path, image_id=stem, axes=axes,
            strengths=None, dataset="nyu_m"))

    return Dataset(name="nyu_m", images=images, stats=dict(
        images_matched=len(images), skipped_no_image=skip_no_image,
        skipped_dense=skip_dense, skipped_bad=skip_bad, max_axes=max_axes))


def load_dendi_reflection(dendi_root: str, max_axes: int = 10) -> Dataset:
    """Load DENDI reflection axes (Seo et al., *EquiSym*, CVPR 2022).

    Layout ``<dendi_root>/symmetry/reflection/coco/<split>/<id>.<ext>(.json)``: the
    image sits next to its annotation (the JSON path minus ``.json``). Each JSON's
    TOP-LEVEL ``figures`` (the current label; nested ``initialLabels`` are an
    annotation-revision history and are ignored) with ``label == "reflection"`` and
    a ``polyline`` shape gives one axis (``coordinates`` ``[[x1, y1], [x2, y2]]``).

    DENDI is dense, multi-axis, with NO per-axis dominance, so all reflection axes
    are kept (per-axis protocol). Images with more than ``max_axes`` axes -- regular
    tilings/rosettes where axis-level discrimination is ill-posed -- are excluded
    (the uniform density cap, default 10)."""
    import json

    images: List[AnnotatedImage] = []
    skip_no_image = skip_dense = skip_empty = skip_bad = 0
    pattern = os.path.join(dendi_root, "symmetry", "reflection", "coco", "**", "*.json")
    for jpath in sorted(glob.glob(pattern, recursive=True)):
        image_path = jpath[:-5]                        # strip ".json" -> "<id>.<ext>"
        if not os.path.exists(image_path):
            image_path = _find_image(os.path.dirname(jpath),
                                     os.path.splitext(os.path.basename(image_path))[0])
            if image_path is None:
                skip_no_image += 1
                continue
        try:
            with open(jpath) as fh:
                figures = json.load(fh).get("figures", [])
        except (ValueError, OSError):
            skip_bad += 1
            continue
        axes: List[Axis] = []
        for g in figures:
            shape = g.get("shape", {})
            if g.get("label") != "reflection" or shape.get("type") != "polyline":
                continue
            coords = shape.get("coordinates", [])
            if len(coords) >= 2:
                (x1, y1), (x2, y2) = coords[0], coords[1]
                axes.append(Axis.from_segment(float(x1), float(y1), float(x2), float(y2)))
        if not axes:
            skip_empty += 1
            continue
        if len(axes) > max_axes:
            skip_dense += 1
            continue
        stem = os.path.splitext(os.path.basename(image_path))[0]
        images.append(AnnotatedImage(
            image_path=image_path, image_id=stem, axes=axes,
            strengths=None, dataset="dendi_reflection"))

    return Dataset(name="dendi_reflection", images=images, stats=dict(
        images_matched=len(images), skipped_no_image=skip_no_image,
        skipped_dense=skip_dense, skipped_empty=skip_empty, skipped_bad=skip_bad,
        max_axes=max_axes))
