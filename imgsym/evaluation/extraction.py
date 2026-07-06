"""Canonical axis-aligned subimage extraction for symmetry scoring.

All 13 scoring methods in :mod:`imgsym.scoring` assume the symmetry axis is the
vertical centerline of the frame and reflect the image with ``np.fliplr``. Real
annotations place axes anywhere, at any orientation. This module maps an
``(image, oriented axis)`` pair to a *canonical* subimage in which the annotated
axis is the exact vertical centerline, so that ``np.fliplr(subimage)`` reflects
the source across the true axis. Feeding that subimage to a scorer therefore
makes its center-axis assumption correct.

Design choices that distinguish this from :func:`imgsym.utils.patch.extract_patch`
(which is fine for the convenience API but unsuitable for measurement):

* **Single resampling pass.** One ``cv2.remap`` does rotation + crop together,
  avoiding the rotate-then-crop double interpolation.
* **Sub-pixel centered axis.** The output has *even* width ``2*Wh``; output
  column ``u`` samples the source at perpendicular offset ``(u + 0.5) - Wh``.
  The axis falls exactly between columns ``Wh-1`` and ``Wh``, so ``fliplr``
  pairs source points reflected across the true axis with no half-pixel bias.
* **Honest borders.** Out-of-image samples are filled with a constant, never
  reflect-padded (reflect-padding would manufacture symmetry). The fraction of
  out-of-image samples is logged so padded crops can be flagged.
* **Logged, relative support.** The chosen support width is recorded as a
  fraction of the image's short side; crops below ``min_support_frac`` are
  flagged degenerate so the evaluation loop can exclude them *and report how
  many* (no silent truncation).

The reflection property holds regardless of the sign chosen for the axis normal
or tangent: flipping either only mirrors the subimage as a whole, which leaves
any symmetry score unchanged.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Canonical axis representation
# --------------------------------------------------------------------------- #
@dataclass
class Axis:
    """An oriented reflection axis in image pixel coordinates.

    The axis is the line through ``(cx, cy)`` with direction ``angle`` (radians,
    measured as ``atan2(dy, dx)`` in image coordinates where y points down).
    ``along_extent`` is the annotated length of the axis; ``perp_extent`` is the
    width of the symmetric region across the axis when known (e.g. from a
    bounding box). Either may be ``None`` when the annotation does not provide
    it, in which case extent policies fall back to the image border.
    """

    cx: float
    cy: float
    angle: float
    along_extent: Optional[float] = None
    perp_extent: Optional[float] = None

    @classmethod
    def from_segment(cls, x1: float, y1: float, x2: float, y2: float) -> "Axis":
        """Build an axis from segment endpoints ``(x1, y1)-(x2, y2)``.

        Used by the line-segment datasets (symComp13/17, NYU, LDRS, DENDI).
        The segment fixes position, orientation and along-axis length; the
        perpendicular extent is unknown and left to the extent policy.
        """
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        angle = float(np.arctan2(y2 - y1, x2 - x1))
        length = float(np.hypot(x2 - x1, y2 - y1))
        return cls(cx=cx, cy=cy, angle=angle, along_extent=length, perp_extent=None)

    @classmethod
    def from_bbox(cls, cx: float, cy: float, width: float, height: float,
                  orientation: str, rotation_deg: float = 0.0) -> "Axis":
        """Build an axis from a PIX2PER bounding-box annotation.

        ``orientation`` is ``"vertical_axis"`` or ``"horizontal_axis"`` (the axis
        direction within the unrotated box), then the whole box is turned by
        ``rotation_deg``. A vertical axis runs along the box height; a horizontal
        axis along the box width. The box supplies *both* extents, which makes
        the ``"bbox"`` policy fully determined for PIX2PER.

        Note: ``rotation_deg`` is applied with the sign as-written in the
        annotation (``angle = base + radians(rotation_deg)``). This convention was
        confirmed correct for PIX2PER by visual inspection of rotated dominant
        axes (see :func:`draw_axis`); no sign flip is needed. PIX2PER rotations
        may exceed 90 degrees (axes are mod 180), so callers should pass
        ``rotation_deg % 180``.
        """
        if orientation.startswith("vertical"):
            base = np.pi / 2.0          # axis runs top-to-bottom (+y)
            along, perp = height, width
        elif orientation.startswith("horizontal"):
            base = 0.0                  # axis runs left-to-right (+x)
            along, perp = width, height
        else:
            raise ValueError(
                f"orientation must be 'vertical_axis' or 'horizontal_axis', got {orientation!r}")
        angle = float(base + np.radians(rotation_deg))
        return cls(cx=float(cx), cy=float(cy), angle=angle,
                   along_extent=float(along), perp_extent=float(perp))

    def endpoints(self, length: Optional[float] = None
                  ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return ``((x1, y1), (x2, y2))`` for drawing, using ``along_extent``."""
        if length is None:
            length = self.along_extent if self.along_extent else 0.0
        half = length / 2.0
        tx, ty = np.cos(self.angle), np.sin(self.angle)
        return ((self.cx - half * tx, self.cy - half * ty),
                (self.cx + half * tx, self.cy + half * ty))


# --------------------------------------------------------------------------- #
# Extent policies: (axis, image_shape) -> (perp_half, along_half) in pixels
# --------------------------------------------------------------------------- #
ExtentPolicy = Callable[[Axis, Tuple[int, int]], Tuple[float, float]]


def _ray_to_border(cx: float, cy: float, dx: float, dy: float,
                   w: int, h: int) -> float:
    """Distance from ``(cx, cy)`` along unit ``(dx, dy)`` to exit ``[0,w-1]x[0,h-1]``."""
    ts: List[float] = []
    if dx > 0:
        ts.append((w - 1 - cx) / dx)
    elif dx < 0:
        ts.append((0 - cx) / dx)
    if dy > 0:
        ts.append((h - 1 - cy) / dy)
    elif dy < 0:
        ts.append((0 - cy) / dy)
    pos = [t for t in ts if t >= 0]
    return min(pos) if pos else 0.0


def _half_extents_to_border(axis: Axis, image_shape: Tuple[int, int]
                            ) -> Tuple[float, float]:
    """Largest symmetric half-extents (perp, along) that stay inside the image."""
    h, w = image_shape[:2]
    ct, st = np.cos(axis.angle), np.sin(axis.angle)
    nx, ny = -st, ct        # axis normal (reflection direction)
    tx, ty = ct, st         # axis tangent (along the axis)
    perp_half = min(_ray_to_border(axis.cx, axis.cy, nx, ny, w, h),
                    _ray_to_border(axis.cx, axis.cy, -nx, -ny, w, h))
    along_half = min(_ray_to_border(axis.cx, axis.cy, tx, ty, w, h),
                     _ray_to_border(axis.cx, axis.cy, -tx, -ty, w, h))
    return float(perp_half), float(along_half)


def policy_min_edge(axis: Axis, image_shape: Tuple[int, int]) -> Tuple[float, float]:
    """Maximal axis-centered window that fits within the image (segment data).

    Perpendicular half-width is bounded by the nearer image edge along the axis
    normal -- the ``min(d_left, d_right)`` rule the scorers themselves use. The
    along-axis half-height is the annotated half-length when available, else the
    distance to the image border along the axis.
    """
    perp_half, along_border = _half_extents_to_border(axis, image_shape)
    if axis.along_extent:
        along_half = min(axis.along_extent / 2.0, along_border)
    else:
        along_half = along_border
    return perp_half, along_half


def policy_bbox(axis: Axis, image_shape: Tuple[int, int]) -> Tuple[float, float]:
    """Use the annotated bounding box for both extents (PIX2PER).

    Falls back to :func:`policy_min_edge` for whichever extent the annotation
    does not provide.
    """
    fallback_perp, fallback_along = policy_min_edge(axis, image_shape)
    perp_half = (axis.perp_extent / 2.0) if axis.perp_extent else fallback_perp
    along_half = (axis.along_extent / 2.0) if axis.along_extent else fallback_along
    return float(perp_half), float(along_half)


def make_fixed(perp_half: Optional[float] = None,
               along_half: Optional[float] = None,
               perp_frac: Optional[float] = None,
               along_frac: Optional[float] = None) -> ExtentPolicy:
    """Build a fixed-extent policy from absolute pixels or fractions of the short side.

    Absolute values take precedence; fractions are of ``min(H, W)``. Unset
    extents fall back to :func:`policy_min_edge`.
    """
    def _policy(axis: Axis, image_shape: Tuple[int, int]) -> Tuple[float, float]:
        h, w = image_shape[:2]
        short = float(min(h, w))
        fb_perp, fb_along = policy_min_edge(axis, image_shape)
        ph = perp_half if perp_half is not None else (
            perp_frac * short if perp_frac is not None else fb_perp)
        ah = along_half if along_half is not None else (
            along_frac * short if along_frac is not None else fb_along)
        return float(ph), float(ah)
    return _policy


_BUILTIN_POLICIES = {
    "min_edge": policy_min_edge,
    "bbox": policy_bbox,
}


def _resolve_policy(policy) -> ExtentPolicy:
    if callable(policy):
        return policy
    if policy in _BUILTIN_POLICIES:
        return _BUILTIN_POLICIES[policy]
    raise ValueError(
        f"Unknown extent policy {policy!r}. Use one of {list(_BUILTIN_POLICIES)}, "
        f"a make_fixed(...) policy, or a custom callable.")


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
@dataclass
class ExtractionInfo:
    """Per-unit provenance and quality flags for one extracted subimage."""

    policy: str
    cx: float
    cy: float
    angle_deg: float
    perp_half: float            # half-width across the axis, pixels
    along_half: float           # half-height along the axis, pixels
    out_w: int
    out_h: int
    perp_support_frac: float    # full perpendicular width / min(H, W)
    along_support_frac: float   # full along-axis height / min(H, W)
    oob_frac: float             # fraction of samples drawn from outside the image
    degenerate: bool
    exclude_reason: Optional[str]


@dataclass
class ExtractionResult:
    subimage: np.ndarray
    info: ExtractionInfo


def extract(image: np.ndarray,
            axis: Axis,
            policy="min_edge",
            *,
            min_support_frac: float = 0.10,
            max_oob_frac: float = 0.20,
            border_value: float = 0.0,
            output_size: Optional[int] = None) -> ExtractionResult:
    """Extract the canonical axis-centered subimage for one ``(image, axis)`` unit.

    Args:
        image: BGR or grayscale image.
        axis: Oriented :class:`Axis` to center.
        policy: Extent policy -- ``"min_edge"``, ``"bbox"``, a :func:`make_fixed`
            policy, or any ``(axis, shape) -> (perp_half, along_half)`` callable.
        min_support_frac: Minimum perpendicular support, as a fraction of
            ``min(H, W)``. Crops narrower than this are flagged ``degenerate`` so
            the caller can exclude them. A relative floor (rather than absolute
            pixels) keeps the criterion scale-invariant across image sizes.
        max_oob_frac: Maximum tolerated fraction of out-of-image samples before a
            crop is flagged ``degenerate``.
        border_value: Constant used for out-of-image samples (never reflect-pad).
        output_size: If set, resize the crop to (output_size, output_size) for
            cross-image comparison (e.g. the perceptual protocol). Safe for
            symmetry -- independent x/y scaling preserves mirror symmetry about
            the centerline. Leave None (native size) for discrimination, where
            true and wrong windows are already equal-sized.

    Returns:
        :class:`ExtractionResult` with the subimage (even width, axis on the
        vertical centerline) and an :class:`ExtractionInfo` record.
    """
    h, w = image.shape[:2]
    perp_half, along_half = _resolve_policy(policy)(axis, (h, w))

    wh = max(1, int(round(perp_half)))
    hh = max(1, int(round(along_half)))
    out_w, out_h = 2 * wh, 2 * hh

    # Output->source sampling grid. The +0.5 offset places the axis exactly
    # between columns wh-1 and wh, so fliplr is an exact reflection.
    u = np.arange(out_w, dtype=np.float32)
    v = np.arange(out_h, dtype=np.float32)
    p = (u + 0.5) - wh                     # perpendicular coordinate
    q = (v + 0.5) - hh                     # along-axis coordinate
    pp, qq = np.meshgrid(p, q)

    ct, st = np.cos(axis.angle), np.sin(axis.angle)
    nx, ny = -st, ct                       # normal
    tx, ty = ct, st                        # tangent
    map_x = (axis.cx + pp * nx + qq * tx).astype(np.float32)
    map_y = (axis.cy + pp * ny + qq * ty).astype(np.float32)

    sub = cv2.remap(image, map_x, map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=border_value)

    if output_size is not None:
        # Resize the canonical crop to a common size for cross-image comparison.
        # Independent x/y scaling preserves mirror symmetry about the centerline,
        # so the symmetry signal is unchanged. Support fractions below stay in
        # native window units (out_w/out_h), independent of the final pixel size.
        sub = cv2.resize(sub, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    oob = float(np.mean((map_x < 0) | (map_x > w - 1) |
                        (map_y < 0) | (map_y > h - 1)))

    short = float(min(h, w))
    perp_support_frac = out_w / short
    along_support_frac = out_h / short

    exclude_reason: Optional[str] = None
    if perp_support_frac < min_support_frac:
        exclude_reason = "tiny_support"
    elif oob > max_oob_frac:
        exclude_reason = "high_oob"
    degenerate = exclude_reason is not None

    policy_name = policy if isinstance(policy, str) else getattr(
        policy, "__name__", "custom")

    info = ExtractionInfo(
        policy=policy_name,
        cx=axis.cx, cy=axis.cy, angle_deg=float(np.degrees(axis.angle)),
        perp_half=float(wh), along_half=float(hh),
        out_w=out_w, out_h=out_h,
        perp_support_frac=perp_support_frac,
        along_support_frac=along_support_frac,
        oob_frac=oob,
        degenerate=degenerate,
        exclude_reason=exclude_reason,
    )
    return ExtractionResult(subimage=sub, info=info)


def partition_units(results: List[ExtractionResult]
                    ) -> Tuple[List[ExtractionResult], List[ExtractionResult]]:
    """Split extractions into (kept, dropped) by the degenerate flag.

    The caller should report ``len(dropped)`` and the reasons rather than
    silently discarding, so coverage stays auditable.
    """
    kept = [r for r in results if not r.info.degenerate]
    dropped = [r for r in results if r.info.degenerate]
    return kept, dropped


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def reflection_residual(subimage: np.ndarray) -> float:
    """Normalized mean ``|sub - fliplr(sub)|`` in [0, 1]; 0 == perfectly symmetric.

    A sanity check: a synthetic image that is exactly symmetric about a known
    axis should yield a residual near 0 after extraction along that axis.
    """
    s = subimage.astype(np.float32)
    diff = np.abs(s - np.fliplr(s))
    denom = np.abs(s).mean() + np.abs(np.fliplr(s)).mean() + 1e-8
    return float(diff.mean() / denom)


def draw_axis(image: np.ndarray, axis: Axis,
              length: Optional[float] = None,
              color: Tuple[int, int, int] = (0, 0, 255),
              thickness: int = 2) -> np.ndarray:
    """Return a copy of ``image`` with the axis drawn -- to validate annotations.

    Especially useful to confirm the PIX2PER rotation-sign convention in
    :meth:`Axis.from_bbox` before running a whole dataset.
    """
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    if length is None:
        length = axis.along_extent if axis.along_extent else 0.5 * min(image.shape[:2])
    (x1, y1), (x2, y2) = axis.endpoints(length)
    cv2.line(vis, (int(round(x1)), int(round(y1))),
             (int(round(x2)), int(round(y2))), color, thickness)
    cv2.circle(vis, (int(round(axis.cx)), int(round(axis.cy))), 3, color, -1)
    return vis
