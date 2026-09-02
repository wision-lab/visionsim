"""Thermal atlas: selection, tile allocation, shelf packing, and UV-space texel rasterization.

Pure geometry/allocation math on numpy arrays - **no bpy import, guarded or otherwise**. This
module has to import and be testable without Blender; :mod:`visionsim.simulate.heatsim.adapter`
(Task 2) is the only place that touches ``bpy`` objects and translates them into the arrays this
module consumes.

Why texels at all (see the design spec for the full argument): the thermal solver and shader both
operate on mesh vertices today, and artist meshes are pathologically uneven - a 16-vertex floor
spanning 80 m2 next to a 21k-vertex orchid. This module decouples both by giving area-dense
objects a scene-wide **temperature atlas** tile sized by surface area (not vertex count), whose
texel centers become simulation sample points and render lookup addresses simultaneously. Objects
that are already sampled densely by their own vertices are excluded and keep the vertex path.

Pipeline, in call order:

1. :func:`surface_area_m2` + :func:`select_for_atlas` - per object, decide vertex-path vs. atlas.
2. :func:`allocate` - size + shelf-pack tiles for every atlas-selected object into one atlas image.
3. :func:`rasterize_tile` - per object, turn its tile-local UV triangles into texel samples
   (position, normal, source face) for every covered texel.
4. :func:`dilate` - after the solve, push valid texel values outward across the tile's invalid
   margin so bilinear sampling at render time never reads an uninitialized texel.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

# `n_verts / max(area_m2, eps)` guards the selection ratio against division by (near-)zero area;
# eps is deliberately far below any real mesh's area so it only ever engages for area==0.
_AREA_EPS = 1.0e-9

# Zero-area (UV-degenerate) triangles are skipped outright; this is well below the UV-space area
# of any single covered texel for tile sizes in [tile_min, tile_max], so it never rejects a
# legitimately thin-but-visible triangle.
_DEGENERATE_UV_AREA_EPS = 1.0e-12

# Tolerance for the half-open edge test's "on the edge" comparison (w == 0). Texel-space
# coordinates here are bounded by `tile_max` (a few hundred) squared, well within float64's exact
# integer range for the synthetic/test geometry this module is exercised with.
_EDGE_EPS = 1.0e-9


@dataclass(frozen=True)
class TileSpec:
    """One object's placement in the shared atlas image."""

    obj_name: str
    size: tuple[int, int]
    """(w, h) texels; both are multiples of 4 and in [tile_min, tile_max] (tiles are square)."""
    offset: tuple[int, int]
    """(x, y) top-left corner of this tile in atlas pixels."""


@dataclass(frozen=True)
class AtlasLayout:
    """The full result of :func:`allocate`: every tile's placement plus the density actually used."""

    atlas_size: tuple[int, int]
    tiles: dict[str, TileSpec]
    effective_density: float
    """Equal to the requested density unless the soft-max safety valve rescaled it down."""
    rescaled: bool
    """True iff the soft-max safety valve engaged (see :func:`allocate`)."""


def surface_area_m2(verts_mm: np.ndarray, faces: np.ndarray) -> float:
    """Total triangle surface area, converted from mm^2 (input units) to m^2."""
    if faces.size == 0:
        return 0.0
    v0 = verts_mm[faces[:, 0]]
    v1 = verts_mm[faces[:, 1]]
    v2 = verts_mm[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area_mm2 = 0.5 * np.linalg.norm(cross, axis=1)
    return float(np.sum(area_mm2)) / 1.0e6


def select_for_atlas(
    n_verts: int, area_m2: float, density: float, *, writeback_possible: bool = True
) -> bool:
    """True iff this object should join the atlas rather than keep the per-vertex path.

    Normally that's a density call: native vertex density below the atlas target means
    the object's own vertices are too sparse to sample it well, so it joins.

    ``writeback_possible=False`` overrides the density rule unconditionally. It signals
    that the caller cannot write a per-vertex result back onto this object's base mesh
    at all (its base and evaluated vertex counts differ, e.g. a Bevel/Subdivision/
    Geometry-Nodes modifier) - the vertex path is not merely low-resolution here, it is
    structurally incapable of holding the solved field, so density is moot and the atlas
    (UV-addressed, not vertex-addressed) is the only path that can represent this object.
    """
    if not writeback_possible:
        return True
    return (n_verts / max(area_m2, _AREA_EPS)) < density


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return math.ceil(value / float(multiple)) * multiple


def _tile_side(area_m2: float, density: float, tile_min: int, tile_max: int) -> int:
    texel_count = max(area_m2, 0.0) * density
    side = math.ceil(math.sqrt(max(texel_count, 0.0)))
    side = _round_up_to_multiple(side, 4)
    return min(max(side, tile_min), tile_max)


def _sides_for_density(
    areas: dict[str, float], density: float, tile_min: int, tile_max: int
) -> dict[str, int]:
    return {name: _tile_side(area, density, tile_min, tile_max) for name, area in areas.items()}


def _shelf_pack(sides: dict[str, int], padding: int) -> tuple[dict[str, tuple[int, int]], tuple[int, int]]:
    """Pack squares (descending height, name tiebreak) into shelves with >=`padding` gaps.

    Bin width is chosen once, up front, from the padded total area (so it scales with the tile
    set rather than being fixed); the atlas then grows downward, shelf by shelf, to fit.
    """
    if not sides:
        return {}, (0, 0)

    items = sorted(sides.items(), key=lambda kv: (-kv[1], kv[0]))
    max_side = max(side for _, side in items)
    padded_area = sum((side + padding) * (side + padding) for _, side in items)
    width = max(max_side, math.ceil(math.sqrt(float(padded_area))))
    width = _round_up_to_multiple(width, 4)

    positions: dict[str, tuple[int, int]] = {}
    x, y, shelf_h, used_w = 0, 0, 0, 0
    for name, side in items:
        if x != 0 and x + side > width:
            y += shelf_h + padding
            x, shelf_h = 0, 0
        positions[name] = (x, y)
        used_w = max(used_w, x + side)
        x += side + padding
        shelf_h = max(shelf_h, side)
    used_h = y + shelf_h

    atlas_size = (_round_up_to_multiple(max(used_w, max_side), 4), _round_up_to_multiple(max(used_h, max_side), 4))
    return positions, atlas_size


def allocate(
    areas: dict[str, float],
    density: float,
    *,
    tile_min: int = 16,
    tile_max: int = 512,
    soft_max: int = 500_000,
    retained_vertex_count: int = 0,
    padding: int = 2,
) -> AtlasLayout:
    """Size and shelf-pack one atlas tile per object in `areas`.

    Tile side is purely area-driven (`ceil(sqrt(area_m2 * density))`, rounded up to a multiple of
    4, clamped to [tile_min, tile_max]) - never squeezed by a budget. `soft_max` is a warn-only
    safety valve: if `sum(side^2) + retained_vertex_count` exceeds it, density is rescaled
    uniformly (down) by `(soft_max - retained_vertex_count) / tiles_total` and tiles are resized
    at the new density; this is a single corrective pass, not an iterative solve, so clamping can
    still leave the total above `soft_max` - the valve's job is to warn loudly and shrink, not to
    silently degrade below the warning, and not to guarantee an exact bound.
    """
    if not areas:
        return AtlasLayout(atlas_size=(0, 0), tiles={}, effective_density=density, rescaled=False)

    sides = _sides_for_density(areas, density, tile_min, tile_max)
    tiles_total = sum(side * side for side in sides.values())
    effective_density = density
    rescaled = False

    if tiles_total + retained_vertex_count > soft_max and tiles_total > 0:
        budget = soft_max - retained_vertex_count
        ratio = budget / tiles_total if budget > 0 else _AREA_EPS
        effective_density = density * ratio
        rescaled = True
        sides = _sides_for_density(areas, effective_density, tile_min, tile_max)
        warnings.warn(
            f"HeatSim atlas: requested density {density:g} texels/m^2 would need {tiles_total} texels "
            f"(+{retained_vertex_count} retained vertices), exceeding soft_max={soft_max}; rescaling to "
            f"effective density {effective_density:g} texels/m^2."
        )

    positions, atlas_size = _shelf_pack(sides, padding)
    tiles = {
        name: TileSpec(obj_name=name, size=(side, side), offset=positions[name])
        for name, side in sides.items()
    }
    return AtlasLayout(atlas_size=atlas_size, tiles=tiles, effective_density=effective_density, rescaled=rescaled)


def _cross2(ux: np.ndarray, uy: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    return ux * vy - uy * vx


def _is_top_left_edge(v1: np.ndarray, v2: np.ndarray) -> bool:
    """Standard top-left fill rule: exactly one of an edge's two directed traversals owns it.

    For any shared boundary between two independently CCW-normalized triangles, the edge is
    traversed in opposite directions by each triangle, so this predicate is true for exactly one
    of the two - guaranteeing on-edge texels are claimed by exactly one triangle.
    """
    if v1[1] == v2[1]:
        return bool(v2[0] < v1[0])
    return bool(v2[1] < v1[1])


def _face_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n)
    if norm < _DEGENERATE_UV_AREA_EPS:
        return np.zeros(3, dtype=np.float64)
    return n / norm


def rasterize_tile(
    verts_mm: np.ndarray, faces: np.ndarray, loop_uv: np.ndarray, tile_size: tuple[int, int]
) -> dict[str, np.ndarray]:
    """Rasterize triangles (given per-triangle-corner UVs already in tile-local [0,1] space).

    Texel centers are tested at ``((x+0.5)/W, (y+0.5)/H)``; a half-open (top-left) edge rule means
    abutting triangles never double-claim a boundary texel. Triangles are processed in `faces`
    order and earlier ones win: once a texel is covered it is never revisited. Zero-UV-area
    (degenerate) triangles are skipped and contribute no texels (and no NaNs).

    Returns arrays for covered texels only: ``xy`` (K,2) int tile-local texel indices,
    ``position_mm`` (K,3) barycentric-interpolated 3D positions, ``normal`` (K,3) unit face
    normals, ``face`` (K,) int source-triangle index into `faces`.
    """
    width, height = tile_size
    covered: np.ndarray = np.zeros((height, width), dtype=bool)

    xy_parts: list[np.ndarray] = []
    pos_parts: list[np.ndarray] = []
    normal_parts: list[np.ndarray] = []
    face_parts: list[np.ndarray] = []

    n_faces = faces.shape[0]
    for fi in range(n_faces):
        i0, i1, i2 = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        uv0 = loop_uv[fi, 0].astype(np.float64) * (width, height)
        uv1 = loop_uv[fi, 1].astype(np.float64) * (width, height)
        uv2 = loop_uv[fi, 2].astype(np.float64) * (width, height)

        area2 = float(_cross2(uv1[0] - uv0[0], uv1[1] - uv0[1], uv2[0] - uv0[0], uv2[1] - uv0[1]))
        if abs(area2) < _DEGENERATE_UV_AREA_EPS:
            continue

        idx = (i0, i1, i2)
        pts = (uv0, uv1, uv2)
        if area2 < 0:
            idx = (i0, i2, i1)
            pts = (uv0, uv2, uv1)
            area2 = -area2
        a, b, c = pts
        ia, ib, ic = idx

        x_min = max(math.floor(min(a[0], b[0], c[0])), 0)
        x_max = min(math.ceil(max(a[0], b[0], c[0])), width)
        y_min = max(math.floor(min(a[1], b[1], c[1])), 0)
        y_max = min(math.ceil(max(a[1], b[1], c[1])), height)
        if x_min >= x_max or y_min >= y_max:
            continue

        xs = np.arange(x_min, x_max, dtype=np.float64) + 0.5
        ys = np.arange(y_min, y_max, dtype=np.float64) + 0.5
        px = xs[np.newaxis, :]
        py = ys[:, np.newaxis]

        w_c = _cross2(b[0] - a[0], b[1] - a[1], px - a[0], py - a[1])
        w_a = _cross2(c[0] - b[0], c[1] - b[1], px - b[0], py - b[1])
        w_b = _cross2(a[0] - c[0], a[1] - c[1], px - c[0], py - c[1])

        # Half-open edge rule: a top-left edge admits w==0 (boundary belongs to this triangle);
        # any other edge requires w>0 (boundary belongs to whichever neighbor treats it as
        # top-left). `_EDGE_EPS` only absorbs floating-point noise around the w==0 comparison.
        mask_c = w_c >= -_EDGE_EPS if _is_top_left_edge(a, b) else w_c >= _EDGE_EPS
        mask_a = w_a >= -_EDGE_EPS if _is_top_left_edge(b, c) else w_a >= _EDGE_EPS
        mask_b = w_b >= -_EDGE_EPS if _is_top_left_edge(c, a) else w_b >= _EDGE_EPS
        mask = mask_c & mask_a & mask_b
        if not mask.any():
            continue

        gy, gx = np.nonzero(mask)
        gxs: np.ndarray = gx + x_min
        gys: np.ndarray = gy + y_min
        keep = ~covered[gys, gxs]
        if not keep.any():
            continue
        gxs, gys, gy, gx = gxs[keep], gys[keep], gy[keep], gx[keep]
        covered[gys, gxs] = True

        wa = w_a[gy, gx] / area2
        wb = w_b[gy, gx] / area2
        wc = w_c[gy, gx] / area2
        positions = (
            verts_mm[ia] * wa[:, np.newaxis] + verts_mm[ib] * wb[:, np.newaxis] + verts_mm[ic] * wc[:, np.newaxis]
        )
        normal = _face_normal(verts_mm[i0], verts_mm[i1], verts_mm[i2])

        xy_parts.append(np.stack([gxs, gys], axis=1).astype(np.int64))
        pos_parts.append(positions)
        normal_parts.append(np.tile(normal, (len(gxs), 1)))
        face_parts.append(np.full(len(gxs), fi, dtype=np.int64))

    if not xy_parts:
        return {
            "xy": np.zeros((0, 2), dtype=np.int64),
            "position_mm": np.zeros((0, 3), dtype=np.float64),
            "normal": np.zeros((0, 3), dtype=np.float64),
            "face": np.zeros((0,), dtype=np.int64),
        }
    return {
        "xy": np.concatenate(xy_parts, axis=0),
        "position_mm": np.concatenate(pos_parts, axis=0),
        "normal": np.concatenate(normal_parts, axis=0),
        "face": np.concatenate(face_parts, axis=0),
    }


def _shift2d(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """arr shifted so that `out[y, x] == arr[y + dy, x + dx]`; out-of-bounds reads become 0."""
    shifted = np.roll(arr, shift=(-dy, -dx), axis=(0, 1))
    if dy > 0:
        shifted[-dy:, ...] = 0
    elif dy < 0:
        shifted[:-dy, ...] = 0
    if dx > 0:
        shifted[:, -dx:, ...] = 0
    elif dx < 0:
        shifted[:, :-dx, ...] = 0
    return shifted


_NEIGHBOR_OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def dilate(image: np.ndarray, valid: np.ndarray, iterations: int = 4) -> np.ndarray:
    """Push-out margin dilation: each invalid texel takes the mean of its valid 8-neighbors.

    Repeated for `iterations` passes; a texel filled on one pass becomes a valid source for the
    next (so the filled region grows outward by up to one texel per pass), but once a texel is
    valid - originally or newly filled - its value is never touched again.

    `valid` must be 2D (``(H, W)``) regardless of whether `image` is single- or multi-channel
    (``(H, W)`` or ``(H, W, C)``) - one validity bit per texel, not per channel.
    """
    valid_arr = np.asarray(valid)
    if valid_arr.ndim != 2:
        raise ValueError(f"dilate: `valid` must be 2D (H, W), got shape {valid_arr.shape}")

    out = np.array(image, dtype=np.float64, copy=True)
    valid_mask = np.array(valid_arr, dtype=bool, copy=True)

    for _ in range(iterations):
        if valid_mask.all():
            break
        valid_f = valid_mask.astype(np.float64)
        count = np.zeros(valid_mask.shape, dtype=np.float64)
        total = np.zeros_like(out)
        for dy, dx in _NEIGHBOR_OFFSETS:
            nb_valid = _shift2d(valid_f, dy, dx)
            nb_val = _shift2d(out, dy, dx)
            count += nb_valid
            total += nb_val * (nb_valid if out.ndim == 2 else nb_valid[..., np.newaxis])

        fillable = (~valid_mask) & (count > 0)
        if out.ndim == 2:
            out[fillable] = total[fillable] / count[fillable]
        else:
            out[fillable] = total[fillable] / count[fillable][..., np.newaxis]
        valid_mask |= fillable

    return out
