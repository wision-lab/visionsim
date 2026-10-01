"""Map Blender scene geometry and Cycles bakes to thermal solve points.

The solver uses millimetres, so geometry, irradiance and density are converted
at the boundary from metres, W/m² and kg/m³ respectively.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from visionsim.simulate.config import ThermalConfig
from visionsim.simulate.heatsim import atlas, cache, materials
from visionsim.simulate.heatsim.names import (
    ATLAS_COVERAGE_PROP,
    ATLAS_UV_LAYER_NAME,
    BAKE_UV_LAYER_NAME,
)
from visionsim.simulate.heatsim.physics import CYCLES_LOUT_TO_IRRADIANCE

try:
    import bpy  # type: ignore
except ImportError:  # pragma: no cover - only hit outside Blender
    bpy = None  # type: ignore

_log = logging.getLogger("rich")

# Unit conversions (everything the solver sees is in mm-based units).
_M_TO_MM = 1000.0
_KGM3_TO_KGMM3 = 1.0e9  # divide: kg/m^3 -> kg/mm^3  (1000**3)
_WM2_TO_WMM2 = 1.0e6    # divide: W/m^2 -> W/mm^2     (1000**2)


# ---------------------------------------------------------------------------
# Object selection + material resolution
# ---------------------------------------------------------------------------


def gather_meshes(scene: Any) -> list:
    """Select visible, nonempty meshes enabled for thermal simulation.

    Give linked duplicates independent mesh datablocks before writing UVs or
    attributes so one object's thermal field cannot overwrite another's.
    """
    out: list = []
    for obj in scene.objects:
        if getattr(obj, "type", None) != "MESH":
            continue
        if not obj.visible_get() or obj.hide_render:
            continue
        mesh = getattr(obj, "data", None)
        if mesh is None or not len(mesh.vertices) or not len(mesh.polygons):
            continue
        if not bool(getattr(obj, "heat_simulation_enabled", True)):
            continue
        out.append(obj)
    _ensure_single_user_meshes(out)
    return out


def _ensure_single_user_meshes(sim_objects: list) -> None:
    """Copy shared mesh datablocks before object-specific thermal writes."""
    unshared = 0
    for obj in sim_objects:
        mesh = getattr(obj, "data", None)
        if mesh is None or getattr(mesh, "users", 1) <= 1:
            continue
        obj.data = mesh.copy()
        unshared += 1
    if unshared:
        # Extra memory for heavily-instanced scenes (each un-shared copy is a full mesh
        # datablock) - worth a visible log line, not a warning (this is expected/correct
        # behavior for any scene using linked duplicates, not a misconfiguration).
        _log.info(
            "[heatsim.adapter] un-shared %d mesh datablock(s) referenced by multiple "
            "simulated objects (linked duplicates) so each object's solved field/atlas "
            "UVs write independently instead of colliding.", unshared,
        )


def _is_set(mat: Any, attr: str) -> bool:
    """True iff *attr* was explicitly set on the per-object PropertyGroup *mat*.

    ``obj.heat_sim_material`` is a ``PointerProperty`` registered on every object,
    so ``mat`` is never ``None`` and a ``FloatProperty`` always returns its group
    default - the global ``defaults`` (``--config.thermal.*``) would otherwise be
    unreachable.  Blender's ``bpy_struct.is_property_set`` distinguishes an
    explicitly-authored value from the registered default, restoring the locked
    "per-object overrides, globals as fallback" contract.  Non-Blender fakes that
    expose ``is_property_set`` are honoured too; anything else is treated as unset.
    """
    if mat is None:
        return False
    checker = getattr(mat, "is_property_set", None)
    if checker is None:
        return False
    try:
        return bool(checker(attr))
    except Exception:   # noqa: BLE001
        return False


def resolve_material(obj: Any, defaults: dict) -> dict:
    """Resolve per-object thermal parameters.

    Priority: an *explicitly set* per-object value on ``obj.heat_sim_material``
    (detected via :func:`_is_set`), else the global ``defaults`` dict.  Because the
    PropertyGroup is registered on every object, only ``is_property_set`` can tell a
    user-authored override from the registered group default; without that gate the
    global ``--config.thermal.*`` knobs would be silently inert.  SI units throughout
    (``thermal_diffusivity`` in mm^2/s, ``density`` in kg/m^3, ``specific_heat`` in
    J/(kg.K)).
    """
    mat = getattr(obj, "heat_sim_material", None)

    def _pick(attr: str, key: str) -> float:
        if _is_set(mat, attr):
            return float(getattr(mat, attr))
        return float(defaults[key])

    role = "FEM_PARTICIPANT"
    dirichlet_T = 0.0
    if mat is not None and _is_set(mat, "thermal_role"):
        role = str(mat.thermal_role or "FEM_PARTICIPANT").upper()
    if mat is not None and _is_set(mat, "dirichlet_temperature_K"):
        dirichlet_T = float(mat.dirichlet_temperature_K or 0.0)

    return {
        "initial_temperature_K": _pick("initial_temperature_K", "initial_temperature_K"),
        "thermal_diffusivity_mm2_s": _pick("thermal_diffusivity_mm2_s", "thermal_diffusivity_mm2_s"),
        "density_kg_m3": _pick("density_kg_m3", "density_kg_m3"),
        "specific_heat_J_kgK": _pick("specific_heat_J_kgK", "specific_heat_J_kgK"),
        "emissivity": float(np.clip(_pick("emissivity", "emissivity"), 0.0, 1.0)),
        "thermal_role": role,
        "dirichlet_temperature_K": dirichlet_T,
    }


# Robust percentile bounds for the preview colormap. Raw min/max lets a handful
# of runaway/artifact vertices (low heat-capacity materials or coarse-mesh spikes
# under an aggressive irradiance_scale) stretch tmax to thousands of Kelvin, which
# crushes the entire ambient scene to the cool/black end of inferno and leaves
# only the outlier blob visible. Clipping to P1..P99 discards those tails so the
# colormap spans the temperatures the bulk of the scene actually occupies.
_PREVIEW_PCT_LOW = 1.0
_PREVIEW_PCT_HIGH = 99.0


def global_temperature_range(history: dict[str, Any], default_K: float) -> tuple[float, float]:
    """Robust global colormap range ``(tmin, tmax)`` in Kelvin over the solved scene.

    Spans the final temperatures of every solved object, so the preview covers the
    actual data instead of a fixed 295-400 K band. To keep a few artifact-hot
    vertices from destroying the exposure, the bounds are the **1st and 99th
    percentiles** of the pooled final-timestep temperatures rather than the raw
    min/max. The lower bound is floored at ``default_K`` — unsolved meshes are
    stamped at that temperature — and the span is widened to at least 1 K so a
    near-uniform field does not collapse to a single colour.

    Args:
        history: ``{obj_name: (timesteps, vertices) array}`` from :func:`solve_scene`.
        default_K: Default initial temperature stamped on unsolved meshes.

    Returns:
        ``(tmin, tmax)`` with ``tmax - tmin >= 1.0``.
    """
    finals = []
    for arr in history.values():
        a = np.asarray(arr, dtype=float)
        if a.size:
            row = a[-1] if a.ndim >= 2 else a
            finals.append(np.asarray(row, dtype=float).reshape(-1))
    if not finals:
        return float(default_K), float(default_K) + 1.0
    pooled = np.concatenate(finals)
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return float(default_K), float(default_K) + 1.0
    tmin = float(np.percentile(pooled, _PREVIEW_PCT_LOW))
    tmax = float(np.percentile(pooled, _PREVIEW_PCT_HIGH))
    tmin = min(tmin, float(default_K))
    if tmax - tmin < 1.0:
        tmax = tmin + 1.0
    return tmin, tmax


def _extract_geometry(obj: Any) -> tuple | None:
    """``(verts_mm (N,3) float64, faces (M,3) int32, n_verts)`` for the evaluated
    mesh, or ``None`` if it has no geometry. Verts are world-space x 1000 (mm)
    and quads are triangulated. Cycles vertex bakes use these evaluated indices.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh = obj.evaluated_get(depsgraph).data
    n_verts = len(mesh.vertices)
    if n_verts == 0 or len(mesh.polygons) == 0:
        return None

    flat: np.ndarray = np.zeros(n_verts * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", flat)
    verts: np.ndarray = flat.reshape(n_verts, 3)
    mw = np.array(obj.matrix_world, dtype=np.float64)
    verts = (verts @ mw[:3, :3].T) + mw[:3, 3]
    verts = verts * _M_TO_MM  # world metres -> mm

    poly_count = len(mesh.polygons)
    loop_starts: np.ndarray = np.zeros(poly_count, dtype=np.int32)
    loop_totals: np.ndarray = np.zeros(poly_count, dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", loop_starts)
    mesh.polygons.foreach_get("loop_total", loop_totals)
    loop_vidx: np.ndarray = np.zeros(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vidx)

    face_arrays = []
    tri_starts = loop_starts[loop_totals == 3]
    if tri_starts.size:
        face_arrays.append(
            np.column_stack([loop_vidx[tri_starts], loop_vidx[tri_starts + 1], loop_vidx[tri_starts + 2]])
        )
    quad_starts = loop_starts[loop_totals == 4]
    if quad_starts.size:
        v0 = loop_vidx[quad_starts]
        v1 = loop_vidx[quad_starts + 1]
        v2 = loop_vidx[quad_starts + 2]
        v3 = loop_vidx[quad_starts + 3]
        face_arrays.append(np.column_stack([v0, v1, v2]))
        face_arrays.append(np.column_stack([v0, v2, v3]))
    if not face_arrays:
        return None

    faces = np.vstack(face_arrays).astype(np.int32)
    return verts, faces, n_verts


def _vertex_writeback_matches(obj: Any, evaluated_mm: np.ndarray) -> bool:
    """Require base vertex indices to match the evaluated solve points."""
    vertices = obj.data.vertices
    if len(vertices) != len(evaluated_mm):
        return False
    if not hasattr(vertices, "foreach_get"):
        return True  # Simple test meshes do not expose Blender's bulk API.
    local: np.ndarray = np.empty((len(vertices), 3), dtype=np.float64)
    vertices.foreach_get("co", local.reshape(-1))
    world = np.asarray(obj.matrix_world, dtype=np.float64)
    base_mm = ((local @ world[:3, :3].T) + world[:3, 3]) * _M_TO_MM
    return bool(np.allclose(base_mm, evaluated_mm, rtol=0.0, atol=1e-5))


# ---------------------------------------------------------------------------
# Thermal atlas (texel-sim) build
# ---------------------------------------------------------------------------


@dataclass
class AtlasPlan:
    """The output of :func:`build_atlas_plan`: per-object texel tables ready for
    :func:`_combine`'s TEXEL branch, plus the shared tile layout and the allocator
    settings that produced it (kept around so :func:`solve_scene` can fold them into
    the solve cache key).

    ``texels`` maps object name -> ``{"position_mm" (K,3), "normal" (K,3), "uv" (K,2),
    "xy" (K,2) int64, "face" (K,) int64, "face_material_index" (M,) int32}``. ``"xy"`` is
    the tile-LOCAL integer texel coordinate (before the tile's ``AtlasLayout`` offset is
    added) - :func:`write_atlas` uses it together with ``layout.tiles[name].offset``
    to scatter this object's solved texel temperatures into the shared atlas image. Only
    objects that made it all the way through selection, UV prep and rasterization appear
    here; every other sim object (excluded by :func:`atlas.select_for_atlas`, or rejected
    after a UV/vertex-count failure) is simply absent and keeps the per-vertex path in
    ``_combine``.
    """

    layout: atlas.AtlasLayout
    texels: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    density: float = 0.0
    tile_min: int = 16
    tile_max: int = 512
    soft_max: int = 500_000
    digest: str = ""


def _atlas_digest(
    layout: atlas.AtlasLayout,
    tile_min: int,
    tile_max: int,
    soft_max: int,
    texels: dict[str, dict[str, np.ndarray]],
) -> str:
    """Stable digest of an atlas allocation, for the solve cache key.

    Must be computed from the FINISHED :class:`AtlasPlan`, not from ``atlas.allocate``'s
    output alone: tiles are allocated for every object :func:`atlas.select_for_atlas`
    admits, but the per-object rasterization loop in :func:`build_atlas_plan` can still
    drop an object afterwards (UV unavailable, zero texels rasterized, evaluated mesh
    missing the atlas UV layer). An object that is allocated a tile but never
    contributes texels must NOT hash identically to one that fully participates - that
    previously let a change which promoted dropped objects into the atlas leave the
    cache key unchanged, silently reusing a stale solve. ``texels`` (``AtlasPlan.texels``)
    is keyed only by objects that actually rasterized, so folding in each one's realized
    texel count (sorted, for iteration-order stability) captures real participation on
    top of the existing tile geometry.
    """
    payload = {
        "density": round(float(layout.effective_density), 6),
        "tile_min": int(tile_min),
        "tile_max": int(tile_max),
        "soft_max": int(soft_max),
        "atlas_size": list(layout.atlas_size),
        "tiles": sorted(
            (name, list(spec.size), list(spec.offset)) for name, spec in layout.tiles.items()
        ),
        "texel_counts": sorted(
            (name, int(obj_texels["xy"].shape[0])) for name, obj_texels in texels.items()
        ),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _prepare_bake_uv(obj: Any) -> None:
    """Prepare a thermal UV layer for Cycles baking."""
    from visionsim.simulate.heatsim import irradiance

    irradiance.prepare_object_bake_uv(obj)


def _face_uv_and_slots_from_mesh(mesh: Any, uv_layer_name: str) -> tuple | None:
    """``(loop_uv (M,3,2) float64 in [0,1], face_material_index (M,) int32)`` for an
    arbitrary Blender mesh datablock, triangulated identically to
    :func:`_extract_geometry` (all triangle-polygons first, then each quad's two
    triangles in ``(v0,v1,v2)`` / ``(v0,v2,v3)`` order) so a texel's ``face`` index
    from ``atlas.rasterize_tile`` lines up with the same triangle in both this
    function's output and :func:`_extract_geometry`'s ``faces`` (when both are read
    from the SAME mesh - base or evaluated). Reads UVs from the named UV layer on
    ``mesh.loops``. Returns ``None`` if the mesh has no polygons or lacks that UV layer.

    Shared core for :func:`_extract_evaluated_face_uv_and_slots`, which reads the
    :func:`_extract_evaluated_face_uv_and_slots` (evaluated mesh).
    """
    if mesh is None:
        return None
    uv_layers = getattr(mesh, "uv_layers", None)
    uv_layer = uv_layers.get(uv_layer_name) if uv_layers is not None else None
    if uv_layer is None:
        return None
    poly_count = len(mesh.polygons)
    if poly_count == 0:
        return None

    loop_starts: np.ndarray = np.zeros(poly_count, dtype=np.int32)
    loop_totals: np.ndarray = np.zeros(poly_count, dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", loop_starts)
    mesh.polygons.foreach_get("loop_total", loop_totals)
    mat_index: np.ndarray = np.zeros(poly_count, dtype=np.int32)
    mesh.polygons.foreach_get("material_index", mat_index)

    uv_flat: np.ndarray = np.zeros(len(mesh.loops) * 2, dtype=np.float64)
    uv_layer.data.foreach_get("uv", uv_flat)
    loop_uv_all = uv_flat.reshape(-1, 2)

    uv_parts = []
    slot_parts = []
    tri_mask = loop_totals == 3
    tri_starts = loop_starts[tri_mask]
    if tri_starts.size:
        uv_parts.append(
            np.stack([loop_uv_all[tri_starts], loop_uv_all[tri_starts + 1], loop_uv_all[tri_starts + 2]], axis=1)
        )
        slot_parts.append(mat_index[tri_mask])
    quad_mask = loop_totals == 4
    quad_starts = loop_starts[quad_mask]
    if quad_starts.size:
        u0 = loop_uv_all[quad_starts]
        u1 = loop_uv_all[quad_starts + 1]
        u2 = loop_uv_all[quad_starts + 2]
        u3 = loop_uv_all[quad_starts + 3]
        uv_parts.append(np.stack([u0, u1, u2], axis=1))
        uv_parts.append(np.stack([u0, u2, u3], axis=1))
        slot_parts.append(mat_index[quad_mask])
        slot_parts.append(mat_index[quad_mask])
    if not uv_parts:
        return None

    loop_uv = np.vstack(uv_parts).astype(np.float64)
    face_material_index = np.concatenate(slot_parts).astype(np.int32)
    return loop_uv, face_material_index


def _extract_evaluated_face_uv_and_slots(obj: Any, uv_layer_name: str) -> tuple | None:
    """:func:`_face_uv_and_slots_from_mesh` for ``obj``'s EVALUATED mesh (post-modifier).

    Used to read back a UV layer (e.g. ``ATLAS_UV_LAYER_NAME``) that was written to the
    base mesh and has since propagated through the modifier stack - Bevel interpolates
    named UV layers onto new geometry, EdgeSplit duplicates them, and most Geometry
    Nodes setups preserve them, so this is how atlas participants with topology-changing
    modifiers get UVs that correspond 1:1 with :func:`_extract_geometry`'s evaluated
    ``faces`` for the SAME object. Returns ``None`` if the evaluated mesh has no polygons
    or the modifier stack dropped the named layer (some Geometry Nodes setups do).
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh = obj.evaluated_get(depsgraph).data
    return _face_uv_and_slots_from_mesh(mesh, uv_layer_name)


def _write_atlas_uv_layer(obj: Any, tile: atlas.TileSpec, atlas_size: tuple, src_layer_name: str) -> None:
    """Remap ``src_layer_name``'s per-loop UVs into ``tile``'s placement inside the
    shared atlas image and store the result as a fresh ``ATLAS_UV_LAYER_NAME`` UV layer
    on ``obj``'s base mesh. The modifier stack propagates this layer to the
    evaluated mesh used for rasterization and rendering. The caller verifies that
    propagation succeeded.
    """
    mesh = getattr(obj, "data", None)
    if mesh is None:
        return
    uv_layers = getattr(mesh, "uv_layers", None)
    if uv_layers is None:
        return
    src = uv_layers.get(src_layer_name)
    if src is None:
        return
    aw, ah = atlas_size
    if aw <= 0 or ah <= 0:
        return
    try:
        n_loops = len(mesh.loops)
        flat: np.ndarray = np.zeros(n_loops * 2, dtype=np.float64)
        src.data.foreach_get("uv", flat)
        uv = flat.reshape(-1, 2)

        tw, th = tile.size
        tx, ty = tile.offset
        atlas_uv = np.empty_like(uv)
        atlas_uv[:, 0] = (tx + uv[:, 0] * tw) / aw
        atlas_uv[:, 1] = (ty + uv[:, 1] * th) / ah

        if ATLAS_UV_LAYER_NAME in uv_layers:
            uv_layers.remove(uv_layers[ATLAS_UV_LAYER_NAME])
        dst = uv_layers.new(name=ATLAS_UV_LAYER_NAME)
        dst.data.foreach_set("uv", atlas_uv.ravel())
        mesh.update()
        # Force the depsgraph to re-evaluate the modifier stack with the new UV layer
        # in place, so the very next evaluated-mesh access (build_atlas_plan's read-back)
        # sees it instead of a stale cached evaluation from before this write.
        if bpy is not None:
            bpy.context.view_layer.update()
    except Exception as exc:   # noqa: BLE001
        _log.warning("[heatsim.adapter] '%s': failed to write %s: %s", obj.name, ATLAS_UV_LAYER_NAME, exc)


def build_atlas_plan(scene: Any, sim_objects: list, cfg: dict) -> AtlasPlan:
    """Select and rasterize thermal solve points on evaluated mesh surfaces."""
    mode = str(cfg.get("render_domain", "AUTO")).upper()
    if mode not in {"AUTO", "TEXEL"}:
        raise ValueError(f"Unsupported atlas representation {mode!r}")
    density = float(cfg.get("atlas_texel_density", 1500.0))  # keep in sync with ThermalConfig.atlas_texel_density
    tile_min = int(cfg.get("atlas_tile_min", 16))
    tile_max = int(cfg.get("atlas_tile_max", 512))
    soft_max = int(cfg.get("atlas_texel_soft_max", 500_000))

    geoms: dict[str, tuple] = {}
    areas: dict[str, float] = {}
    retained_vertex_count = 0
    for obj in sim_objects:
        geom = _extract_geometry(obj)
        if geom is None:
            continue
        verts, faces, n = geom
        geoms[obj.name] = (obj, verts, faces, n)
        area_m2 = atlas.surface_area_m2(verts, faces)
        writeback_possible = _vertex_writeback_matches(obj, verts)
        if area_m2 <= 0:
            if mode == "TEXEL" or not writeback_possible:
                raise ValueError(f"{obj.name!r} has no nondegenerate surface for thermal atlas sampling")
            retained_vertex_count += n
            continue
        triangles = verts[faces]
        triangle_areas = 0.5 * np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        ) / 1e6
        coarse_face = bool(np.max(triangle_areas) * density > 4.0)
        use_atlas = mode == "TEXEL" or coarse_face or atlas.select_for_atlas(
            n, area_m2, density, writeback_possible=writeback_possible
        )
        _log.info(
            "thermal: %s uses %s samples (area %.3g m², %d evaluated vertices, "
            "vertex write-back %s, largest triangle %.3g m²)",
            obj.name, "TEXEL" if use_atlas else "VERTEX", area_m2, n,
            "safe" if writeback_possible else "unsafe", float(np.max(triangle_areas)),
        )
        if use_atlas:
            areas[obj.name] = area_m2
        else:
            retained_vertex_count += n

    layout = atlas.allocate(
        areas, density, tile_min=tile_min, tile_max=tile_max, soft_max=soft_max,
        retained_vertex_count=retained_vertex_count, padding=_ATLAS_PACKING_PADDING,
    )

    texels: dict[str, dict[str, np.ndarray]] = {}
    for name in areas:
        obj, verts, faces, _n = geoms[name]
        tile = layout.tiles.get(name)
        if tile is None:
            raise RuntimeError(f"No thermal atlas tile was allocated for {name!r}")

        # Write the atlas UV layer onto the BASE mesh first (so Bevel/EdgeSplit/GN
        # modifiers propagate it) and force a depsgraph update, then read triangles'
        # UVs + material indices back from the EVALUATED mesh - the same mesh
        # `verts`/`faces` above came from - so face indices line up 1:1 regardless of
        # whether the base and evaluated vertex counts agree.
        _prepare_bake_uv(obj)
        _write_atlas_uv_layer(obj, tile, layout.atlas_size, BAKE_UV_LAYER_NAME)
        uv_result = _extract_evaluated_face_uv_and_slots(obj, ATLAS_UV_LAYER_NAME)
        if uv_result is None:
            raise RuntimeError(f"{name!r}: evaluated mesh has no usable {ATLAS_UV_LAYER_NAME} layer")
        atlas_loop_uv, face_material_index = uv_result
        if atlas_loop_uv.shape[0] != faces.shape[0]:
            raise RuntimeError(f"{name!r}: thermal UV and evaluated mesh triangulations differ")

        # `atlas_loop_uv` is already atlas-global [0,1] (written by _write_atlas_uv_layer
        # as `(tile.offset + bake_uv * tile.size) / atlas_size`); invert that remap back to
        # tile-local [0,1] here so `atlas.rasterize_tile` keeps its existing tile-local
        # contract unchanged.
        tw, th = tile.size
        tx, ty = tile.offset
        aw, ah = layout.atlas_size
        loop_uv = np.empty_like(atlas_loop_uv)
        loop_uv[..., 0] = (atlas_loop_uv[..., 0] * aw - tx) / tw
        loop_uv[..., 1] = (atlas_loop_uv[..., 1] * ah - ty) / th

        raster = atlas.rasterize_tile(verts, faces, loop_uv, tile.size)
        if raster["xy"].shape[0] == 0:
            raise RuntimeError(f"{name!r}: thermal atlas rasterized no solve points")

        uv_at_texel = np.empty((raster["xy"].shape[0], 2), dtype=np.float64)
        uv_at_texel[:, 0] = (raster["xy"][:, 0].astype(np.float64) + 0.5) / tw
        uv_at_texel[:, 1] = (raster["xy"][:, 1].astype(np.float64) + 0.5) / th

        texels[name] = {
            "position_mm": raster["position_mm"],
            "normal": raster["normal"],
            "uv": uv_at_texel,
            "xy": raster["xy"],
            "face": raster["face"],
            "face_material_index": face_material_index,
        }

    digest = _atlas_digest(layout, tile_min, tile_max, soft_max, texels)
    return AtlasPlan(
        layout=layout, texels=texels, density=layout.effective_density,
        tile_min=tile_min, tile_max=tile_max, soft_max=soft_max, digest=digest,
    )


# Fill small UV holes and the filtering margin without joining neighboring tiles.
# Larger uncovered areas remain invalid and use the shader's fallback temperature.
_ATLAS_DILATE_ITERATIONS = 8
_ATLAS_PACKING_PADDING = 17
assert 2 * _ATLAS_DILATE_ITERATIONS <= _ATLAS_PACKING_PADDING, "dilation would bridge tile padding"


def _scatter_atlas_arrays(history: dict, atlas_plan: AtlasPlan) -> tuple[np.ndarray, np.ndarray]:
    """Pure-numpy core of :func:`write_atlas`: scatter + dilate, no ``bpy``.

    Returns ``(temperature, alpha)``, both ``(H, W)`` float64/bool-as-float64 arrays sized
    to ``atlas_plan.layout.atlas_size``. ``temperature`` is dilated so bilinear sampling
    never reads an untouched (zero) texel; ``alpha`` is 1.0 wherever the dilation actually
    reached (originally-solved texels AND their filled margin), 0.0 everywhere else
    (never-covered tile interior, inter-tile padding, and any unused atlas margin).
    """
    width, height = atlas_plan.layout.atlas_size
    temp: np.ndarray = np.zeros((height, width), dtype=np.float64)
    valid: np.ndarray = np.zeros((height, width), dtype=bool)

    for name, tex in atlas_plan.texels.items():
        tile = atlas_plan.layout.tiles.get(name)
        hist = history.get(name)
        if tile is None or hist is None:
            continue
        arr = np.asarray(hist, dtype=np.float64)
        xy = tex.get("xy")
        xy = np.asarray(xy, dtype=np.int64) if xy is not None else None
        if arr.ndim != 2 or arr.shape[0] == 0 or xy is None or xy.shape[0] != arr.shape[1]:
            _log.warning(
                "[heatsim.adapter] write_atlas: '%s' history shape %s does not match its "
                "texel table (%s); skipped -- that object's tile stays unsolved (dilated "
                "from neighbours, or zero/invalid if isolated).",
                name, arr.shape, None if xy is None else xy.shape,
            )
            continue

        final = arr[-1]
        tx, ty = tile.offset
        px = xy[:, 0] + tx
        py = xy[:, 1] + ty
        temp[py, px] = final
        valid[py, px] = True

    # Dilate the temperature field, then dilate a {0,1} "alpha image" seeded with the SAME
    # `valid` mask and iteration count: every fillable pixel there is a weighted mean of
    # neighbours that are each exactly 1.0 (valid texels always hold 1.0 in this second
    # array), so the result is exactly 1.0 wherever the real dilation above reached, and
    # exactly 0.0 wherever it didn't -- i.e. the post-dilation validity mask, with no new
    # atlas.py API needed.
    temp_dilated = atlas.dilate(temp, valid, iterations=_ATLAS_DILATE_ITERATIONS)
    alpha_dilated = atlas.dilate(valid.astype(np.float64), valid, iterations=_ATLAS_DILATE_ITERATIONS)
    alpha = (alpha_dilated > 0.5).astype(np.float64)
    return temp_dilated, alpha


def write_atlas(
    history: dict, atlas_plan: AtlasPlan, cache_root: Path,
    defaults: dict | None = None, assignment: Any | None = None,
) -> Path:
    """Write the final-timestep texel temperatures to a 32-bit float EXR atlas image.

    Scatters every atlas-participating object's LAST solved timestep into the shared atlas
    array at its tile's texels (:func:`_scatter_atlas_arrays`), push-out dilates across the
    invalid margin so render-time bilinear filtering never samples an untouched texel, and
    saves an RGBA float EXR (R=G=B=temperature in Kelvin, A=1 where valid/dilated coverage
    exists, 0 elsewhere) to ``<cache_root>/atlas_<digest>/atlas_temperature.exr``. The caller
    (``blender.py``) is expected to load and pack the result as the ``HeatSim_Temperature_Atlas``
    Blender image before wiring the shader (:mod:`thermal_shader`) or rendering.

    Args:
        history: ``{obj_name: (timesteps, K) array}`` as returned by :func:`solve_scene`
            (called with this same ``atlas_plan``); ``K`` must match each atlas object's
            texel count for its temperatures to be scattered (a mismatch is logged and that
            object's tile is left to dilation/zero).
        atlas_plan: The plan from :func:`build_atlas_plan` that produced ``history``'s
            TEXEL-mode objects.
        cache_root: Root directory for thermal caches (same one passed to :func:`solve_scene`).

    Returns:
        Path to the written EXR file.
    """
    cache_root = Path(cache_root)
    result_hash = hashlib.sha256((atlas_plan.digest or "noatlas").encode())
    for name in sorted(atlas_plan.texels):
        result_hash.update(name.encode())
        result_hash.update(np.asarray(history[name][-1], dtype="<f8").tobytes())
    width, height = atlas_plan.layout.atlas_size
    width, height = max(int(width), 1), max(int(height), 1)
    if atlas_plan.layout.atlas_size == (0, 0) or not atlas_plan.texels:
        temp_dilated: np.ndarray = np.zeros((height, width), dtype=np.float64)
        alpha: np.ndarray = np.zeros((height, width), dtype=np.float64)
    else:
        temp_dilated, alpha = _scatter_atlas_arrays(history, atlas_plan)

    emissivity_history: dict[str, np.ndarray] = {}
    for name, tex in atlas_plan.texels.items():
        count = len(tex["xy"])
        eps: np.ndarray = np.full(count, ThermalConfig.emissivity, dtype=np.float64)
        if defaults is not None:
            obj = next((o for o in bpy.context.scene.objects if o.name == name), None)
            if obj is None:
                raise RuntimeError(f"Thermal atlas object {name!r} is missing from the scene")
            material = resolve_material(obj, defaults)
            eps.fill(float(material["emissivity"]))
            if assignment is not None:
                per_face = materials.resolve_face_materials(
                    obj, assignment, material, np.asarray(tex["face_material_index"], dtype=np.int64)
                )
                if per_face is not None:
                    eps = np.asarray(per_face["eps"], dtype=np.float64)[np.asarray(tex["face"], dtype=np.int64)]
        emissivity_history[name] = eps[None, :]
    emissivity = (
        _scatter_atlas_arrays(emissivity_history, atlas_plan)[0]
        if emissivity_history else np.zeros((height, width), dtype=np.float64)
    )

    rgba: np.ndarray = np.zeros((height, width, 4), dtype=np.float32)
    rgba[..., 0] = temp_dilated
    rgba[..., 1] = emissivity
    rgba[..., 2] = temp_dilated
    rgba[..., 3] = alpha

    result_hash.update(rgba.tobytes())
    out_dir = cache_root / f"atlas_{result_hash.hexdigest()[:20]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "atlas_temperature.exr"

    write_image_name = "HeatSim_Temperature_Atlas_Write"
    existing = bpy.data.images.get(write_image_name)
    if existing is not None:
        bpy.data.images.remove(existing)
    image = bpy.data.images.new(write_image_name, width=width, height=height, alpha=True, float_buffer=True)
    # Temperatures are data, not color: bpy.data.images.new()'s default colorspace tag
    # (here, effectively a non-identity "Linear Rec.709" role under this OCIO config --
    # empirically confirmed to apply the sRGB-like OETF on save) makes Image.save() treat
    # the raw Kelvin values as scene-linear color and re-encode them, corrupting the
    # written EXR's absolute values (295.0 K -> ~11.23 in the file). Tag Non-Color so
    # save()/load() are the identity transform and the file holds the literal float value.
    image.colorspace_settings.name = "Non-Color"
    image.pixels.foreach_set(rgba.ravel())
    image.filepath_raw = str(out_path)
    image.file_format = "OPEN_EXR"
    try:
        image.save()
    finally:
        bpy.data.images.remove(image)

    _log.debug(
        "[heatsim.adapter] write_atlas: wrote %s (%dx%d, %d object(s))",
        out_path, width, height, len(atlas_plan.texels),
    )
    return out_path


def _sample_bilinear(pixels: np.ndarray, width: int, height: int, uv: np.ndarray) -> np.ndarray:
    """Vectorized bilinear luminance sample from an ``(H, W, 3)`` image at ``(K, 2)``
    UV coordinates in ``[0, 1]``. Mirrors ``irradiance._bilinear_sample`` (same
    clamp/floor/lerp math and Rec.709 luma weights) but for many points at once."""
    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(uv[:, 1], 0.0, 1.0)
    x = u * (width - 1)
    y = v * (height - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    tx = (x - x0)[:, np.newaxis]
    ty = (y - y0)[:, np.newaxis]

    c00 = pixels[y0, x0]
    c10 = pixels[y0, x1]
    c01 = pixels[y1, x0]
    c11 = pixels[y1, x1]
    c0 = c00 * (1 - tx) + c10 * tx
    c1 = c01 * (1 - tx) + c11 * tx
    rgb = c0 * (1 - ty) + c1 * ty
    return rgb @ np.array((0.2126, 0.7152, 0.0722), dtype=np.float64)


def _texel_albedo(scene: Any, obj: Any, uv_at_texel: np.ndarray, texture_size: int) -> np.ndarray:
    """Sample a Cycles albedo bake at the thermal texel centers."""
    k = int(uv_at_texel.shape[0])
    if k == 0:
        return np.zeros(0, dtype=np.float64)
    from visionsim.simulate.heatsim import irradiance

    baked = irradiance.bake_albedo_map(scene, obj, texture_size)
    if baked is None or getattr(baked, "pixels", None) is None:
        raise RuntimeError(f"Albedo bake failed for {obj.name!r}")
    luma = _sample_bilinear(baked.pixels, int(baked.width), int(baked.height), uv_at_texel)
    return np.clip(luma, 0.0, 1.0)


def _texel_irradiance_cycles(
    scene: Any, obj: Any, uv_at_texel: np.ndarray, texture_size: int, samples: int | None = None
) -> np.ndarray:
    """Bilinear-sample ``obj``'s Cycles irradiance bake at texel UV centers.

    The Cycles counterpart to :func:`_texel_albedo`, sampling the same UVs from the
    DIFFUSE DIRECT+INDIRECT bake instead of the COLOR bake. Returns *incident* W/m^2;
    the caller applies (1 - albedo) to get absorbed flux.

    """
    k = int(uv_at_texel.shape[0])
    if k == 0:
        return np.zeros(0, dtype=np.float64)
    from visionsim.simulate.heatsim import irradiance

    baked = irradiance.bake_irradiance_map(scene, obj, texture_size, samples=samples)
    if baked is None or getattr(baked, "pixels", None) is None:
        raise RuntimeError(f"Irradiance bake failed for {obj.name!r}")
    lum = _sample_bilinear(baked.pixels, int(baked.width), int(baked.height), uv_at_texel)
    return np.maximum(np.asarray(lum, dtype=np.float64) * CYCLES_LOUT_TO_IRRADIANCE, 0.0)


def _compute_irradiance_cycles(scene: Any, sim_objects: list, solver_cfg: dict, defaults: dict) -> dict:
    """Bake incident light and albedo at vertices and return absorbed W/m²."""
    from visionsim.simulate.heatsim import irradiance

    texture_size = int(solver_cfg.get("irradiance_texture_size", 512))
    bake_samples = int(solver_cfg.get("bake_samples", 1024))
    out: dict = {}
    for obj in sim_objects:
        if resolve_material(obj, defaults)["thermal_role"] == "DIRICHLET_SOURCE":
            continue  # mirrors compute_per_vertex_irradiance's own Dirichlet skip
        baked = irradiance.bake_irradiance_map(scene, obj, texture_size, samples=bake_samples)
        if baked is None or getattr(baked, "vertex_flux", None) is None:
            raise RuntimeError(f"Irradiance bake failed for {obj.name!r}")
        incident = np.asarray(baked.vertex_flux, dtype=np.float64).reshape(-1)
        albedo = irradiance.bake_vertex_albedo(scene, obj, texture_size)
        if albedo.shape != incident.shape or not np.all(np.isfinite(incident)):
            raise RuntimeError(f"Bake samples do not match the evaluated mesh for {obj.name!r}")
        out[obj] = np.maximum(incident * (1.0 - albedo), 0.0)
    return out


def _compute_texel_irradiance(
    scene: Any, sim_objects: list, atlas_plan: AtlasPlan, solver_cfg: dict, defaults: dict
) -> dict:
    """Return absorbed Cycles flux at each participating atlas texel."""

    texture_size = int(solver_cfg.get("irradiance_texture_size", 512))
    bake_samples = int(solver_cfg.get("bake_samples", 1024))
    by_name = {o.name: o for o in sim_objects}

    out: dict = {}
    for name, tex in atlas_plan.texels.items():
        obj = by_name.get(name)
        if obj is None:
            continue
        if resolve_material(obj, defaults)["thermal_role"] == "DIRICHLET_SOURCE":
            continue  # mirrors compute_per_vertex_irradiance's own Dirichlet skip
        uv = np.asarray(tex["uv"], dtype=np.float64)
        albedo = _texel_albedo(scene, obj, uv, texture_size)
        incident = _texel_irradiance_cycles(scene, obj, uv, texture_size, samples=bake_samples)
        flux = incident * (1.0 - albedo)
        if not np.all(np.isfinite(flux)):
            raise RuntimeError(f"Non-finite absorbed flux for {obj.name!r}")
        out[obj] = np.asarray(flux, dtype=np.float64).reshape(-1)
    return out


# ---------------------------------------------------------------------------
# Cycles irradiance
# ---------------------------------------------------------------------------


def _compute_irradiance(scene: Any, sim_objects: list, solver_cfg: dict, defaults: dict) -> dict:
    """Return Cycles-baked absorbed flux at each vertex."""
    return _compute_irradiance_cycles(scene, sim_objects, solver_cfg, defaults)


# ---------------------------------------------------------------------------
# Combine objects into one FEM system
# ---------------------------------------------------------------------------


def _combine_texel_object(
    obj: Any,
    tex: dict[str, np.ndarray],
    flux_by_obj: dict,
    defaults: dict,
    assignment: Any | None,
    irradiance_scale: float,
    verts_l: list,
    irr_l: list,
    t0_l: list,
    alpha_l: list,
    rho_l: list,
    c_l: list,
    eps_l: list,
    bmask_l: list,
    layout: list,
    offset: int,
) -> int:
    """Append one atlas-participating object's texel points to :func:`_combine`'s
    per-array accumulators and return the updated ``offset``.

    Mirrors the per-vertex branch's material/Dirichlet-pinning logic exactly, just at
    per-FACE resolution (:func:`materials.resolve_face_materials`, no seam averaging)
    instead of per vertex, and reading position/normal from the rasterized tile instead
    of :func:`_extract_geometry`.
    """
    positions = np.asarray(tex["position_mm"], dtype=np.float64).reshape(-1, 3)
    k = int(positions.shape[0])
    if k == 0:
        return offset

    mat = resolve_material(obj, defaults)
    face_idx = np.asarray(tex["face"], dtype=np.int64)

    per_face = None
    if assignment is not None:
        per_face = materials.resolve_face_materials(
            obj, assignment, mat, np.asarray(tex["face_material_index"], dtype=np.int64)
        )

    flux = flux_by_obj.get(obj)
    if flux is not None and int(np.asarray(flux).reshape(-1).shape[0]) == k:
        irr = (np.asarray(flux, dtype=np.float64).reshape(-1) / _WM2_TO_WMM2) * irradiance_scale
    else:
        irr = np.zeros(k, dtype=np.float64)

    if per_face is not None:
        # Per-face resolution: every field is already (K,) via `face_idx` lookup.
        # Dirichlet texels are pinned individually - alpha=0, no incident flux,
        # excluded from the radiation/convection boundary - exactly what the
        # per-vertex/object-level branches do. T0 for a non-pinned texel is the
        # simulation's ambient initial condition, not resolve_face_materials'
        # per-face value (which, for a face on a Dirichlet slot, IS the reservoir
        # temperature) - mirrors the per-vertex branch's same reasoning.
        rho = per_face["rho"][face_idx]
        c_vec = per_face["c"][face_idx]
        eps = per_face["eps"][face_idx]
        dmask = per_face["dirichlet_mask"][face_idx]
        t0 = np.where(dmask, per_face["t0"][face_idx], mat["initial_temperature_K"])
        alpha = np.where(dmask, 0.0, per_face["alpha"][face_idx])
        irr = np.where(dmask, 0.0, irr)
        bmask = ~dmask
    else:
        rho = np.full(k, mat["density_kg_m3"], dtype=np.float64)
        c_vec = np.full(k, mat["specific_heat_J_kgK"], dtype=np.float64)
        eps = np.full(k, mat["emissivity"], dtype=np.float64)
        if mat["thermal_role"] == "DIRICHLET_SOURCE":
            t_dir = mat["dirichlet_temperature_K"] or mat["initial_temperature_K"]
            t0 = np.full(k, float(t_dir), dtype=np.float64)
            alpha = np.zeros(k, dtype=np.float64)
            irr = np.zeros(k, dtype=np.float64)
            bmask = np.zeros(k, dtype=bool)
        else:
            t0 = np.full(k, mat["initial_temperature_K"], dtype=np.float64)
            alpha = np.full(k, mat["thermal_diffusivity_mm2_s"], dtype=np.float64)
            bmask = np.ones(k, dtype=bool)

    verts_l.append(positions)
    irr_l.append(irr)
    t0_l.append(t0)
    alpha_l.append(alpha)
    rho_l.append(rho)
    c_l.append(c_vec)
    eps_l.append(eps)
    bmask_l.append(bmask)
    layout.append((obj.name, offset, k, "TEXEL"))
    return offset + k


def _combine(
    sim_objects: list,
    flux_by_obj: dict,
    defaults: dict,
    solver_cfg: dict,
    assignment: Any | None = None,
    atlas_plan: AtlasPlan | None = None,
) -> SimpleNamespace | None:
    """Stack per-object geometry + material vectors into one solver-ready system.

    Surface vertices come first (layout records each object's slice); optional
    interior POINTS-mode samples are appended afterwards by
    the surface slices stay valid.

    When *assignment* (a parsed thermal sidecar) is supplied and an object has
    usable material slots, alpha/rho/c/eps/T0 and the Dirichlet mask are resolved
    **per vertex** from the slot assignment
    (:func:`materials.resolve_vertex_materials`) instead of being filled with one
    object-level constant. ``resolve_material`` still supplies the per-object
    fallback for unassigned slots, so addon-authored blends are unaffected. With
    ``assignment=None`` this function behaves exactly as before.

    When *atlas_plan* is supplied, any object with an entry in ``atlas_plan.texels``
    contributes its **texel points** instead of its mesh vertices: position/normal come
    straight from the rasterized tile, and materials are resolved **per face**
    (:func:`materials.resolve_face_materials`, exact - no seam averaging) instead of
    per vertex. Every other object (absent from ``atlas_plan.texels``, or
    ``atlas_plan=None`` entirely) contributes exactly as today. ``layout`` entries gain
    a trailing ``kind`` tag (``"VERTEX"`` or ``"TEXEL"``) so callers can tell the two
    apart; with ``atlas_plan=None`` every entry is ``"VERTEX"`` and every other array is
    unchanged.
    """
    irradiance_scale = float(defaults.get("irradiance_scale", 1.0))

    verts_l: list[np.ndarray] = []
    faces_l: list[np.ndarray] = []
    irr_l: list[np.ndarray] = []
    t0_l: list[np.ndarray] = []
    alpha_l: list[np.ndarray] = []
    rho_l: list[np.ndarray] = []
    c_l: list[np.ndarray] = []
    eps_l: list[np.ndarray] = []
    bmask_l: list[np.ndarray] = []
    layout: list = []  # (name, offset, n, kind) over the surface points
    geom_by_obj: dict = {}
    offset = 0
    atlas_texels = atlas_plan.texels if atlas_plan is not None else {}

    for obj in sim_objects:
        tex = atlas_texels.get(obj.name)
        if tex is not None:
            offset = _combine_texel_object(
                obj, tex, flux_by_obj, defaults, assignment, irradiance_scale,
                verts_l, irr_l, t0_l, alpha_l, rho_l, c_l, eps_l, bmask_l, layout, offset,
            )
            continue

        geom = _extract_geometry(obj)
        if geom is None:
            continue
        verts, faces, n = geom
        geom_by_obj[obj] = geom
        mat = resolve_material(obj, defaults)

        per_vertex = None
        if assignment is not None:
            per_vertex = materials.resolve_vertex_materials(obj, assignment, mat)
            # resolve_vertex_materials walks the object's *base* mesh, so its arrays
            # are sized to len(obj.data.vertices). _combine operates on the *evaluated*
            # geometry from _extract_geometry, whose vertex count differs when the
            # object carries topology-changing modifiers (Subdivision, Array, ...).
            # When they disagree we cannot map slots to evaluated vertices, so fall
            # back to the object-level path for this object rather than crash - the
            # same shape-mismatch degradation write_frame_attributes already applies.
            if per_vertex is not None and int(per_vertex["alpha"].shape[0]) != n:
                _log.warning(
                    "[heatsim.adapter] '%s': base mesh has %d verts but evaluated geometry has %d "
                    "(topology-changing modifier); per-slot thermal materials skipped, using object-level values.",
                    obj.name, int(per_vertex["alpha"].shape[0]), n,
                )
                per_vertex = None

        flux = flux_by_obj.get(obj)
        if flux is not None and int(np.asarray(flux).reshape(-1).shape[0]) == n:
            irr = (np.asarray(flux, dtype=np.float64).reshape(-1) / _WM2_TO_WMM2) * irradiance_scale
        else:
            irr = np.zeros(n, dtype=np.float64)

        if per_vertex is not None:
            # Only Dirichlet vertices start at the fixed source temperature;
            # neighboring participants retain the ambient initial condition.
            rho = per_vertex["rho"]
            c_vec = per_vertex["c"]
            eps = per_vertex["eps"]
            dmask = per_vertex["dirichlet_mask"]
            t0 = np.where(dmask, per_vertex["t0"], mat["initial_temperature_K"])
            alpha = np.where(dmask, 0.0, per_vertex["alpha"])
            irr = np.where(dmask, 0.0, irr)
            bmask = ~dmask
        else:
            rho = np.full(n, mat["density_kg_m3"], dtype=np.float64)
            c_vec = np.full(n, mat["specific_heat_J_kgK"], dtype=np.float64)
            eps = np.full(n, mat["emissivity"], dtype=np.float64)
            if mat["thermal_role"] == "DIRICHLET_SOURCE":
                # Pinned reservoir: no diffusion, no incident flux, excluded from the
                # radiation/convection boundary (mirrors fem_adapter Dirichlet setup).
                t_dir = mat["dirichlet_temperature_K"] or mat["initial_temperature_K"]
                t0 = np.full(n, float(t_dir), dtype=np.float64)
                alpha = np.zeros(n, dtype=np.float64)
                irr = np.zeros(n, dtype=np.float64)
                bmask = np.zeros(n, dtype=bool)
            else:
                t0 = np.full(n, mat["initial_temperature_K"], dtype=np.float64)
                alpha = np.full(n, mat["thermal_diffusivity_mm2_s"], dtype=np.float64)
                bmask = np.ones(n, dtype=bool)

        verts_l.append(verts)
        faces_l.append(faces + offset)
        irr_l.append(irr)
        t0_l.append(t0)
        alpha_l.append(alpha)
        rho_l.append(rho)
        c_l.append(c_vec)
        eps_l.append(eps)
        bmask_l.append(bmask)
        layout.append((obj.name, offset, n, "VERTEX"))
        offset += n

    if not verts_l:
        return None

    faces_arr = np.vstack(faces_l).astype(np.int32) if faces_l else np.zeros((0, 3), dtype=np.int32)
    combined = SimpleNamespace(
        verts=np.vstack(verts_l),
        faces=faces_arr,
        irradiance=np.concatenate(irr_l),
        t0=np.concatenate(t0_l),
        alpha=np.concatenate(alpha_l),
        density=np.concatenate(rho_l) / _KGM3_TO_KGMM3,  # kg/m^3 -> kg/mm^3
        c=np.concatenate(c_l),
        eps=np.concatenate(eps_l),
        boundary_mask=np.concatenate(bmask_l),
        surface_count=offset,
        layout=layout,
    )

    return combined


def _run_solver(combined: SimpleNamespace, solver_cfg: dict, defaults: dict) -> np.ndarray:
    """Drive ``HeatSimFEM`` exactly like ``tests/test_heatsim_solver.py``.

    ``NUM_FRAME_DELTA = timestep_s * 60`` so ``dt = NUM_FRAME_DELTA / 60 == timestep_s``;
    ``record_time == sim_time`` records every step => history shape ``(sim_steps+1, N)``.
    """
    from visionsim.simulate.heatsim.solver import HeatSimFEM

    sim_time_s = float(solver_cfg.get("sim_time_s", 1.0))
    timestep_s = float(solver_cfg.get("timestep_s", 0.05))
    device = str(solver_cfg.get("device", "cpu"))

    gen_params = SimpleNamespace(
        device=device,
        RHO=float(defaults["density_kg_m3"]) / _KGM3_TO_KGMM3,  # kg/mm^3 fallback scalar
        C=float(defaults["specific_heat_J_kgK"]),
        K=float(defaults["thermal_diffusivity_mm2_s"]),
        NUM_FRAME_DELTA=timestep_s * 60.0,
    )
    sim_params = SimpleNamespace(
        sim_radiation=True,
        sim_convection=False,  # CONVECTION_COEFF is 0 anyway
        add_tikhonov_reg=False,
        sim_time=sim_time_s,
        record_time=sim_time_s,  # record_attimestep == 0 => record all steps
    )

    fem = HeatSimFEM(gen_params, sim_params)

    history = fem.perform_gt_heat_simulation(
        verts_np=combined.verts.copy(),
        faces_np=None,
        boundary_faces_np=None,
        boundary_verts_mask_override=combined.boundary_mask,
        u0=combined.t0,
        irradiance_map=combined.irradiance,
        thermal_diffusivity_map=combined.alpha,
        density_map=combined.density,
        specific_heat_map=combined.c,
        emissivity_map=combined.eps,
    )
    return np.asarray(history, dtype=np.float64)


def _split_history(history: np.ndarray, combined: SimpleNamespace) -> dict:
    """Trim interior points and split ``(T, N_total)`` into per-object ``(T, N)``."""
    u = history
    if u.ndim != 2 or u.shape[1] < combined.surface_count or not np.isfinite(u).all():
        raise RuntimeError(
            "Thermal solver returned an invalid field for the scene sampling layout"
        )
    out: dict = {}
    for name, off, n, _kind in combined.layout:
        out[name] = np.ascontiguousarray(u[:, off : off + n])
    return out


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def solve_scene(
    scene: Any,
    *,
    defaults: dict,
    solver_cfg: dict,
    cache_root: Path,
    assignment: Any | None = None,
    atlas_plan: AtlasPlan | None = None,
    source_digest: str | None = None,
    recompute: bool = False,
) -> dict:
    """Cache-aware FEM heat solve for ``scene``.

    A validated cache hit returns the stored per-object history. Otherwise the
    scene is sampled, Cycles irradiance is baked, and the shared point-cloud
    system is integrated before its history is split by object.

    When *atlas_plan* (from :func:`build_atlas_plan`) is supplied, its
    atlas-participating objects additionally get per-texel irradiance
    (:func:`_compute_texel_irradiance`) merged into the same ``flux_by_obj`` the
    per-vertex path already builds, and it is threaded through to :func:`_combine`'s
    TEXEL branch. ``atlas_plan=None`` uses evaluated mesh vertices throughout.

    Returns ``{obj_name: (timesteps, vertices) ndarray}``.
    """
    cache_root = Path(cache_root)
    sim_objects = gather_meshes(scene)

    blend_path = Path(str(getattr(getattr(bpy, "data", None), "filepath", "") or ""))
    key_cfg = {
        "solver": dict(solver_cfg),
        "defaults": dict(defaults),
        "objects": sorted(o.name for o in sim_objects),
        "assignments": None if assignment is None else assignment.digest,
    }
    # Atlas layout only affects the key when texel sampling is requested.
    if atlas_plan is not None:
        key_cfg["atlas"] = {
            "density": atlas_plan.density,
            "tile_min": atlas_plan.tile_min,
            "tile_max": atlas_plan.tile_max,
            "soft_max": atlas_plan.soft_max,
            "layout_digest": atlas_plan.digest,
        }
    key = cache.cache_key(blend_path, key_cfg, source_digest or "")

    expected_counts = {
        obj.name: len(atlas_plan.texels[obj.name]["xy"])
        if atlas_plan is not None and obj.name in atlas_plan.texels
        else len(obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices)
        for obj in sim_objects
    } if source_digest is not None else None
    cached = (
        cache.read_temperatures(cache_root, key, expected_counts)
        if source_digest is not None and not recompute else None
    )
    if cached is not None:
        _log.debug("[heatsim.adapter] cache hit: %s", key)
        return cached

    if not sim_objects:
        if source_digest is not None:
            cache.write_temperatures(cache_root, key, {}, {"objects": [], "timesteps": 0})
        return {}

    atlas_names = set(atlas_plan.texels) if atlas_plan is not None else set()
    vertex_objects = [o for o in sim_objects if o.name not in atlas_names]
    flux_by_obj = _compute_irradiance(scene, vertex_objects, solver_cfg, defaults)
    if atlas_plan is not None and atlas_plan.texels:
        flux_by_obj.update(_compute_texel_irradiance(scene, sim_objects, atlas_plan, solver_cfg, defaults))
    combined = _combine(sim_objects, flux_by_obj, defaults, solver_cfg, assignment=assignment, atlas_plan=atlas_plan)
    if combined is None:
        if source_digest is not None:
            cache.write_temperatures(cache_root, key, {}, {"objects": [], "timesteps": 0})
        return {}

    history = _run_solver(combined, solver_cfg, defaults)
    per_object = _split_history(history, combined)

    meta = {
        "solver_cfg": dict(solver_cfg),
        "objects": [name for name, _, _, _ in combined.layout],
        "timesteps": int(history.shape[0]) if history.ndim == 2 else 0,
        "surface_count": int(combined.surface_count),
    }
    if source_digest is not None:
        cache.write_temperatures(cache_root, key, per_object, meta)
    _log.debug("[heatsim.adapter] solved %d object(s); history %s", len(per_object), history.shape)
    return per_object


def _write_point_float_attr(mesh: Any, name: str, values: np.ndarray) -> None:
    """(Re)create a POINT/FLOAT mesh attribute from ``values``."""
    if name in mesh.attributes:
        try:
            mesh.attributes.remove(mesh.attributes[name])
        except Exception:
            logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
    attr = mesh.attributes.new(name=name, type="FLOAT", domain="POINT")
    attr.data.foreach_set("value", np.asarray(values, dtype=np.float32))


def _fallback_temperature_K(obj: Any, defaults: dict, default_T: float) -> float:
    """Object-level fallback temperature for ``write_frame_attributes``.

    A ``DIRICHLET_SOURCE`` (e.g. a topology-changing hot liquid whose
    evaluated vertex count doesn't line up with its base mesh, so it can't be
    given a per-vertex history) must still render at its reservoir
    temperature rather than ambient. Everything else (FEM participants)
    keeps the ambient ``initial_temperature_K`` default.
    """
    material = resolve_material(obj, defaults)
    if material["thermal_role"] == "DIRICHLET_SOURCE":
        return float(material["dirichlet_temperature_K"]) or float(material["initial_temperature_K"])
    return default_T


def _write_emissivity_attr(obj: Any, mesh: Any, defaults: dict, assignment: Any | None, n: int) -> None:
    """Write the ``emissivity`` POINT attribute, per-vertex from ``assignment`` when
    available (falling back to the object's single resolved material emissivity
    otherwise). Shared by the normal per-vertex write-back path and the constant-fill
    fallback (:func:`_write_constant_fill_attributes`) so both leave the SAME emissivity
    signal for the gray-body shader -- only the temperature detail differs between them.
    """
    material = resolve_material(obj, defaults)
    eps_vec = None
    if assignment is not None:
        per_vertex = materials.resolve_vertex_materials(obj, assignment, material)
        if per_vertex is not None:
            eps_vec = np.asarray(per_vertex["eps"], dtype=np.float32).reshape(-1)
    if eps_vec is None or eps_vec.shape[0] != n:
        eps_vec = np.full(n, float(material["emissivity"]), dtype=np.float32)
    _write_point_float_attr(mesh, "emissivity", eps_vec)


def _write_constant_fill_attributes(
    obj: Any, mesh: Any, defaults: dict, assignment: Any | None, fill_T: float
) -> None:
    """Constant-fill ``sim_temperature`` (and ``emissivity``) for a vertex-path object
    whose per-vertex write-back is impossible this frame (topology mismatch or missing
    history) -- called instead of leaving both attributes absent.

    An absent ``sim_temperature`` makes the ``temperature`` AOV emit 0 K (see
    ``thermal_shader._build_temperature_source_chain``'s ``is_valid = sim_temperature >
    1.0`` gate), which silently discards a real solved field down to nothing worse than
    if the object had never been simulated. A constant fill is coarser than genuine
    per-vertex detail but is never worse than 0 K or (for the shape-mismatch case)
    ambient -- see ``write_frame_attributes`` for how ``fill_T`` is chosen.
    """
    n = len(mesh.vertices)
    _write_point_float_attr(mesh, "sim_temperature", np.full(n, fill_T, dtype=np.float32))
    _write_emissivity_attr(obj, mesh, defaults, assignment, n)


def write_frame_attributes(
    scene: Any,
    history: dict,
    timestep: int,
    defaults: dict,
    assignment: Any | None = None,
    atlas_plan: AtlasPlan | None = None,
) -> None:
    """Write per-vertex temperatures for the chosen ``timestep`` (use ``-1`` for last).

    For every simulated mesh (present in ``history``) this writes a
    ``sim_temperature`` (FLOAT/POINT) attribute for that timestep plus a constant
    ``emissivity`` (FLOAT/POINT) attribute. Vertex-path objects whose per-vertex
    write-back is impossible this frame -- ``history`` has no entry for them, or its
    per-vertex count doesn't match the base mesh (a topology-changing modifier) -- still
    get a CONSTANT-fill ``sim_temperature``/``emissivity`` (never left absent: an absent
    ``sim_temperature`` makes the ``temperature`` AOV emit 0 K, see
    :func:`_write_constant_fill_attributes`), plus an OBJECT-level
    ``heatsim_default_temperature`` custom property so downstream rendering still has a
    sane fallback too: ``defaults['initial_temperature_K']`` (ambient) for FEM
    participants, or the object's own ``dirichlet_temperature_K`` reservoir temperature
    for a ``DIRICHLET_SOURCE`` (e.g. a topology-changing hot liquid whose vertex count
    can't be tracked per-frame) so it still renders hot instead of at ambient. This does
    NOT apply to atlas participants (see below) -- they deliberately get no per-vertex
    attribute at all.

    When *assignment* is supplied the ``emissivity`` attribute is resolved **per
    vertex** from the object's material slots rather than stamped as one
    object-level constant, so per-slot emissivity reaches the gray-body radiance
    shader. In LWIR that difference (polished metal ~0.05 vs painted ~0.9)
    dominates how the rendered frame looks.

    When *atlas_plan* is supplied (TEXEL render domain), objects present in
    ``atlas_plan.texels`` are atlas participants: their per-pixel signal comes from the
    rendered atlas image (:func:`write_atlas`) sampled by the shader, not from a per-vertex
    mesh attribute, so this function writes ONLY their ``heatsim_default_temperature``
    fallback (used where the atlas mix factor is 0, e.g. margin the dilation never reached)
    and skips the ``sim_temperature``/``emissivity`` point-attribute write entirely -- even
    if ``history[obj.name]``'s texel count happens to equal the mesh's vertex count.  Every
    mesh also gets an explicit ``ATLAS_COVERAGE_PROP`` OBJECT-domain float (1.0 for atlas
    participants, 0.0 otherwise); the shader multiplies this into its atlas mix factor so a
    non-participant object can never pick up stray atlas coverage through the default (0,0,0)
    vector a missing ``HeatSim_Atlas_UV`` attribute produces. Non-atlas objects are written
    exactly as when ``atlas_plan=None``.
    """
    default_T = float(defaults["initial_temperature_K"])
    atlas_names = set(atlas_plan.texels) if atlas_plan is not None else set()
    for obj in scene.objects:
        if getattr(obj, "type", None) != "MESH":
            continue
        mesh = getattr(obj, "data", None)
        if mesh is None:
            continue

        obj[ATLAS_COVERAGE_PROP] = 1.0 if obj.name in atlas_names else 0.0

        if obj.name in atlas_names:
            # TEXEL: this object's field lives in the atlas image, not on its mesh --
            # only the fallback (used wherever the atlas mix factor is 0) is written.
            # Remove any stale sim_temperature/emissivity POINT attributes left by a
            # prior VERTEX-mode run on this same mesh (long-lived RPyC service can
            # switch modes between prepare_thermal calls): the shader's vertex path
            # reads sim_temperature whenever it's present and > 1.0, so a leftover
            # attribute would bleed stale per-vertex temperatures through atlas holes
            # (mix factor 0) instead of the fresh fallback written below.
            for attr_name in ("sim_temperature", "emissivity"):
                if attr_name in mesh.attributes:
                    mesh.attributes.remove(mesh.attributes[attr_name])
            obj["heatsim_default_temperature"] = _fallback_temperature_K(obj, defaults, default_T)
            continue

        if atlas_plan is None and ATLAS_COVERAGE_PROP in obj:
            # VERTEX mode: clear a stale coverage gate left by a prior TEXEL-mode run
            # on this same object (long-lived RPyC service can switch modes between
            # prepare_thermal calls). Otherwise the shader's atlas mix factor stays
            # gated open by a stale alpha * stale coverage=1.0, letting a stale packed
            # atlas image win over the fresh per-vertex sim_temperature written below.
            del obj[ATLAS_COVERAGE_PROP]

        hist = history.get(obj.name)
        if hist is None:
            # No solve history at all for this object (e.g. a DIRICHLET_SOURCE whose
            # topology changes every frame, so it was never given a per-vertex field).
            # Still constant-fill sim_temperature/emissivity -- an absent sim_temperature
            # makes the temperature AOV emit 0 K instead of this object's reservoir/ambient
            # fallback (see _write_constant_fill_attributes).
            fallback_T = _fallback_temperature_K(obj, defaults, default_T)
            obj["heatsim_default_temperature"] = fallback_T
            _log.warning(
                "[heatsim.adapter] '%s': no solve history for this object; sim_temperature "
                "constant-filled at %.2f K instead of left absent.", obj.name, fallback_T,
            )
            _write_constant_fill_attributes(obj, mesh, defaults, assignment, fallback_T)
            continue

        arr = np.asarray(hist)
        n = len(mesh.vertices)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != n or not np.isfinite(arr).all():
            raise RuntimeError(f"{obj.name!r}: solved vertex field cannot be written to the base mesh")
        if bpy is not None:
            geometry = _extract_geometry(obj)
            if geometry is None or not _vertex_writeback_matches(obj, geometry[0]):
                raise RuntimeError(f"{obj.name!r}: evaluated vertices cannot be safely written to the base mesh")

        row = np.asarray(arr[timestep], dtype=np.float32).reshape(-1)
        _write_point_float_attr(mesh, "sim_temperature", row)
        _write_emissivity_attr(obj, mesh, defaults, assignment, n)
        mesh.update()
