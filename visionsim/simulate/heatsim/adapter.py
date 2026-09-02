"""Scene adapter: Blender scene -> solver inputs -> per-vertex temperatures.

Hand-written glue that distils the data-shaping core of heat-sim-blender's
``addon/lib/fem_adapter.py`` (@ 543ee81) into four functions:

* :func:`gather_meshes`         - select the meshes that participate in the solve.
* :func:`resolve_material`      - per-object thermal params (``heat_sim_material`` -> ``defaults``).
* :func:`solve_scene`           - cache-aware geometry -> irradiance -> FEM solve -> per-object history.
* :func:`write_frame_attributes`- stamp ``sim_temperature`` / ``emissivity`` (and a fallback temp).

Unit / sign / dt conventions are preserved verbatim from upstream because the
wrong units silently corrupt the physics:

* geometry in **millimetres** (world coords x 1000),
* irradiance W/m^2 -> W/mm^2 (/1e6) then x ``irradiance_scale``,
* density kg/m^3 -> kg/mm^3 (/1e9) wherever the solver consumes it,
* the FEM solver is driven exactly as ``tests/test_heatsim_solver.py`` drives it
  (``NUM_FRAME_DELTA = timestep_s * 60`` so ``dt = NUM_FRAME_DELTA / 60``;
  ``record_time == sim_time`` records every step).

The module imports ``bpy`` defensively so it stays importable (for
linting / type-checking) outside Blender; the bpy-coupled solver and Direct-Kernel
irradiance modules are imported lazily inside :func:`solve_scene`.
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

from visionsim.simulate.heatsim import atlas, cache, materials
from visionsim.simulate.heatsim.constants import (
    ATLAS_COVERAGE_PROP,
    ATLAS_UV_LAYER_NAME,
    BAKE_UV_LAYER_NAME,
    CYCLES_LOUT_TO_IRRADIANCE,
)

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
    """Return the MESH objects that take part in the heat solve.

    A mesh participates when it is visible, renderable and (per-object) heat
    simulation is enabled. Mirrors the filter at ``fem_adapter.py:2867,2875``;
    a missing ``heat_simulation_enabled`` attribute defaults to ``True``.

    Also un-shares each returned object's mesh datablock (see
    :func:`_ensure_single_user_meshes`). This is the single choke point every
    solve/atlas entry point pulls its object list from (:func:`solve_scene`,
    :func:`solve_scene_animated`, and the adapter's TEXEL atlas-plan caller), so doing
    it here guarantees it happens once, early, before any per-vertex attribute or atlas
    UV layer is ever written for these objects - regardless of which of those three
    paths runs first in a given ``prepare_thermal`` call.
    """
    out: list = []
    for obj in scene.objects:
        if getattr(obj, "type", None) != "MESH":
            continue
        if not obj.visible_get() or obj.hide_render:
            continue
        if not bool(getattr(obj, "heat_simulation_enabled", True)):
            continue
        out.append(obj)
    _ensure_single_user_meshes(out)
    return out


def _ensure_single_user_meshes(sim_objects: list) -> None:
    """Give every object in ``sim_objects`` its own single-user mesh datablock.

    Linked duplicates (multiple objects pointing at the same ``Mesh`` datablock, e.g.
    Blender's Alt-D) share per-vertex attributes AND UV layers on that one datablock, so
    writing ``sim_temperature``/the atlas UV layer for one object silently overwrites --
    or is overwritten by -- every other object sharing it (last write wins). Copying the
    mesh (``obj.data = obj.data.copy()``) whenever ``obj.data.users > 1`` makes each
    object's write target independent.

    Idempotent: once an object's mesh is single-user, ``users`` stays 1 on every later
    call (e.g. repeated ``prepare_thermal`` calls on a long-lived RPyC service), so a
    second pass over the same objects is a no-op.
    """
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
    except Exception:  # pragma: no cover - defensive
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


def read_authored_irradiance_scale(scene: Any) -> float | None:
    """Return the heat-sim addon's authored scene-level ``irradiance_scale``.

    The addon stores it under ``scene.heat_sim_settings.irradiance_scale``.
    VisionSim does not register that scene-level PropertyGroup, so the value is
    read from the raw ID-property (``scene.get("heat_sim_settings")`` returns an
    ``IDPropertyGroup``). Returns ``None`` when the blend has no authored
    heat-sim scene settings, so the caller keeps its own default.
    """
    try:
        raw = scene.get("heat_sim_settings")
        if raw is None:
            return None
        val = raw.get("irradiance_scale")
        return None if val is None else float(val)
    except Exception:
        return None


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

    Spans the **final-timestep** temperatures of every solved object (M1 renders
    the final state on every frame), so the thermal preview colormap covers the
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


def global_temperature_range_animated(history: dict[str, Any], default_K: float) -> tuple[float, float]:
    """Global colormap range ``(tmin, tmax)`` in Kelvin spanning EVERY frame of an animated solve.

    Like :func:`global_temperature_range`, but folds in the full per-frame history instead
    of only the final timestep. The animated (M2) field keeps evolving frame to frame, so a
    final-frame-only range would clip the preview colormap against later (typically hotter)
    frames -- this instead gives a single, stable scale for the whole sequence.

    Args:
        history: ``{obj_name: (n_frames, n_verts) array}`` from :func:`solve_scene_animated`.
        default_K: Default initial temperature stamped on unsolved meshes.

    Returns:
        ``(tmin, tmax)`` with ``tmax - tmin >= 1.0``.
    """
    arrays = [np.asarray(arr, dtype=float) for arr in history.values() if np.asarray(arr).size]
    if not arrays:
        return float(default_K), float(default_K) + 1.0
    tmin = min(float(np.min(a)) for a in arrays)
    tmax = max(float(np.max(a)) for a in arrays)
    tmin = min(tmin, float(default_K))
    if tmax - tmin < 1.0:
        tmax = tmin + 1.0
    return tmin, tmax


# ---------------------------------------------------------------------------
# Geometry extraction (evaluated mesh -> world mm + triangulated faces)
# ---------------------------------------------------------------------------


def _extract_geometry(obj: Any) -> tuple | None:
    """``(verts_mm (N,3) float64, faces (M,3) int32, n_verts)`` for the evaluated
    mesh, or ``None`` if it has no geometry. Verts are world-space x 1000 (mm)
    and quads are triangulated, matching ``fem_adapter._extract_mesh_data``.

    The evaluated mesh's vertex order/count matches the Direct-Kernel irradiance
    extraction (it uses the same ``foreach_get('co')`` path), so the returned
    flux aligns index-for-index with these vertices.
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
    added) - :func:`write_atlas` (Task 3) uses it together with ``layout.tiles[name].offset``
    to scatter this object's solved texel temperatures into the shared atlas image. Only
    objects that made it all the way through selection, UV prep and rasterization appear
    here; every other sim object (excluded by :func:`atlas.select_for_atlas`, or demoted
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
    """Lazy-import wrapper around ``irradiance.prepare_object_bake_uv``.

    Keeps this module importable without ``bpy`` (mirrors :func:`_compute_irradiance`'s
    lazy import of ``irradiance_kernel``); tests monkeypatch this function directly.
    """
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
    on ``obj``'s BASE mesh (per spec ``4.1`` - the thermal AOV shader samples the atlas
    image through this layer at render time). It must land on the base mesh, not a
    to_mesh() copy, so the modifier stack propagates it (Bevel interpolates named UV
    layers onto new geometry, EdgeSplit duplicates them, most Geometry Nodes setups
    preserve them) - :func:`build_atlas_plan` then forces a depsgraph update and reads
    it back via :func:`_extract_evaluated_face_uv_and_slots` to rasterize from geometry
    that matches what the solver actually used. Best-effort: never raises, since a
    failure here must not abort the solve - it degrades to the per-vertex path via the
    caller's own "evaluated mesh lacks the atlas UV layer" check.
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
    except Exception as exc:  # pragma: no cover - defensive, mirrors irradiance.py's style
        _log.warning("[heatsim.adapter] '%s': failed to write %s: %s", obj.name, ATLAS_UV_LAYER_NAME, exc)


def build_atlas_plan(scene: Any, sim_objects: list, cfg: dict) -> AtlasPlan:
    """Select, size and rasterize the thermal atlas for ``sim_objects``.

    Per §4.1-4.2 of the design spec: an object joins the atlas iff its native vertex
    density is below ``cfg['atlas_texel_density']`` (:func:`atlas.select_for_atlas`);
    joining objects get a tile sized by surface area (:func:`atlas.allocate`) and
    rasterized into texel sample points (:func:`atlas.rasterize_tile`) using the
    object's bake UV (created/unwrapped by the existing hardened
    ``irradiance.prepare_object_bake_uv`` machinery).

    Rasterization is fully EVALUATED-mesh based: the atlas UV layer is written to the
    BASE mesh (so Bevel/EdgeSplit/Geometry-Nodes modifier stacks propagate it), the
    depsgraph is forced to re-evaluate, and both the triangle geometry (from
    :func:`_extract_geometry`) and the UVs/material indices used to rasterize (from
    :func:`_extract_evaluated_face_uv_and_slots`) are read from that SAME evaluated
    mesh - so a mismatch between the base mesh's vertex count and the evaluated
    geometry's (Bevel, Geometry Nodes, EdgeSplit, ...) is no longer meaningful and no
    longer demotes the object. A temperature atlas is UV-addressed, not
    vertex-addressed, so it does not care how many vertices the base mesh has.

    An object is demoted to the vertex path (excluded from the returned plan's
    ``texels``, with a warning - never raised) when: its bake-UV prep fails to produce
    a usable UV layer, the modifier stack drops the atlas UV layer entirely (some
    Geometry Nodes setups do), or its UV triangulation doesn't match the evaluated
    geometry's triangle count. Every check degrades gracefully; this function never
    raises for a per-object failure.
    """
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
        # `n` is the EVALUATED vertex count `_extract_geometry` just read; the per-vertex
        # write-back path (`write_frame_attributes`) can only write onto the BASE mesh
        # (`obj.data.vertices`). When a topology-changing modifier (Bevel, Subdivision,
        # Geometry Nodes, ...) makes those counts differ, write-back is structurally
        # impossible regardless of how dense the object already looks, so the density
        # rule doesn't apply - force atlas participation instead of silently discarding
        # the solved field later.
        writeback_possible = len(obj.data.vertices) == n
        if atlas.select_for_atlas(n, area_m2, density, writeback_possible=writeback_possible):
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
            continue

        # Write the atlas UV layer onto the BASE mesh first (so Bevel/EdgeSplit/GN
        # modifiers propagate it) and force a depsgraph update, then read triangles'
        # UVs + material indices back from the EVALUATED mesh - the same mesh
        # `verts`/`faces` above came from - so face indices line up 1:1 regardless of
        # whether the base and evaluated vertex counts agree.
        _prepare_bake_uv(obj)
        _write_atlas_uv_layer(obj, tile, layout.atlas_size, BAKE_UV_LAYER_NAME)
        uv_result = _extract_evaluated_face_uv_and_slots(obj, ATLAS_UV_LAYER_NAME)
        if uv_result is None:
            _log.warning(
                "[heatsim.adapter] '%s': %s unavailable on the evaluated mesh (bake-UV unwrap "
                "failed, or the modifier stack dropped the named layer); demoted from the atlas "
                "to the per-vertex path.", name, ATLAS_UV_LAYER_NAME,
            )
            continue
        atlas_loop_uv, face_material_index = uv_result
        if atlas_loop_uv.shape[0] != faces.shape[0]:
            _log.warning(
                "[heatsim.adapter] '%s': UV triangulation (%d tris) does not match evaluated "
                "geometry (%d tris); demoted from the atlas to the per-vertex path.",
                name, atlas_loop_uv.shape[0], faces.shape[0],
            )
            continue

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
            _log.warning(
                "[heatsim.adapter] '%s': atlas tile rasterized zero texels; "
                "demoted from the atlas to the per-vertex path.", name,
            )
            continue

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


# Push-out margin dilation, in texels. Invariant: 2 * iterations <= the packing
# `padding` (see build_atlas_plan's atlas.allocate(..., padding=_ATLAS_PACKING_PADDING)
# call). Each dilate pass grows a tile's valid region by at most 1px, and TWO adjacent
# tiles dilate toward each other from both sides of the gap -- so the gap must be wide
# enough for both expansions with no meeting point, or the middle gap texels would be
# filled with the MEAN of two unrelated objects' temperatures. The assert below enforces
# it; see the design spec's "seam bleeding" risk note.
#
# One iteration is not enough. A single pass protects only a 1-texel ring, but the atlas
# also contains INTERIOR holes -- texels inside a tile that no solved element scattered
# into. Measured on visionsim50/kitchen1: 25 invalid components inside/near tile interiors
# sized 1-462 texels, far beyond a one-texel margin. Render-time bilinear filtering then
# straddles those valid/invalid edges and drags T_effective under the solve floor (2665
# sub-295 K pixels in a single frame; they disappear under render_domain=VERTEX).
#
# 8 passes fill a hole of radius <= 8 texels outright, and the shader thresholds the atlas
# alpha so anything still unfilled falls back to the object-level default rather than
# blending a half-zeroed colour (see thermal_shader._build_temperature_source_chain).
# Padding grows in lockstep to keep neighbouring tiles from meeting; the cost is a modest
# increase in packed atlas area.
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


def write_atlas(history: dict, atlas_plan: AtlasPlan, cache_root: Path) -> Path:
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
    out_dir = cache_root / f"atlas_{atlas_plan.digest or 'noatlas'}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "atlas_temperature.exr"

    width, height = atlas_plan.layout.atlas_size
    width, height = max(int(width), 1), max(int(height), 1)
    if atlas_plan.layout.atlas_size == (0, 0) or not atlas_plan.texels:
        temp_dilated: np.ndarray = np.zeros((height, width), dtype=np.float64)
        alpha: np.ndarray = np.zeros((height, width), dtype=np.float64)
    else:
        temp_dilated, alpha = _scatter_atlas_arrays(history, atlas_plan)

    rgba: np.ndarray = np.zeros((height, width, 4), dtype=np.float32)
    rgba[..., 0] = temp_dilated
    rgba[..., 1] = temp_dilated
    rgba[..., 2] = temp_dilated
    rgba[..., 3] = alpha

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
    try:
        image.colorspace_settings.name = "Non-Color"
    except Exception:  # pragma: no cover - defensive, mirrors irradiance.py's style
        pass
    image.pixels.foreach_set(rgba.ravel())
    image.filepath_raw = str(out_path)
    image.file_format = "OPEN_EXR"
    image.save()
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
    """Bilinear-sample ``obj``'s Cycles albedo bake at texel UV centers.

    Reuses ``irradiance.bake_albedo_map`` - the same bake primitive
    ``irradiance_kernel.get_or_bake_vertex_albedo`` reduces to per-vertex values - but
    samples its ``(H, W, 3)`` pixel grid directly instead of averaging it down, so the
    texel gets its own native-resolution albedo. On any bake failure every texel gets
    albedo=0 (full absorption), the same fallback the per-vertex kernel path uses for a
    missing bake.
    """
    k = int(uv_at_texel.shape[0])
    if k == 0:
        return np.zeros(0, dtype=np.float64)
    try:
        from visionsim.simulate.heatsim import irradiance

        baked = irradiance.bake_albedo_map(scene, obj, texture_size)
    except Exception as exc:  # pragma: no cover - defensive, mirrors irradiance.py's style
        _log.warning("[heatsim.adapter] '%s': albedo bake failed for texel sampling: %s", obj.name, exc)
        baked = None
    if baked is None or getattr(baked, "pixels", None) is None:
        return np.zeros(k, dtype=np.float64)
    luma = _sample_bilinear(baked.pixels, int(baked.width), int(baked.height), uv_at_texel)
    return np.clip(luma, 0.0, 1.0)


def _texel_irradiance_cycles(
    scene: Any, obj: Any, uv_at_texel: np.ndarray, texture_size: int, samples: int | None = None
) -> np.ndarray:
    """Bilinear-sample ``obj``'s Cycles irradiance bake at texel UV centers.

    The Cycles counterpart to :func:`_texel_albedo`, sampling the same UVs from the
    DIFFUSE DIRECT+INDIRECT bake instead of the COLOR bake. Returns *incident* W/m^2;
    the caller applies (1 - albedo) to get absorbed flux.

    On bake failure every texel gets 0 W/m^2 - the surface then simply holds its
    initial temperature, which is the same failure mode as a missing Direct-Kernel
    contribution and is preferable to inventing flux.
    """
    k = int(uv_at_texel.shape[0])
    if k == 0:
        return np.zeros(0, dtype=np.float64)
    try:
        from visionsim.simulate.heatsim import irradiance

        baked = irradiance.bake_irradiance_map(scene, obj, texture_size, samples=samples)
    except Exception as exc:  # pragma: no cover - defensive, mirrors irradiance.py's style
        _log.warning("[heatsim.adapter] '%s': irradiance bake failed for texel sampling: %s", obj.name, exc)
        baked = None
    if baked is None or getattr(baked, "pixels", None) is None:
        return np.zeros(k, dtype=np.float64)
    lum = _sample_bilinear(baked.pixels, int(baked.width), int(baked.height), uv_at_texel)
    return np.maximum(np.asarray(lum, dtype=np.float64) * CYCLES_LOUT_TO_IRRADIANCE, 0.0)


def _compute_irradiance_cycles(scene: Any, sim_objects: list, solver_cfg: dict, defaults: dict) -> dict:
    """Per-vertex irradiance from a Cycles bake -> ``{obj: (N,) float64 W/m^2 absorbed}``.

    Drop-in alternative to :func:`_compute_irradiance` (the analytic Direct Kernel),
    selected by ``solver_cfg["irradiance_source"] == "CYCLES_BAKE"``.

    The Direct Kernel counts only ``LIGHT`` objects plus an unshadowed SH9 sky and
    models no indirect bounce, so a scene lit by emissive geometry gets no flux at all
    (visionsim50/classroom: 6.17 m^2 of emissive windows contributing exactly zero).
    Cycles resolves emissive meshes, bounce and portals because it is a path tracer.

    The bake yields *incident* irradiance, so (1 - albedo) is applied here to match the
    Direct Kernel's absorbed-flux contract. Albedo comes from the bake that already runs
    for the kernel path, so no extra Cycles work is introduced by the conversion.
    """
    from visionsim.simulate.heatsim import irradiance, irradiance_kernel

    texture_size = int(solver_cfg.get("irradiance_texture_size", 512))
    bake_samples = int(solver_cfg.get("bake_samples", 1024))
    out: dict = {}
    for obj in sim_objects:
        if resolve_material(obj, defaults)["thermal_role"] == "DIRICHLET_SOURCE":
            continue  # mirrors compute_per_vertex_irradiance's own Dirichlet skip
        try:
            baked = irradiance.bake_irradiance_map(scene, obj, texture_size, samples=bake_samples)
        except Exception as exc:  # pragma: no cover - defensive
            _log.warning("[heatsim.adapter] '%s': irradiance bake failed: %s", obj.name, exc)
            baked = None
        if baked is None or getattr(baked, "vertex_flux", None) is None:
            continue
        incident = np.asarray(baked.vertex_flux, dtype=np.float64).reshape(-1)
        # `get_or_bake_vertex_albedo` takes a LIST and returns {name: (N,) array}; its
        # `texture_size` is keyword-only. Getting either wrong raises TypeError, and a
        # bare `except` here would turn that into albedo=0 -- i.e. full absorption, up to
        # ~4x the correct flux on a light surface -- with no way to notice. Both
        # degradation paths below therefore log.
        albedo = None
        try:
            albedo_by_name = irradiance_kernel.get_or_bake_vertex_albedo(
                scene, [obj], texture_size=texture_size
            )
            raw = albedo_by_name.get(obj.name)
            if raw is not None:
                albedo = np.clip(np.asarray(raw, dtype=np.float64).reshape(-1), 0.0, 1.0)
        except Exception as exc:  # pragma: no cover - defensive
            _log.warning(
                "[heatsim.adapter] '%s': vertex albedo bake failed (%s); assuming full "
                "absorption, which OVERESTIMATES absorbed flux.", obj.name, exc,
            )
        if albedo is None:
            albedo = np.zeros_like(incident)
        elif albedo.shape != incident.shape:
            _log.warning(
                "[heatsim.adapter] '%s': albedo has %d entries but irradiance has %d "
                "(the two bakes disagree about which mesh they sampled); assuming full "
                "absorption, which OVERESTIMATES absorbed flux.",
                obj.name, albedo.shape[0], incident.shape[0],
            )
            albedo = np.zeros_like(incident)
        out[obj] = np.maximum(incident * (1.0 - albedo), 0.0)
    return out


def _compute_texel_irradiance(
    scene: Any, sim_objects: list, atlas_plan: AtlasPlan, solver_cfg: dict, defaults: dict
) -> dict:
    """Per-texel Direct-Kernel irradiance for every atlas-participating object.

    Returns ``{obj: (K,) float64 W/m^2 absorbed}``, keyed the same way as
    :func:`_compute_irradiance`'s return value so both can feed the same
    ``flux_by_obj`` dict into :func:`_combine`.

    Builds the scene BVH backend exactly **once** per call (not once per atlas
    object) and reuses it across every ``compute_irradiance_at_points`` call below
    via that function's optional ``backend`` parameter -- with N atlas objects this
    was previously N full scene BVH rebuilds. Also threads the same
    ``direct_kernel_soft_shadow_rays`` knob :func:`_compute_irradiance` reads from
    ``solver_cfg`` through to the texel kernel, so both irradiance paths use the
    same shadow-ray sample count instead of the kernel's hardcoded default.
    """
    from visionsim.simulate.heatsim import bvh_backend, irradiance_kernel

    texture_size = int(solver_cfg.get("irradiance_texture_size", 512))
    n_samples_for_area = int(solver_cfg.get("direct_kernel_soft_shadow_rays", 8))
    use_cycles = str(solver_cfg.get("irradiance_source", "DIRECT_KERNEL")).upper() == "CYCLES_BAKE"
    bake_samples = int(solver_cfg.get("bake_samples", 1024))
    by_name = {o.name: o for o in sim_objects}

    # Only the Direct Kernel traces shadow rays; building a whole-scene BVH for the
    # Cycles path costs a full build that is then never queried. Still built exactly
    # once (pre-loop), not per object.
    backend = None
    if not use_cycles:
        backend = bvh_backend.best_available()
        backend.build_for_meshes(irradiance_kernel._collect_scene_meshes_world(scene))

    out: dict = {}
    for name, tex in atlas_plan.texels.items():
        obj = by_name.get(name)
        if obj is None:
            continue
        if resolve_material(obj, defaults)["thermal_role"] == "DIRICHLET_SOURCE":
            continue  # mirrors compute_per_vertex_irradiance's own Dirichlet skip
        positions = np.asarray(tex["position_mm"], dtype=np.float64)
        normals = np.asarray(tex["normal"], dtype=np.float64)
        uv = np.asarray(tex["uv"], dtype=np.float64)
        albedo = _texel_albedo(scene, obj, uv, texture_size)
        if use_cycles:
            # Cycles bake gives INCIDENT irradiance; apply (1 - albedo) to match the
            # Direct Kernel's absorbed-flux contract.
            flux = _texel_irradiance_cycles(scene, obj, uv, texture_size, samples=bake_samples) * (1.0 - albedo)
        else:
            flux = irradiance_kernel.compute_irradiance_at_points(
                scene, positions, normals, albedo, backend=backend, n_samples_for_area=n_samples_for_area
            )
        out[obj] = np.asarray(flux, dtype=np.float64).reshape(-1)
    return out


# ---------------------------------------------------------------------------
# Direct-Kernel irradiance
# ---------------------------------------------------------------------------


def _compute_irradiance(scene: Any, sim_objects: list, solver_cfg: dict, defaults: dict) -> dict:
    """Run the Direct-Kernel and return ``{obj: (N,) float64 W/m^2 absorbed}``."""
    if str(solver_cfg.get("irradiance_source", "DIRECT_KERNEL")).upper() == "CYCLES_BAKE":
        return _compute_irradiance_cycles(scene, sim_objects, solver_cfg, defaults)

    from visionsim.simulate.heatsim import irradiance_kernel

    # SimpleNamespace surrogate for the addon's scene PropertyGroup (the kernel
    # only reads these via getattr(..., default)). Sky occlusion defaults off so
    # the kernel takes the unoccluded SH9 sky path (no per-vertex AO bake).
    settings = SimpleNamespace(
        irradiance_texture_size=int(solver_cfg.get("irradiance_texture_size", 512)),
        enable_sky_occlusion=bool(solver_cfg.get("enable_sky_occlusion", False)),
        sky_ao_min_for_bent=float(solver_cfg.get("sky_ao_min_for_bent", 0.02)),
        direct_kernel_soft_shadow_rays=int(solver_cfg.get("direct_kernel_soft_shadow_rays", 8)),
    )
    raw = irradiance_kernel.compute_per_vertex_irradiance(scene, list(sim_objects), settings)
    out: dict = {}
    for obj, payload in raw.items():
        flux = payload.get("vertex_flux") if isinstance(payload, dict) else payload
        if flux is not None:
            out[obj] = np.asarray(flux, dtype=np.float64).reshape(-1)
    return out


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
    byte-identical to before this parameter existed.
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
            # Per-slot resolution: every field is already (N,). Dirichlet vertices
            # are pinned individually - alpha=0, no incident flux, excluded from the
            # radiation/convection boundary - exactly what the object-level branch
            # below does, just per vertex. T0 is the simulation's *initial condition*,
            # not a constitutive property: a FEM-participant vertex starts at ambient
            # even where it seams against a Dirichlet slot (materials.resolve_vertex_materials
            # area-weights T0 like any other continuous field, which would otherwise
            # pre-seed seam vertices with a slice of the reservoir's temperature before
            # the solver ever runs a step -- mirrors the object-level branch below,
            # which always starts a non-Dirichlet object at initial_temperature_K).
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
    domain = str(solver_cfg.get("domain", "POINTS")).upper()
    backend = str(solver_cfg.get("laplacian_backend", "ROBUST")).upper()
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

    fem = HeatSimFEM(
        gen_params,
        sim_params,
        laplacian_domain=domain,
        laplacian_backend=backend,
    )

    is_points = domain == "POINTS"
    history = fem.perform_gt_heat_simulation(
        verts_np=combined.verts.copy(),
        faces_np=None if is_points else combined.faces.copy(),
        boundary_faces_np=None if is_points else combined.faces.copy(),
        # NOTE: boundary_verts_mask_override is honoured by the solver in POINTS
        # mode only.  In MESH mode the boundary is derived from face topology and
        # the per-vertex override is silently ignored — Dirichlet sources are not
        # correctly excluded in MESH mode (known follow-up).
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
    # Guard: in MESH mode the solver compresses the vertex array to only
    # face-referenced vertices, so the column count can be LESS than
    # surface_count.  Slicing per-object offsets into a shorter array would
    # silently return wrong temperatures.  Raise loudly instead.
    if u.ndim == 2 and u.shape[1] < combined.surface_count:
        raise RuntimeError(
            f"_split_history: solver returned {u.shape[1]} columns but combined "
            f"geometry has {combined.surface_count} surface vertices.  This happens "
            f"in MESH mode when the mesh contains orphan/unreferenced vertices that "
            f"the solver drops.  Use POINTS domain (the supported M1 path) or ensure "
            f"every vertex is referenced by at least one face."
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
) -> dict:
    """Cache-aware FEM heat solve for ``scene``.

    On a cache hit the stored per-object ``(timesteps, vertices)`` history is
    returned untouched. On a miss: gather meshes -> Direct-Kernel irradiance
    (W/m^2 -> W/mm^2) -> combine -> ``HeatSimFEM.perform_gt_heat_simulation`` ->
    split the combined history back per object -> write the cache.

    When *atlas_plan* (from :func:`build_atlas_plan`) is supplied, its
    atlas-participating objects additionally get per-texel irradiance
    (:func:`_compute_texel_irradiance`) merged into the same ``flux_by_obj`` the
    per-vertex path already builds, and it is threaded through to :func:`_combine`'s
    TEXEL branch. ``atlas_plan=None`` (the default) reproduces today's behaviour
    exactly, including the cache key.

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
    # The "atlas" key is entirely ABSENT (not present-with-value-None) when no atlas is
    # in play, so key_cfg's JSON -- and therefore the SHA1 cache key -- is byte-identical
    # to before the atlas feature existed. Adding an "atlas": None entry here would still
    # change the JSON (and thus every existing .heatsim cache's key) even though nothing
    # about the solve changed; see solve_scene's docstring guarantee below.
    if atlas_plan is not None:
        key_cfg["atlas"] = {
            "density": atlas_plan.density,
            "tile_min": atlas_plan.tile_min,
            "tile_max": atlas_plan.tile_max,
            "soft_max": atlas_plan.soft_max,
            "layout_digest": atlas_plan.digest,
        }
    key = cache.cache_key(blend_path, key_cfg)

    cached = cache.read_temperatures(cache_root, key)
    if cached is not None:
        _log.debug("[heatsim.adapter] cache hit: %s", key)
        return cached

    if not sim_objects:
        cache.write_temperatures(cache_root, key, {}, {"objects": []})
        return {}

    atlas_names = set(atlas_plan.texels) if atlas_plan is not None else set()
    vertex_objects = [o for o in sim_objects if o.name not in atlas_names]
    flux_by_obj = _compute_irradiance(scene, vertex_objects, solver_cfg, defaults)
    if atlas_plan is not None and atlas_plan.texels:
        flux_by_obj.update(_compute_texel_irradiance(scene, sim_objects, atlas_plan, solver_cfg, defaults))
    combined = _combine(sim_objects, flux_by_obj, defaults, solver_cfg, assignment=assignment, atlas_plan=atlas_plan)
    if combined is None:
        cache.write_temperatures(cache_root, key, {}, {"objects": []})
        return {}

    history = _run_solver(combined, solver_cfg, defaults)
    per_object = _split_history(history, combined)

    meta = {
        "solver_cfg": dict(solver_cfg),
        "objects": [name for name, _, _, _ in combined.layout],
        "timesteps": int(history.shape[0]) if history.ndim == 2 else 0,
        "surface_count": int(combined.surface_count),
    }
    cache.write_temperatures(cache_root, key, per_object, meta)
    _log.debug("[heatsim.adapter] solved %d object(s); history %s", len(per_object), history.shape)
    return per_object


def _scene_fps(scene: Any) -> float:
    """``scene.render.fps / scene.render.fps_base``, guarded against a zero base."""
    r = scene.render
    fps = float(getattr(r, "fps", 24) or 24)
    fps_base = float(getattr(r, "fps_base", 1.0) or 1.0)
    if fps_base <= 0.0:
        fps_base = 1.0
    return fps / fps_base


def _order_fem_first(sim_objects: list, defaults: dict) -> tuple[list, set]:
    """Split ``sim_objects`` into FEM-participant-first, Dirichlet-source-last order.

    A Dirichlet source's evaluated vertex count may change frame to frame (e.g. a
    regenerated fluid mesh). Keeping it last in the combine order means its resize
    never shifts the ``_combine`` offsets of any FEM-participant object, so the
    FEM-participant surface prefix has a stable width across frames -- exactly the
    property :func:`solve_scene_animated` relies on to carry ``u_prev`` forward by
    index.
    """
    fem: list = []
    dirichlet: list = []
    dirichlet_names: set = set()
    for obj in sim_objects:
        role = resolve_material(obj, defaults)["thermal_role"]
        if role == "DIRICHLET_SOURCE":
            dirichlet.append(obj)
            dirichlet_names.add(obj.name)
        else:
            fem.append(obj)
    return fem + dirichlet, dirichlet_names


def _slot_level_dirichlet_mismatches(sim_objects: list, assignment: Any, defaults: dict) -> list:
    """Objects with a **slot-level** ``DIRICHLET_SOURCE`` assignment their own
    ``heat_sim_material.thermal_role`` does not already declare.

    :func:`_order_fem_first` (and the per-substep re-pin loop in
    :func:`solve_scene_animated`) only ever look at the object-level role, so a
    material slot the sidecar assigns ``DIRICHLET_SOURCE`` is never re-pinned
    between substeps in animated mode even though :func:`_combine` correctly
    zeroes its ``alpha``/``irradiance`` and clears its ``boundary_mask`` for the
    static solve. This is a detector for that gap, used to warn the caller
    rather than silently drifting -- see :func:`solve_scene_animated`.
    """
    names: list = []
    for obj in sim_objects:
        if resolve_material(obj, defaults)["thermal_role"] == "DIRICHLET_SOURCE":
            continue  # already re-pinned wholesale as an object-level Dirichlet source
        for slot in getattr(obj, "material_slots", []):
            material = getattr(slot, "material", None)
            if material is None:
                continue
            entry = assignment.entry_for(str(getattr(material, "name", "")))
            if entry is not None and entry.role == "DIRICHLET_SOURCE":
                names.append(obj.name)
                break
    return names


def solve_scene_animated(
    scene: Any,
    *,
    defaults: dict,
    solver_cfg: dict,
    cache_root: Path,
    frame_start: int,
    frame_end: int,
    every_n: int,
    substeps_per_frame: int,
    assignment: Any | None = None,
) -> tuple[dict, list]:
    """Transient per-frame FEM solve over an animated scene (Phase 1 / M2).

    ``FEM_PARTICIPANT`` objects (stable topology) evolve: ``u_prev`` carries the
    previous frame's final state forward by vertex index. ``DIRICHLET_SOURCE``
    objects (topology may change frame to frame, e.g. a regenerated fluid mesh)
    are a constant-temperature reservoir -- re-extracted every frame purely for
    *position* (so they couple heat into nearby FEM-participant vertices through
    the POINTS kNN Laplacian; see ``solver._build_matrices``, which builds that
    Laplacian over ALL combined points regardless of source object) and reset to
    the reservoir temperature every frame. Their own field never evolves and is
    not recorded.

    On a vertex-count change of any Dirichlet source, the combined system is
    rebuilt from scratch for that frame (via :func:`_combine`); the
    FEM-participant prefix of ``u_prev`` is preserved unchanged and the Dirichlet
    slice is refilled with its (possibly keyframed) reservoir temperature. A
    vertex-count change on a FEM-participant object is NOT supported (matches
    heat-sim-blender's ANIMATE Phase 1) and raises ``RuntimeError`` with a clear
    message pointing at ``thermal_role``.

    POINTS domain only. Per the M2 design (irradiance re-bake is a deferred
    non-goal -- see the design spec), irradiance is not computed here: coupling
    into the FEM participants comes from the Dirichlet reservoir's *position* via
    the shared POINTS Laplacian, which is sufficient for a "hot pour" testbed.
    Known v1 limitation: interior-point-volume samples (POINTS domain,
    temperature every frame rather than carried forward (deformation-aware
    interior continuity is an explicit M2 non-goal).

    Known limitation -- **slot-level Dirichlet sources are not supported here**:
    when *assignment* is given, a material slot the sidecar assigns
    ``DIRICHLET_SOURCE`` is pinned correctly in the static :func:`solve_scene`
    but is only re-pinned *once per frame* here, not after every substep (the
    per-substep re-pin loop below only knows about object-level
    ``thermal_role``). Across substeps the FEM/Dirichlet coupling then lets
    those vertices drift away from their reservoir temperature. A scene
    exhibiting this logs a warning naming the affected object(s); use the
    static solve for such scenes until per-vertex Dirichlet is supported in
    the animated substep loop.

    Returns ``({obj_name: (n_frames, n_surface_verts) ndarray}, frames)`` for
    FEM-participant objects only -- Dirichlet sources are not recorded, since
    their temperature is always the constant we configured. Cache-aware: a hit
    on ``cache.read_animated`` short-circuits the solve entirely.
    """
    cache_root = Path(cache_root)
    sim_objects = gather_meshes(scene)

    frame_start = int(frame_start)
    frame_end = int(frame_end)
    every_n = max(1, int(every_n))
    substeps_per_frame = max(1, int(substeps_per_frame))
    frames = list(range(frame_start, frame_end + 1, every_n))

    blend_path = Path(str(getattr(getattr(bpy, "data", None), "filepath", "") or ""))
    key_cfg = {
        "solver": dict(solver_cfg),
        "defaults": dict(defaults),
        "objects": sorted(o.name for o in sim_objects),
        "animated": True,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "every_n": every_n,
        "substeps": substeps_per_frame,
        "assignments": None if assignment is None else assignment.digest,
    }
    key = cache.cache_key(blend_path, key_cfg)
    cache_dir = cache_root / key

    cached = cache.read_animated(cache_dir)
    if cached is not None:
        _log.debug("[heatsim.adapter] animated cache hit: %s", key)
        return cached

    if not frames or not sim_objects:
        cache.write_animated(cache_dir, {}, frames, {"objects": []})
        return {}, frames

    ordered_objects, dirichlet_names = _order_fem_first(sim_objects, defaults)
    objects_by_name = {o.name: o for o in ordered_objects}

    # M2 design: irradiance re-bake is a deferred non-goal for animated mode --
    # ``_combine`` below is always called with an empty ``flux_by_obj``, so every
    # object's incident flux is always zero. The only possible heat source for an
    # animated solve is therefore a DIRICHLET_SOURCE reservoir; warn once so a
    # scene with neither doesn't silently stay at ambient for the whole run.
    if not dirichlet_names:
        _log.warning(
            "[heatsim.adapter] solve_scene_animated: no DIRICHLET_SOURCE object and "
            "animated mode never re-bakes irradiance -- this scene has no heat "
            "source, so the solved field will stay at ambient temperature for the "
            "entire run."
        )

    if assignment is not None:
        mismatched = _slot_level_dirichlet_mismatches(ordered_objects, assignment, defaults)
        if mismatched:
            _log.warning(
                "[heatsim.adapter] solve_scene_animated: %s: material slot(s) assigned "
                "DIRICHLET_SOURCE in the thermal sidecar, but the object-level "
                "thermal_role does not declare it. Slot-level Dirichlet sources are NOT "
                "re-pinned per substep in animated mode, so these vertices will drift "
                "away from their reservoir temperature within a frame -- use the static "
                "solve (solve_scene) for this scene until per-vertex Dirichlet is "
                "supported in the animated substep loop.",
                ", ".join(sorted(mismatched)),
            )

    from visionsim.simulate.heatsim.solver import HeatSimFEM

    device = str(solver_cfg.get("device", "cpu"))
    domain = str(solver_cfg.get("domain", "POINTS")).upper()
    backend = str(solver_cfg.get("laplacian_backend", "ROBUST")).upper()

    fps = _scene_fps(scene)
    dt = (1.0 / fps) / float(substeps_per_frame)

    gen_params = SimpleNamespace(
        device=device,
        RHO=float(defaults["density_kg_m3"]) / _KGM3_TO_KGMM3,
        C=float(defaults["specific_heat_J_kgK"]),
        K=float(defaults["thermal_diffusivity_mm2_s"]),
        NUM_FRAME_DELTA=dt * 60.0,
    )
    sim_params = SimpleNamespace(
        sim_radiation=True,
        sim_convection=False,
        add_tikhonov_reg=False,
        sim_time=0.0,
        record_time=0.0,
    )
    fem = HeatSimFEM(
        gen_params,
        sim_params,
        laplacian_domain=domain,
        laplacian_backend=backend,
    )

    orig_frame = int(scene.frame_current)
    history_rows: dict = {}
    u_prev: np.ndarray | None = None
    n_fem_surface: int | None = None

    try:
        for f in frames:
            scene.frame_set(int(f))
            combined = _combine(ordered_objects, {}, defaults, solver_cfg, assignment=assignment)
            if combined is None:
                raise RuntimeError(
                    f"[heatsim.adapter] solve_scene_animated: no geometry at frame {f} "
                    "(all sim objects vanished mid-run)."
                )

            fem_entries = [
                (name, off, n) for name, off, n, _kind in combined.layout if name not in dirichlet_names
            ]
            cur_n_fem_surface = sum(n for _, _, n in fem_entries)

            if u_prev is None:
                u_prev = combined.t0.copy()
                n_fem_surface = cur_n_fem_surface
            else:
                if cur_n_fem_surface != n_fem_surface:
                    raise RuntimeError(
                        "[heatsim.adapter] solve_scene_animated: a FEM_PARTICIPANT "
                        f"object's vertex count changed at frame {f} (expected "
                        f"{n_fem_surface} surface vertices, got {cur_n_fem_surface}). "
                        "Animated mode requires stable topology for FEM participants; "
                        "set thermal_role=DIRICHLET_SOURCE for meshes with changing "
                        "topology (e.g. fluids)."
                    )
                new_u_prev = combined.t0.copy()
                new_u_prev[:n_fem_surface] = u_prev[:n_fem_surface]
                u_prev = new_u_prev

            # Dirichlet pin: resolve each reservoir vertex's target temperature
            # fresh every frame (also covers a future keyframed
            # dirichlet_temperature_K) and hand the indices/values to
            # simulate_for_pose so it re-pins them internally AFTER EVERY
            # substep's CG solve, not just once on the returned array -- the
            # FEM/Dirichlet coupling weight in the solve would otherwise let
            # pinned nodes drift within a frame across substeps (and drift
            # further as substeps_per_frame grows).
            dir_idx: list[int] = []
            dir_val: list[float] = []
            for name, off, n, _kind in combined.layout:
                if name in dirichlet_names:
                    mat = resolve_material(objects_by_name[name], defaults)
                    t_target = mat["dirichlet_temperature_K"] or mat["initial_temperature_K"]
                    dir_idx.extend(range(off, off + n))
                    dir_val.extend([t_target] * n)

            states = fem.simulate_for_pose(
                combined.verts,
                combined.faces,
                combined.boundary_mask,
                u_prev,
                combined.irradiance,
                combined.alpha,
                combined.density,
                combined.c,
                combined.eps,
                num_substeps=substeps_per_frame,
                dt=dt,
                dirichlet_indices=dir_idx or None,
                dirichlet_values=dir_val or None,
            )  # (substeps, N); Dirichlet rows already pinned every substep.

            u_prev = states[-1].copy()
            for name, off, n in fem_entries:
                history_rows.setdefault(name, []).append(
                    np.asarray(states[-1, off : off + n], dtype=np.float64)
                )
    finally:
        scene.frame_set(orig_frame)

    history = {name: np.stack(rows, axis=0) for name, rows in history_rows.items()}

    meta = {
        "objects": sorted(history),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "every_n": every_n,
        "substeps": substeps_per_frame,
    }
    cache.write_animated(cache_dir, history, frames, meta)
    _log.debug("[heatsim.adapter] animated solve: %d object(s), %d frame(s)", len(history), len(frames))
    return history, frames


def _write_point_float_attr(mesh: Any, name: str, values: np.ndarray) -> None:
    """(Re)create a POINT/FLOAT mesh attribute from ``values``."""
    if name in mesh.attributes:
        try:
            mesh.attributes.remove(mesh.attributes[name])
        except Exception:
            pass
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

        if atlas_plan is not None:
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
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != n:
            # Topology mismatch: a modifier changed the vertex count between the
            # evaluated mesh the FEM solve ran on (``hist``'s vertex axis) and this base
            # mesh (``n`` verts) sim_temperature must live on -- or the history is
            # otherwise empty/malformed. Per-vertex write-back is structurally impossible
            # either way, but the field itself is real, so constant-fill sim_temperature
            # at the mean of its final timestep (preserves the actual heating) instead of
            # collapsing to ambient, and instead of leaving the attribute absent (which
            # would render as 0 K -- worse than either).
            obj["heatsim_default_temperature"] = _fallback_temperature_K(obj, defaults, default_T)
            if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
                fill_T = float(np.mean(np.asarray(arr[timestep], dtype=np.float64)))
            else:
                fill_T = _fallback_temperature_K(obj, defaults, default_T)
            _log.warning(
                "[heatsim.adapter] '%s': solved history has %d vertex-axis entries but the "
                "base mesh has %d vert(s) (a modifier changed the vertex count); per-vertex "
                "sim_temperature detail was replaced by a constant fill (%.2f K).",
                obj.name, arr.shape[1] if arr.ndim == 2 else -1, n, fill_T,
            )
            _write_constant_fill_attributes(obj, mesh, defaults, assignment, fill_T)
            continue

        row = np.asarray(arr[timestep], dtype=np.float32).reshape(-1)
        _write_point_float_attr(mesh, "sim_temperature", row)
        _write_emissivity_attr(obj, mesh, defaults, assignment, n)
        mesh.update()
