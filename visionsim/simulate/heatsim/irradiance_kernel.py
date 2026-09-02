# Vendored from heat-sim-blender:addon/lib/irradiance_kernel.py @ e5b4afe
"""Direct-Kernel orchestrator: per-vertex irradiance without Cycles bakes.

This is the alternative to `irradiance.bake_irradiance_for_active_frame()`
(Cycles texture bake → sample at vertices). It replaces the Cycles direct
+ indirect light pass with:

  1. **Per-light analytic** form-factors from ``light_models`` (sun, point,
     spot, area).
  2. **Per-vertex visibility** via Embree (or mathutils.bvhtree fallback)
     in ``bvh_backend`` — one ray per (light × vertex) sample point. Area
     lights take K rays per vertex (stratified shadow MC).
  3. **Sky / HDRI** via 9-coefficient SH irradiance using
     ``sh9_sky.prefilter_world`` + ``eval_per_vertex``. No rays — the
     SH9 reconstruction implicitly assumes the sky is unoccluded above
     the upper hemisphere of each vertex.
  4. **Per-vertex albedo** baked once via Cycles (``bake_albedo_map``)
     and cached on disk as a sidecar npz. Sampled at vertices via the
     existing UV-mapped sampler; stored as a mesh attribute named
     ``albedo`` for re-use.

Output: ``{obj: {"vertex_flux": (N,) float64 W/m²}}`` — absorbed flux per
vertex (post-(1−albedo)). The FEM solver consumes this directly through
the existing ``combined_irradiance`` slot, bypassing texture sampling.

Relative to the Cycles bake this is faster (no GPU pass, no UV unwrap,
no texture allocation) but sacrifices fidelity: no indirect bounce, no
specular contribution, sky is unshadowed by definition. For thermal use
on time-varying scenes this is generally a win — see
``proposals/phase-3-direct-light-vertex-kernel.md``.
"""

from __future__ import annotations

import logging
import time

import bpy
import numpy as np

from visionsim.simulate.heatsim import bvh_backend, light_models, occluders, sh9_sky, sky_visibility

_log = logging.getLogger("rich")


# ---------------------------------------------------------------------------
# Geometry extraction
# ---------------------------------------------------------------------------


def _extract_world_geometry(obj: bpy.types.Object) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Returns ``(world_verts (N,3) m, world_normals (N,3) unit, faces (M,3) int32)``
    for ``obj``'s evaluated mesh, or None if extraction fails. Verts are in
    meters (Blender's native unit) so the BVH backend gets consistent units."""
    if obj.type != "MESH":
        return None
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.data
    if len(mesh.vertices) == 0 or len(mesh.polygons) == 0:
        return None

    n_v = len(mesh.vertices)
    co = np.zeros(n_v * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(n_v, 3)

    nrm = np.zeros(n_v * 3, dtype=np.float64)
    mesh.vertices.foreach_get("normal", nrm)
    nrm = nrm.reshape(n_v, 3)

    mw = np.array(obj.matrix_world, dtype=np.float64)
    R = mw[:3, :3]
    t = mw[:3, 3]
    world_verts = (co @ R.T) + t  # (N, 3) meters
    # Normal transform: inv-transpose of R.
    try:
        N_mat = np.linalg.inv(R).T
    except np.linalg.LinAlgError:
        N_mat = R
    world_normals = nrm @ N_mat.T
    n_norm = np.linalg.norm(world_normals, axis=1, keepdims=True)
    n_norm = np.where(n_norm > 1e-12, n_norm, 1.0)
    world_normals = world_normals / n_norm

    # Triangulate via loop_triangles (matches the FEM extractor).
    mesh.calc_loop_triangles()
    n_tri = len(mesh.loop_triangles)
    if n_tri == 0:
        return None
    tri_verts = np.zeros(n_tri * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tri_verts)
    faces = tri_verts.reshape(n_tri, 3)

    return (
        world_verts.astype(np.float64),
        world_normals.astype(np.float64),
        faces.astype(np.int32),
    )


def _collect_scene_meshes_world(scene: bpy.types.Scene) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build the occluder BVH input set: every renderable, shadow-casting mesh
    in the scene, in world meters (so non-sim props still cast shadows)."""
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        if obj.hide_render or not obj.visible_get():
            continue
        if not occluders.casts_shadow(obj):
            continue
        geom = _extract_world_geometry(obj)
        if geom is None:
            continue
        verts, _normals, faces = geom
        out.append((verts, faces))
    return out


# ---------------------------------------------------------------------------
# Albedo: per-vertex via the existing Cycles albedo bake (cached on disk +
# stored as a mesh attribute so subsequent solves don't re-bake).
# ---------------------------------------------------------------------------


def _read_vertex_albedo_attr(obj: bpy.types.Object, attr_name: str) -> np.ndarray | None:
    """Try to read a (N,) float per-vertex albedo from the named POINT/FLOAT
    attribute. Returns None if missing or wrong shape."""
    mesh = getattr(obj, "data", None)
    if mesh is None:
        return None
    try:
        attr = mesh.attributes.get(attr_name)
    except Exception:
        attr = None
    if attr is None:
        return None
    if getattr(attr, "domain", None) != "POINT" or getattr(attr, "data_type", None) != "FLOAT":
        return None
    n = len(mesh.vertices)
    if n != len(attr.data):
        return None
    out = np.zeros((n,), dtype=np.float32)
    try:
        attr.data.foreach_get("value", out)
    except Exception:
        return None
    return out.astype(np.float64)


def store_vertex_flux_attr(obj: bpy.types.Object, vals: np.ndarray, attr_name: str = "sim_irradiance") -> None:
    """Persist (N,) per-vertex absorbed flux (W/m²) as a POINT/FLOAT
    mesh attribute. Used by View Flux to render per-vertex irradiance
    through the turbo colormap in DIRECT_KERNEL mode."""
    mesh = getattr(obj, "data", None)
    if mesh is None:
        return
    if len(mesh.vertices) != int(vals.shape[0]):
        return
    if attr_name in mesh.attributes:
        try:
            mesh.attributes.remove(mesh.attributes[attr_name])
        except Exception:
            pass
    try:
        attr = mesh.attributes.new(name=attr_name, type="FLOAT", domain="POINT")
        attr.data.foreach_set("value", np.asarray(vals, dtype=np.float32))
        mesh.update()
    except Exception:
        pass


def _store_vertex_albedo_attr(obj: bpy.types.Object, attr_name: str, vals: np.ndarray) -> None:
    """Store (N,) per-vertex albedo as a POINT/FLOAT mesh attribute on the
    original (non-evaluated) mesh. Skipped silently on vertex-count mismatch
    (object has modifiers that change topology — kernel albedo not supported
    for those today)."""
    mesh = getattr(obj, "data", None)
    if mesh is None:
        return
    if len(mesh.vertices) != int(vals.shape[0]):
        return
    if attr_name in mesh.attributes:
        try:
            mesh.attributes.remove(mesh.attributes[attr_name])
        except Exception:
            pass
    try:
        attr = mesh.attributes.new(name=attr_name, type="FLOAT", domain="POINT")
        attr.data.foreach_set("value", np.asarray(vals, dtype=np.float32))
        mesh.update()
    except Exception:
        pass


def _bake_vertex_albedo_via_cycles(
    scene: bpy.types.Scene,
    obj: bpy.types.Object,
    texture_size: int,
) -> np.ndarray | None:
    """Run the existing Cycles albedo bake and reduce to (N,) per-vertex
    grayscale (luminance). Returns None on bake failure."""
    from visionsim.simulate.heatsim import irradiance

    baked = irradiance.bake_albedo_map(scene, obj, texture_size)
    if baked is None or baked.pixels is None:
        return None

    # Must match what the solver indexes against -- see irradiance._mesh_to_sample. This
    # is the third place the same base-vs-evaluated mesh choice is made; getting it wrong
    # here sizes the albedo to the pre-modifier vertex count, and the caller then discards
    # it in favour of albedo=0 (full absorption), overestimating absorbed flux.
    mesh = irradiance._mesh_to_sample(obj)
    mesh.calc_loop_triangles()
    if len(mesh.loop_triangles) == 0:
        return None

    from visionsim.simulate.heatsim.constants import BAKE_UV_LAYER_NAME

    uv_layer = (
        mesh.uv_layers.get(BAKE_UV_LAYER_NAME)
        or mesh.uv_layers.active
        or (mesh.uv_layers[0] if mesh.uv_layers else None)
    )
    if uv_layer is None:
        return None

    n_loops = len(mesh.loops)
    uv_flat = np.zeros(n_loops * 2, dtype=np.float64)
    uv_layer.data.foreach_get("uv", uv_flat)
    uv = uv_flat.reshape(-1, 2)

    loop_vert_indices = np.zeros(n_loops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vert_indices)

    pixels = baked.pixels
    h, w = int(pixels.shape[0]), int(pixels.shape[1])
    u = uv[:, 0] - np.floor(uv[:, 0])
    v = uv[:, 1] - np.floor(uv[:, 1])
    px = np.clip((u * (w - 1)).astype(np.int32), 0, w - 1)
    py = np.clip((v * (h - 1)).astype(np.int32), 0, h - 1)
    sampled = pixels[py, px]  # (n_loops, 3)
    luma = (
        sampled[..., 0] * 0.2126 + sampled[..., 1] * 0.7152 + sampled[..., 2] * 0.0722
    ).astype(np.float64)
    luma = np.clip(luma, 0.0, 1.0)

    n_verts = len(mesh.vertices)
    accum = np.zeros(n_verts, dtype=np.float64)
    counts = np.zeros(n_verts, dtype=np.int32)
    np.add.at(accum, loop_vert_indices, luma)
    np.add.at(counts, loop_vert_indices, 1)
    counts = np.where(counts > 0, counts, 1)
    return accum / counts


def get_or_bake_vertex_albedo(
    scene: bpy.types.Scene,
    objects: list[bpy.types.Object],
    *,
    texture_size: int,
    attr_name: str = "albedo",
    disk_cache: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """For each object, return its per-vertex albedo (N,) in [0, 1].

    Resolution order:
      1. The mesh attribute ``attr_name`` if present and shape matches.
      2. The disk_cache dict (loaded by the caller from the kernel albedo
         sidecar).
      3. Cycles bake → store as both attribute and (caller's responsibility
         to) disk cache.

    Returns ``{obj.name: (N,) float64}``. Objects whose albedo can't be
    obtained (no UVs, bake failure) are absent from the return — the caller
    should treat those as albedo=0 (full absorption) for safety.
    """
    out: dict[str, np.ndarray] = {}
    if disk_cache is None:
        disk_cache = {}

    for obj in objects:
        # 1. Existing mesh attribute.
        vals = _read_vertex_albedo_attr(obj, attr_name)
        # An all-zero albedo is the degenerate "fully absorbing" fallback (and the
        # sentinel a stale/cross-tool cache leaves behind), never a real bake — so
        # ignore it and fall through to a fresh bake.
        if vals is not None and vals.size > 0 and float(np.max(vals)) > 0.0:
            out[obj.name] = np.clip(vals, 0.0, 1.0)
            continue

        # 2. Disk cache.
        cached = disk_cache.get(obj.name)
        if (cached is not None
                and int(cached.shape[0]) == len(obj.data.vertices)
                and np.asarray(cached).size > 0
                and float(np.max(np.asarray(cached))) > 0.0):
            vals = np.clip(np.asarray(cached, dtype=np.float64), 0.0, 1.0)
            _store_vertex_albedo_attr(obj, attr_name, vals)
            out[obj.name] = vals
            continue

        # 3. Cycles bake (one-shot).
        _log.debug("[HeatSim:Kernel] Baking per-vertex albedo for '%s' (one-time, then cached).", obj.name)
        vals = _bake_vertex_albedo_via_cycles(scene, obj, texture_size)
        if vals is None:
            _log.debug("[HeatSim:Kernel] Albedo bake failed for '%s'; treating as black (full absorption).", obj.name)
            continue
        vals = np.clip(vals, 0.0, 1.0)
        _store_vertex_albedo_attr(obj, attr_name, vals)
        out[obj.name] = vals

    return out


# ---------------------------------------------------------------------------
# Visibility-aware light contribution
# ---------------------------------------------------------------------------


def _accumulate_light_contributions(
    light_objs: list[bpy.types.Object],
    vert_positions: np.ndarray,
    vert_normals: np.ndarray,
    backend,
    n_samples_for_area: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Returns (N,) W/m² unshadowed → shadowed direct irradiance from all
    lights. ``backend`` is a built BVH (ready for shadow_rays())."""
    Nv = int(vert_positions.shape[0])
    total = np.zeros((Nv,), dtype=np.float64)

    for light in light_objs:
        irr_per_sample, origins, dirs, max_dists = light_models.evaluate_light(
            light, vert_positions, vert_normals,
            n_samples_for_area=n_samples_for_area, rng=rng,
        )
        if not bool(np.any(irr_per_sample > 0)):
            continue
        S, N = irr_per_sample.shape
        # Only fire rays for samples with nonzero unshadowed contribution.
        mask = irr_per_sample > 0
        if not bool(np.any(mask)):
            continue

        # Flatten the (S, N) stack into a single ray batch.
        flat_o = origins.reshape(S * N, 3)
        flat_d = dirs.reshape(S * N, 3)
        flat_md = max_dists.reshape(S * N)
        flat_mask = mask.reshape(S * N)

        # Shadow-ray only for masked samples.
        contributions = np.zeros((S * N,), dtype=np.float64)
        if int(flat_mask.sum()) > 0:
            occluded = backend.shadow_rays(
                flat_o[flat_mask], flat_d[flat_mask], flat_md[flat_mask],
            )
            visible = ~occluded
            irr_flat = irr_per_sample.reshape(S * N)
            contributions[flat_mask] = irr_flat[flat_mask] * visible.astype(np.float64)
        contributions = contributions.reshape(S, N)
        total += contributions.sum(axis=0)

    return total


# ---------------------------------------------------------------------------
# Sky cache (in-memory; one entry per session)
# ---------------------------------------------------------------------------


_SKY_CACHE: dict[str, np.ndarray] = {}


def _get_sky_coefficients(world) -> np.ndarray:
    """Cached SH9 prefilter. Hashes the world; recomputes when the hash
    changes (or on the first call of a session)."""
    key = sh9_sky.world_hash(world)
    cached = _SKY_CACHE.get(key)
    if cached is not None:
        return cached
    coeffs = sh9_sky.prefilter_world(world)
    _SKY_CACHE[key] = coeffs
    return coeffs


def invalidate_sky_cache() -> None:
    """Drop the in-memory SH9 cache. Called by Reset / source-switch."""
    _SKY_CACHE.clear()


# ---------------------------------------------------------------------------
# Shared points->irradiance core (used by both the per-vertex and texel paths)
# ---------------------------------------------------------------------------


def _sky_irradiance_plain(normals: np.ndarray, sky_coeffs: np.ndarray, sky_has_energy: bool) -> np.ndarray:
    """Unoccluded SH9 sky irradiance at ``normals`` - (N,) W/m².

    No per-point bent-normal/AO occlusion: this is what the per-vertex path already
    falls back to when sky occlusion is disabled or unavailable for a vertex, and it is
    the ONLY sky term the texel path uses (texels have no baked per-point AO)."""
    if not sky_has_energy:
        return np.zeros((int(normals.shape[0]),), dtype=np.float64)
    return sh9_sky.eval_per_vertex(normals, sky_coeffs)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_per_vertex_irradiance(
    scene: bpy.types.Scene,
    sim_objects: list[bpy.types.Object],
    settings,
    *,
    objects_to_compute: list[bpy.types.Object] | None = None,
    force_sky_revisibility: bool = False,
) -> dict[bpy.types.Object, dict]:
    """Direct-Kernel replacement for ``bake_irradiance_for_active_frame``.

    Returns ``{obj: {"vertex_flux": (N,) float64 W/m² absorbed}}``.

    The flux is post-albedo (i.e. ``Q_abs = E_total · (1 − albedo_gray)``)
    so the FEM solver can install it directly without further conversion.

    ``objects_to_compute`` is the per-frame subset to update (mirrors
    ``bake_irradiance_for_active_frame``'s `objects_to_bake`). Defaults to
    every sim object.
    """
    if objects_to_compute is None:
        objects_to_compute = list(sim_objects)
    # Dirichlet sources don't need a per-vertex flux contribution — their
    # temperature is pinned regardless of what light hits them. Skip them
    # entirely (no albedo bake, no sky-visibility bake, no flux output).
    objects_to_compute = [
        o for o in objects_to_compute
        if str(getattr(getattr(o, "heat_sim_material", None), "thermal_role", "FEM_PARTICIPANT")).upper()
        != "DIRICHLET_SOURCE"
    ]
    if not objects_to_compute:
        return {}

    t0 = time.perf_counter()

    # 1. Gather lights, scene-wide mesh occluders, and the BVH backend.
    light_objs = [
        o for o in scene.objects
        if getattr(o, "type", None) == "LIGHT" and not o.hide_render and o.visible_get()
    ]
    scene_meshes = _collect_scene_meshes_world(scene)
    backend = bvh_backend.best_available()
    backend.build_for_meshes(scene_meshes)

    # 2. Sky SH9 (cached).
    sky_coeffs = _get_sky_coefficients(getattr(scene, "world", None))
    sky_has_energy = sh9_sky.coeffs_have_energy(sky_coeffs)

    # 3. Per-vertex albedo (cached on disk + as mesh attribute).
    from visionsim.simulate.heatsim import temperature_io
    albedo_disk = temperature_io.read_kernel_albedo_cache(scene) or {}
    texture_size_for_albedo = int(getattr(settings, "irradiance_texture_size", 512))
    vertex_albedo_map = get_or_bake_vertex_albedo(
        scene, objects_to_compute,
        texture_size=texture_size_for_albedo,
        disk_cache=albedo_disk,
    )
    # Persist any newly-baked albedos to disk.
    if vertex_albedo_map:
        merged = dict(albedo_disk)
        merged.update(vertex_albedo_map)
        temperature_io.write_kernel_albedo_cache(scene, merged)

    # 3b. Per-vertex sky visibility (bent normal + AO), cached on disk +
    # as mesh attributes. Skipped when sky is unenergetic or feature is off.
    enable_sky_occ = bool(getattr(settings, "enable_sky_occlusion", True))
    sky_vis_map: dict[str, dict[str, np.ndarray]] = {}
    if enable_sky_occ and sky_has_energy:
        sky_vis_map = sky_visibility.get_or_bake_for_objects(
            scene, objects_to_compute, backend, settings,
            force_rebake=bool(force_sky_revisibility),
        )
    ao_floor = float(getattr(settings, "sky_ao_min_for_bent", 0.02))

    # 4. Per-receiver computation.
    n_samples_for_area = max(1, int(getattr(settings, "direct_kernel_soft_shadow_rays", 8)))
    rng = np.random.default_rng(0)
    out: dict[bpy.types.Object, dict] = {}

    for obj in objects_to_compute:
        geom = _extract_world_geometry(obj)
        if geom is None:
            continue
        world_verts, world_normals, _faces = geom
        Nv = int(world_verts.shape[0])

        # Direct light contribution (visibility-aware).
        direct = _accumulate_light_contributions(
            light_objs, world_verts, world_normals,
            backend, n_samples_for_area, rng,
        )

        # Sky contribution. SH9 is the prefiltered ambient form; when
        # sky-occlusion is enabled we evaluate it at the per-vertex bent
        # normal and attenuate by AO. When AO falls below the floor the
        # bent normal is unreliable (very few visible rays) — fall back
        # to the surface normal.
        sv = sky_vis_map.get(obj.name) if sky_has_energy else None
        if sky_has_energy and enable_sky_occ and sv is not None:
            bn = np.asarray(sv["bent_normal"], dtype=np.float64)
            ao = np.asarray(sv["ao"], dtype=np.float64)
            if int(bn.shape[0]) == Nv and int(ao.shape[0]) == Nv:
                use_bent = (ao >= ao_floor)[:, None]
                eval_normals = np.where(use_bent, bn, world_normals)
                sky_irr = sh9_sky.eval_per_vertex(eval_normals, sky_coeffs) * ao
            else:
                sky_irr = _sky_irradiance_plain(world_normals, sky_coeffs, sky_has_energy)
        else:
            sky_irr = _sky_irradiance_plain(world_normals, sky_coeffs, sky_has_energy)

        e_total = direct + sky_irr  # (N,) W/m²

        # Albedo absorption.
        alb = vertex_albedo_map.get(obj.name)
        if alb is not None and int(alb.shape[0]) == Nv:
            absorbed = e_total * (1.0 - alb)
        else:
            # Missing albedo → assume fully absorbing (safer than fully reflecting).
            absorbed = e_total

        out[obj] = {"vertex_flux": absorbed.astype(np.float64)}
        store_vertex_flux_attr(obj, absorbed)

        try:
            mean_e = float(np.mean(e_total))
            max_e = float(np.max(e_total))
            mean_a = float(np.mean(absorbed))
            _log.debug(
                "  %s (Nv=%d): E_mean=%.2f W/m², E_max=%.2f, absorbed_mean=%.2f (post-albedo)",
                obj.name, Nv, mean_e, max_e, mean_a,
            )
        except Exception:
            pass

    dt = time.perf_counter() - t0
    _log.debug(
        "[HeatSim:Kernel] Direct-kernel irradiance computed for %d/%d object(s) in %.2f s "
        "(BVH backend: %s).",
        len(out), len(objects_to_compute), dt, backend.name,
    )
    return out


# ---------------------------------------------------------------------------
# Texel-sim entry point (Task 2 / thermal atlas)
# ---------------------------------------------------------------------------


def compute_irradiance_at_points(
    scene: bpy.types.Scene,
    positions_mm: np.ndarray,
    normals: np.ndarray,
    albedo: np.ndarray,
    *,
    backend=None,
    n_samples_for_area: int = 8,
) -> np.ndarray:
    """Direct-Kernel irradiance at arbitrary world-space points - the texel-sim entry point.

    Factored from :func:`compute_per_vertex_irradiance`'s per-object core
    (:func:`_accumulate_light_contributions` + :func:`_sky_irradiance_plain`) so the two
    paths cannot silently diverge; this function does not change what
    :func:`compute_per_vertex_irradiance` computes for a mesh's own vertices - it is a new,
    additional entry point for point clouds that aren't a mesh's vertex set (e.g. a
    thermal-atlas tile's texel centers from ``atlas.rasterize_tile``).

    Returns ``(K,) W/m²`` ABSORBED flux (post-``(1 - albedo)``), matching
    :func:`compute_per_vertex_irradiance`'s ``vertex_flux`` contract.

    Unlike the per-vertex path this has no baked per-point sky-occlusion (bent normal /
    AO) - every point uses its own surface normal for the (unoccluded) SH9 sky term, the
    same fallback the per-vertex path takes when sky occlusion is disabled or unavailable
    for a given vertex.

    Args:
        scene: The Blender scene (for lights, occluders and the world/sky).
        positions_mm: ``(K, 3)`` world-space positions in **millimetres** (the adapter's
            native unit, e.g. ``atlas.rasterize_tile``'s ``position_mm``); converted to
            metres internally to match the light/BVH pipeline's units
            (``_extract_world_geometry`` works in metres).
        normals: ``(K, 3)`` world-space unit normals.
        albedo: ``(K,)`` per-point albedo in ``[0, 1]``. A shape mismatch against
            ``positions_mm`` is treated as "no albedo" (full absorption), matching
            :func:`compute_per_vertex_irradiance`'s missing-albedo fallback.
        backend: An optional, already-built BVH backend (``bvh_backend.best_available()``
            with ``.build_for_meshes(...)`` already called). When given, the scene-mesh
            collection and BVH build are skipped entirely and this backend is used as-is
            -- lets a caller that queries many point sets against the same static scene
            (e.g. one call per atlas-participating object in a single solve) build the
            BVH once and reuse it, instead of paying a full scene BVH rebuild per call.
            Defaults to ``None``, which preserves today's behaviour: build a fresh backend
            from ``scene`` on every call.
        n_samples_for_area: Shadow-ray sample count for area lights, the same knob
            :func:`compute_per_vertex_irradiance` reads from
            ``settings.direct_kernel_soft_shadow_rays``. Defaults to ``8`` (that setting's
            own default), so omitting it reproduces today's behaviour exactly.
    """
    positions_mm = np.asarray(positions_mm, dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    albedo = np.asarray(albedo, dtype=np.float64).reshape(-1)
    k = int(positions_mm.shape[0])
    if k == 0:
        return np.zeros((0,), dtype=np.float64)

    positions_m = positions_mm / 1000.0

    light_objs = [
        o for o in scene.objects
        if getattr(o, "type", None) == "LIGHT" and not o.hide_render and o.visible_get()
    ]
    if backend is None:
        scene_meshes = _collect_scene_meshes_world(scene)
        backend = bvh_backend.best_available()
        backend.build_for_meshes(scene_meshes)

    sky_coeffs = _get_sky_coefficients(getattr(scene, "world", None))
    sky_has_energy = sh9_sky.coeffs_have_energy(sky_coeffs)

    n_samples_for_area = max(1, int(n_samples_for_area))
    rng = np.random.default_rng(0)

    direct = _accumulate_light_contributions(light_objs, positions_m, normals, backend, n_samples_for_area, rng)
    sky_irr = _sky_irradiance_plain(normals, sky_coeffs, sky_has_energy)
    e_total = direct + sky_irr

    if albedo.shape[0] != k:
        albedo = np.zeros(k, dtype=np.float64)
    absorbed = e_total * (1.0 - np.clip(albedo, 0.0, 1.0))
    return absorbed.astype(np.float64)
