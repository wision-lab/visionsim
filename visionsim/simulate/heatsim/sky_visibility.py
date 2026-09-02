# Vendored from heat-sim-blender:addon/lib/sky_visibility.py @ e5b4afe
"""Per-vertex Bent Normal + Sky AO bake for the Direct Kernel sky term.

The SH9 sky prefilter implicitly assumes every vertex sees the full
upper hemisphere. For receivers tucked under overhangs (indoor scenes,
the underside of objects, contact regions) this over-feeds flux. This
module bakes two scalars per vertex that let us approximate the
visibility cone cheaply:

  - **AO** (scalar in [0, 1]): fraction of hemisphere shadow rays that
    were unoccluded.
  - **bent normal** (unit vector): the average direction of the
    unoccluded rays.

At runtime the kernel evaluates the existing Ramamoorthi-Hanrahan
quadratic at the *bent normal* (instead of the surface normal) and
attenuates by AO. This is the Jimenez et al. 2016 / "directional GTAO"
approximation, the production-game standard for cheap diffuse-IBL
occlusion. ~3-5× cheaper to bake than full SH9 PRT, ~1 extra scalar
multiply per evaluation, fixes the indoor over-receiving problem.

Bake cost on a 14k-vertex scene: ~0.1-0.5 s with K=64 rays/vertex via
Embree. Storage: 4 floats per vertex (3 + 1), persisted as mesh
attributes ``bent_normal`` (FLOAT_VECTOR, POINT) and ``sky_ao``
(FLOAT, POINT) and as a disk sidecar mirroring the existing
albedo-cache layout.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import bpy
import numpy as np

_log = logging.getLogger("rich")

_RAY_EPS_M = 1e-4   # push origin off the surface to avoid self-hits, meters


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _stratified_hemisphere_samples(
    k_rays: int, rng: np.random.Generator
) -> np.ndarray:
    """Return (K, 3) unit vectors in the +Z hemisphere using stratified
    sampling, uniform on the hemisphere (each ray has equal solid-angle
    weight). The local frame is +Z = surface normal; we transform per-vertex
    via the orthonormal basis built in ``_build_world_frames``.

    Sampling: jittered stratified ``side × side`` grid in (cos θ, φ). For
    uniform-on-hemisphere, cos θ ∈ [0, 1] is uniform and φ ∈ [0, 2π) is
    uniform. We over-sample to side² ≥ K then trim.
    """
    side = int(math.ceil(math.sqrt(max(1, k_rays))))
    total = side * side
    iu, jv = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    u = (iu.reshape(-1) + rng.random(total)) / float(side)
    v = (jv.reshape(-1) + rng.random(total)) / float(side)
    if total > k_rays:
        # Trim deterministically (no random subset — we already jittered).
        u = u[:k_rays]
        v = v[:k_rays]
    cos_theta = u                              # uniform on hemisphere → cos θ ∈ [0, 1]
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta * cos_theta, 0.0, 1.0))
    phi = 2.0 * math.pi * v
    local = np.column_stack([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta,
    ]).astype(np.float64)
    return local


def _build_world_frames(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized orthonormal basis per normal using the Duff et al. 2017
    "Building an Orthonormal Basis, Revisited" method (branchless, robust).
    Returns ``(T, B)`` — both (N, 3) — completing a right-handed frame
    [T, B, N] with the input ``normals`` as the third column."""
    n = normals.astype(np.float64)
    sign = np.where(n[:, 2] >= 0.0, 1.0, -1.0)
    a = -1.0 / (sign + n[:, 2])
    b = n[:, 0] * n[:, 1] * a
    T = np.column_stack([
        1.0 + sign * n[:, 0] * n[:, 0] * a,
        sign * b,
        -sign * n[:, 0],
    ])
    B = np.column_stack([
        b,
        sign + n[:, 1] * n[:, 1] * a,
        -n[:, 1],
    ])
    return T, B


def _hemisphere_world_dirs(
    normals: np.ndarray,
    local_samples: np.ndarray,
) -> np.ndarray:
    """Returns (N, K, 3) world-space directions oriented around per-vertex
    normals. ``normals`` is (N, 3); ``local_samples`` is (K, 3) in the
    local +Z-up hemisphere frame."""
    T, B = _build_world_frames(normals)               # (N, 3), (N, 3)
    # world_dir[v, k] = T[v]*local[k,0] + B[v]*local[k,1] + N[v]*local[k,2]
    # Broadcast: (N, 1, 3) * (1, K, 1) → sum
    nrm = normals.astype(np.float64)
    out = (
        T[:, None, :] * local_samples[None, :, 0:1]
        + B[:, None, :] * local_samples[None, :, 1:2]
        + nrm[:, None, :] * local_samples[None, :, 2:3]
    )
    return out


# ---------------------------------------------------------------------------
# Bake core
# ---------------------------------------------------------------------------


def bake_for_object(
    obj: bpy.types.Object,
    backend,
    k_rays: int,
    rng: np.random.Generator,
    far_dist_m: float = 1.0e6,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Bake (bent_normal, ao) for one receiver mesh against ``backend``
    (already built from the scene's occluder set).

    Returns ``(bent_normal (N, 3) float64, ao (N,) float64)`` or None on
    extraction failure. Bent normals are unit-length; AO is in [0, 1].
    Fully-occluded vertices (AO ≈ 0) keep the surface normal as fallback.
    """
    # Lazy import — irradiance_kernel imports us, avoid circularity.
    from visionsim.simulate.heatsim import irradiance_kernel

    geom = irradiance_kernel._extract_world_geometry(obj)
    if geom is None:
        return None
    world_verts, world_normals, _faces = geom
    Nv = int(world_verts.shape[0])
    if Nv == 0:
        return None

    local_samples = _stratified_hemisphere_samples(k_rays, rng)  # (K, 3)
    K = int(local_samples.shape[0])

    world_dirs = _hemisphere_world_dirs(world_normals, local_samples)  # (N, K, 3)

    origins = (
        world_verts.astype(np.float32)[:, None, :]
        + _RAY_EPS_M * world_normals.astype(np.float32)[:, None, :]
    )
    origins = np.broadcast_to(origins, (Nv, K, 3)).copy()
    dirs = world_dirs.astype(np.float32)
    max_dists = np.full((Nv, K), float(far_dist_m), dtype=np.float32)

    flat_o = origins.reshape(Nv * K, 3)
    flat_d = dirs.reshape(Nv * K, 3)
    flat_md = max_dists.reshape(Nv * K)

    occluded = backend.shadow_rays(flat_o, flat_d, flat_md)
    visible = (~occluded).reshape(Nv, K)  # bool (N, K)

    ao = visible.mean(axis=1).astype(np.float64)  # (N,)

    # Bent normal = average of visible directions, then normalize. Fall back
    # to surface normal where no rays survived.
    vis_mask = visible[..., None].astype(np.float64)  # (N, K, 1)
    bent_unnorm = (world_dirs * vis_mask).sum(axis=1)  # (N, 3)
    norms = np.linalg.norm(bent_unnorm, axis=1, keepdims=True)
    has_some_visible = (norms > 1e-12).reshape(-1)
    bent_normal = np.empty_like(world_normals, dtype=np.float64)
    bent_normal[has_some_visible] = (
        bent_unnorm[has_some_visible] / norms[has_some_visible]
    )
    bent_normal[~has_some_visible] = world_normals[~has_some_visible]

    return bent_normal, ao


# ---------------------------------------------------------------------------
# Mesh-attribute persistence
# ---------------------------------------------------------------------------


_BENT_ATTR = "bent_normal"
_AO_ATTR = "sky_ao"


def _read_bent_ao_attrs(obj: bpy.types.Object) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read the mesh-attribute cache for one object. Returns
    ``(bent_normal, ao)`` or None if missing / wrong shape."""
    mesh = getattr(obj, "data", None)
    if mesh is None:
        return None
    try:
        bent_attr = mesh.attributes.get(_BENT_ATTR)
        ao_attr = mesh.attributes.get(_AO_ATTR)
    except Exception:
        return None
    if bent_attr is None or ao_attr is None:
        return None
    if (
        getattr(bent_attr, "domain", None) != "POINT"
        or getattr(bent_attr, "data_type", None) != "FLOAT_VECTOR"
    ):
        return None
    if (
        getattr(ao_attr, "domain", None) != "POINT"
        or getattr(ao_attr, "data_type", None) != "FLOAT"
    ):
        return None
    n = len(mesh.vertices)
    if n != len(bent_attr.data) or n != len(ao_attr.data):
        return None
    bent_flat = np.zeros((n * 3,), dtype=np.float32)
    ao = np.zeros((n,), dtype=np.float32)
    try:
        bent_attr.data.foreach_get("vector", bent_flat)
        ao_attr.data.foreach_get("value", ao)
    except Exception:
        return None
    return bent_flat.reshape(n, 3).astype(np.float64), ao.astype(np.float64)


def _store_bent_ao_attrs(
    obj: bpy.types.Object,
    bent_normal: np.ndarray,
    ao: np.ndarray,
) -> None:
    """Persist (bent_normal, ao) as POINT mesh attributes on ``obj``.
    Skipped silently on vertex-count mismatch (modifier-driven topology)."""
    mesh = getattr(obj, "data", None)
    if mesh is None:
        return
    n = len(mesh.vertices)
    if int(bent_normal.shape[0]) != n or int(ao.shape[0]) != n:
        return
    for attr_name in (_BENT_ATTR, _AO_ATTR):
        if attr_name in mesh.attributes:
            try:
                mesh.attributes.remove(mesh.attributes[attr_name])
            except Exception:
                pass
    try:
        bent_attr = mesh.attributes.new(name=_BENT_ATTR, type="FLOAT_VECTOR", domain="POINT")
        bent_attr.data.foreach_set(
            "vector", np.asarray(bent_normal, dtype=np.float32).reshape(-1)
        )
        ao_attr = mesh.attributes.new(name=_AO_ATTR, type="FLOAT", domain="POINT")
        ao_attr.data.foreach_set("value", np.asarray(ao, dtype=np.float32))
        mesh.update()
    except Exception:
        pass


def remove_bent_ao_attrs(obj: bpy.types.Object) -> None:
    """Drop the cached mesh attributes. Used by Reset / source switch."""
    mesh = getattr(obj, "data", None)
    if mesh is None:
        return
    for attr_name in (_BENT_ATTR, _AO_ATTR):
        if attr_name in mesh.attributes:
            try:
                mesh.attributes.remove(mesh.attributes[attr_name])
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Public entry point — get or bake, with cache resolution
# ---------------------------------------------------------------------------


def get_or_bake_for_objects(
    scene: bpy.types.Scene,
    objects: List[bpy.types.Object],
    backend,
    settings,
    *,
    force_rebake: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """For each object, return ``{"bent_normal", "ao"}``.

    Resolution order per object (skipped when ``force_rebake=True``):
      1. POINT mesh attributes (``bent_normal``, ``sky_ao``)
      2. Disk sidecar (``sky_visibility_cache.npz``)
      3. Bake via Embree, store as attributes + disk cache.

    ``backend`` must be a pre-built BVH for the full scene's occluder set
    (typically the same backend the kernel built for direct-light visibility).
    """
    from visionsim.simulate.heatsim import temperature_io

    k_rays = int(getattr(settings, "sky_ao_rays_per_vertex", 64))
    k_rays = max(4, min(k_rays, 1024))

    disk_cache = temperature_io.read_sky_visibility_cache(scene) or {}
    out: Dict[str, Dict[str, np.ndarray]] = {}
    rng = np.random.default_rng(0)
    newly_baked: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    t0 = time.perf_counter()

    for obj in objects:
        if not force_rebake:
            attr_cached = _read_bent_ao_attrs(obj)
            if attr_cached is not None:
                bn, ao = attr_cached
                out[obj.name] = {"bent_normal": bn, "ao": ao}
                continue
            disk_payload = disk_cache.get(obj.name)
            if disk_payload is not None:
                bn = np.asarray(disk_payload["bent_normal"], dtype=np.float64)
                ao = np.asarray(disk_payload["ao"], dtype=np.float64)
                if int(bn.shape[0]) == len(obj.data.vertices) and int(ao.shape[0]) == bn.shape[0]:
                    _store_bent_ao_attrs(obj, bn, ao)
                    out[obj.name] = {"bent_normal": bn, "ao": ao}
                    continue

        # Bake.
        _log.debug("[HeatSim:SkyVis] Baking bent normal + AO for '%s' (K=%d rays/vertex).", obj.name, k_rays)
        result = bake_for_object(obj, backend, k_rays, rng)
        if result is None:
            continue
        bn, ao = result
        _store_bent_ao_attrs(obj, bn, ao)
        newly_baked[obj.name] = (bn.astype(np.float32), ao.astype(np.float32))
        out[obj.name] = {"bent_normal": bn, "ao": ao}

    if newly_baked:
        merged = dict(disk_cache)
        for name, (bn, ao) in newly_baked.items():
            merged[name] = {"bent_normal": bn, "ao": ao}
        temperature_io.write_sky_visibility_cache(scene, merged)
        dt = time.perf_counter() - t0
        try:
            example = next(iter(newly_baked.values()))
            n_verts = int(example[0].shape[0])
        except Exception:
            n_verts = 0
        _log.debug(
            "[HeatSim:SkyVis] Baked sky visibility for %d object(s) (%d+ verts each) in %.2f s.",
            len(newly_baked), n_verts, dt,
        )

    return out
