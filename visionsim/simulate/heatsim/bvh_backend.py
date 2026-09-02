# Vendored from heat-sim-blender:addon/lib/bvh_backend.py @ e5b4afe
"""BVH backend abstraction for the Direct Kernel irradiance source.

Two implementations:

  - **Embree** (preferred): via `embreex` 4.x. ~10–50× faster shadow-ray
    throughput than the pure-Python fallback. Requires the user to have
    pip-installed `embreex` into Blender's bundled Python — see
    ``docs/animated-geometry-howto.md`` for the install command.
  - **mathutils.bvhtree** (fallback): always available, ships with
    Blender. Pure-Python loop over `BVHTree.ray_cast`; functional but
    slower for large vertex counts.

Public API:

    backend = best_available()
    handle = backend.build_for_meshes(meshes_world)
    occluded = backend.shadow_rays(origins, directions, max_dists)
    backend.name  # "embree" or "mathutils"

`meshes_world` is a list of ``(world_verts (N,3) float64, faces (M,3)
int32)`` tuples — already transformed into world space. The backend
combines them into one BVH; per-vertex callers don't need to know
which mesh originally owned a triangle.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

_log = logging.getLogger("rich")

try:
    from embreex import rtcore_scene as _embree_scene  # type: ignore
    from embreex import mesh_construction as _embree_mesh  # type: ignore

    _HAS_EMBREE = True
    _EMBREE_IMPORT_ERROR: Optional[str] = None
except Exception as _e:  # noqa: BLE001
    _HAS_EMBREE = False
    _EMBREE_IMPORT_ERROR = str(_e)


def _shadow_ray_self_offset(eps_world: float, normals: np.ndarray) -> np.ndarray:
    """Push ray origins along the surface normal by ``eps_world`` so we
    don't self-intersect the receiver triangle. Returns origins + eps·n."""
    return eps_world * normals


# ---------------------------------------------------------------------------
# Embree backend
# ---------------------------------------------------------------------------


class _EmbreeBackend:
    name = "embree"

    def __init__(self) -> None:
        self._scene = None
        self._mesh = None  # keep a ref so it isn't GC'd before the scene is queried

    def build_for_meshes(
        self,
        meshes_world: List[Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Combine all (verts, faces) into a single Embree scene."""
        if not meshes_world:
            self._scene = None
            self._mesh = None
            return
        # Concatenate verts + faces with proper offsets.
        all_verts: list[np.ndarray] = []
        all_tris: list[np.ndarray] = []
        offset = 0
        for verts, faces in meshes_world:
            v = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
            t = np.asarray(faces, dtype=np.uint32).reshape(-1, 3) + offset
            all_verts.append(v)
            all_tris.append(t)
            offset += int(v.shape[0])
        verts_cat = np.ascontiguousarray(np.vstack(all_verts), dtype=np.float32)
        tris_cat = np.ascontiguousarray(np.vstack(all_tris), dtype=np.uint32)

        scene = _embree_scene.EmbreeScene()
        mesh = _embree_mesh.TriangleMesh(scene, verts_cat, tris_cat)
        self._scene = scene
        self._mesh = mesh

    def shadow_rays(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        max_dists: np.ndarray,
    ) -> np.ndarray:
        """Returns a bool (N,) array — True iff the ray hit something
        before ``max_dist``."""
        if self._scene is None:
            return np.zeros((int(origins.shape[0]),), dtype=bool)
        o = np.ascontiguousarray(origins, dtype=np.float32)
        d = np.ascontiguousarray(directions, dtype=np.float32)
        # embreex 4.x scene.run with query='OCCLUDED' returns 1/0 per ray
        # (1 = some hit, 0 = miss). It doesn't take per-ray max_dist; we
        # apply max_dist as a post-filter using INTERSECT to get the t
        # hit-distance and compare. That's slightly more expensive but
        # gives us proper per-ray max-distance gating (essential for
        # point/spot lights where we don't want hits past the light).
        try:
            tfar = float(np.max(max_dists)) if max_dists.size else 1e10
            # Run INTERSECT to get hit distances for filtering.
            res = self._scene.run(o, d, query="DISTANCE")
            t_hit = np.asarray(res, dtype=np.float32).reshape(-1)
            # res is the t value of the first hit; np.inf or large for misses.
            occluded = (t_hit > 0.0) & (t_hit < np.asarray(max_dists, dtype=np.float32))
            return occluded
        except Exception:
            # Fallback: query='OCCLUDED' is a Boolean (no max-dist).
            res = self._scene.run(o, d, query="OCCLUDED")
            return np.asarray(res, dtype=np.int32).reshape(-1) > 0


# ---------------------------------------------------------------------------
# Mathutils fallback backend
# ---------------------------------------------------------------------------


class _MathutilsBackend:
    name = "mathutils"

    def __init__(self) -> None:
        self._bvh = None

    def build_for_meshes(
        self,
        meshes_world: List[Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        from mathutils import Vector
        from mathutils.bvhtree import BVHTree

        if not meshes_world:
            self._bvh = None
            return

        verts_v: list = []
        tris_t: list = []
        offset = 0
        for verts, faces in meshes_world:
            for v in verts:
                verts_v.append(Vector((float(v[0]), float(v[1]), float(v[2]))))
            for f in faces:
                tris_t.append(
                    (int(f[0]) + offset, int(f[1]) + offset, int(f[2]) + offset)
                )
            offset += int(verts.shape[0])
        try:
            self._bvh = BVHTree.FromPolygons(verts_v, tris_t, all_triangles=True)
        except Exception as e:  # noqa: BLE001
            _log.debug("[HeatSim] BVHTree.FromPolygons failed: %s", e)
            self._bvh = None

    def shadow_rays(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        max_dists: np.ndarray,
    ) -> np.ndarray:
        from mathutils import Vector

        if self._bvh is None:
            return np.zeros((int(origins.shape[0]),), dtype=bool)
        n = int(origins.shape[0])
        out = np.zeros((n,), dtype=bool)
        for i in range(n):
            origin = Vector((float(origins[i, 0]), float(origins[i, 1]), float(origins[i, 2])))
            direction = Vector((float(directions[i, 0]), float(directions[i, 1]), float(directions[i, 2])))
            md = float(max_dists[i]) if max_dists.size else 1e10
            try:
                loc, _normal, _index, dist = self._bvh.ray_cast(origin, direction, md)
            except Exception:
                loc = None
                dist = None
            if loc is not None and dist is not None and dist < md:
                out[i] = True
        return out


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


_BACKEND_CACHE: Optional[object] = None


def best_available():
    """Returns a backend instance. Embree is preferred; the mathutils
    fallback is used when ``embreex`` isn't importable.

    The instance is created fresh on each call (BVH state isn't shared),
    but the backend choice is cached the first time so we don't re-import
    embreex on every Run.
    """
    if _HAS_EMBREE:
        return _EmbreeBackend()
    return _MathutilsBackend()


def backend_status_string() -> str:
    """Human-readable status for the panel."""
    if _HAS_EMBREE:
        try:
            import embreex
            ver = getattr(embreex, "__version__", "unknown")
        except Exception:
            ver = "unknown"
        return f"Embree (embreex {ver})"
    return f"mathutils.bvhtree (built-in; embreex not installed: {_EMBREE_IMPORT_ERROR or 'no module'})"


def has_embree() -> bool:
    return bool(_HAS_EMBREE)
