# Vendored from heat-sim-blender:addon/lib/light_models.py @ e5b4afe
"""Per-light analytic diffuse-irradiance forms for the Direct Kernel.

Each light type contributes one of:

  - **Sun (directional)**: 1 shadow ray per vertex.
    E = light.energy · max(0, n · L_to_sun)
    where ``light.energy`` is W/m² (Blender's sun is a per-area unit
    directly).

  - **Point**: 1 shadow ray per vertex.
    E = light.energy / (4π·d²) · max(0, n · L_to_light)
    where ``light.energy`` is W (radiated power isotropically).

  - **Spot**: same as point × spot attenuation (smoothstep between
    ``cos(spot_size·(1−spot_blend)/2)`` and ``cos(spot_size/2)``).

  - **Area**: K stratified shadow samples across the area surface.
    Per-sample contribution is the form-factor + Lambert emission:
    dE = L_e · cos_emit · cos_recv / d² · dA
    Summed over K, with K rays per vertex queried against the BVH.

The orchestrator (`irradiance_kernel.compute_per_vertex_irradiance`)
calls ``evaluate_light(...)`` per light, which returns the per-sample
unshadowed irradiance and the per-sample shadow-ray geometry. The
orchestrator then batch-queries the BVH and masks contributions by
visibility.

Returns:
    `(irradiance_per_sample, ray_origins, ray_dirs, ray_max_dists)`
        - irradiance_per_sample: (S, N) float64 W/m² unshadowed
        - ray_origins:           (S, N, 3) float32 world coordinates
        - ray_dirs:              (S, N, 3) float32 unit vectors
        - ray_max_dists:         (S, N) float32 (∞ for sun, distance for others)

S = 1 for sun/point/spot; S = soft_shadow_rays for area.
N = number of vertices in the receiver mesh.
"""

from __future__ import annotations

import math

import numpy as np

_RAY_EPS = 1e-4  # mm-ish; pushes the ray origin off the surface so we don't self-hit.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _world_axis_z_neg(matrix_world) -> np.ndarray:
    """Blender's lights point along their local -Z axis (Cycles convention).
    Returns the world-space direction of the light's emission axis as a
    unit vector."""
    m = np.asarray(matrix_world, dtype=np.float64)
    # Take the world rotation of the local (0, 0, -1) axis.
    local_axis = np.array([0.0, 0.0, -1.0])
    world = (m[:3, :3] @ local_axis)
    n = float(np.linalg.norm(world))
    return (world / n) if n > 1e-12 else local_axis


def _light_world_pos(matrix_world) -> np.ndarray:
    """World-space position of the light."""
    m = np.asarray(matrix_world, dtype=np.float64)
    return m[:3, 3].astype(np.float32)


# ---------------------------------------------------------------------------
# Sun
# ---------------------------------------------------------------------------


def evaluate_sun(
    light_obj,
    vert_positions: np.ndarray,  # (N, 3) world meters
    vert_normals: np.ndarray,    # (N, 3) world unit
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    energy = float(getattr(light_obj.data, "energy", 0.0) or 0.0)
    sun_emit_dir = _world_axis_z_neg(light_obj.matrix_world)  # direction sun emits toward
    # Direction from a vertex TO the sun (opposite of emission direction).
    L = (-sun_emit_dir).astype(np.float32)

    n = vert_normals.astype(np.float64)
    cos_theta = np.maximum(0.0, n @ L.astype(np.float64))  # (N,)
    irr = (energy * cos_theta).astype(np.float64)  # (N,) W/m²

    # Shadow rays: from vertex pushed off surface, toward the sun, very far.
    Nv = int(vert_positions.shape[0])
    origins = (vert_positions.astype(np.float32)
               + (_RAY_EPS * vert_normals.astype(np.float32)))
    origins = origins.reshape(1, Nv, 3)
    dirs = np.broadcast_to(L.reshape(1, 1, 3), (1, Nv, 3)).astype(np.float32)
    max_dists = np.full((1, Nv), 1e9, dtype=np.float32)

    return irr.reshape(1, Nv), origins.copy(), dirs.copy(), max_dists


# ---------------------------------------------------------------------------
# Point
# ---------------------------------------------------------------------------


def evaluate_point(
    light_obj,
    vert_positions: np.ndarray,
    vert_normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    energy = float(getattr(light_obj.data, "energy", 0.0) or 0.0)
    light_pos = _light_world_pos(light_obj.matrix_world).astype(np.float64)

    delta = light_pos.reshape(1, 3) - vert_positions.astype(np.float64)  # (N, 3)
    d2 = np.einsum("ij,ij->i", delta, delta)  # (N,)
    # Near-field floor: Blender models a POINT light as a sphere of radius
    # `shadow_soft_size` (0 by default), not a literal zero-size point. Inside
    # that radius the 1/(4*pi*d^2) point-source falloff has no physical meaning
    # and diverges without bound as a receiver approaches the light's origin --
    # a `max(d2, 1e-20)` numerical floor is nowhere near tight enough to catch
    # this (see the identical, *reached* bug in evaluate_area below). Clamp d^2
    # to at least the light's own radius (and a small absolute floor for a
    # literal radius=0 light) so a close receiver saturates instead of exploding.
    _radius = float(getattr(light_obj.data, "shadow_soft_size", 0.0) or 0.0)
    d2 = np.maximum(d2, max(_radius * _radius, 1e-6))
    d = np.sqrt(d2)  # (N,)
    L = delta / d[:, None]  # (N, 3) unit vec from vertex to light

    n = vert_normals.astype(np.float64)
    cos_theta = np.maximum(0.0, np.einsum("ij,ij->i", n, L))
    # Isotropic point: energy distributed over 4π steradians.
    irr = (energy / (4.0 * math.pi * d2)) * cos_theta  # (N,) W/m²

    Nv = int(vert_positions.shape[0])
    origins = (vert_positions.astype(np.float32)
               + (_RAY_EPS * vert_normals.astype(np.float32)))
    return (
        irr.reshape(1, Nv),
        origins.reshape(1, Nv, 3).astype(np.float32),
        L.astype(np.float32).reshape(1, Nv, 3),
        d.astype(np.float32).reshape(1, Nv) - _RAY_EPS,
    )


# ---------------------------------------------------------------------------
# Spot
# ---------------------------------------------------------------------------


def evaluate_spot(
    light_obj,
    vert_positions: np.ndarray,
    vert_normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    energy = float(getattr(light_obj.data, "energy", 0.0) or 0.0)
    light_pos = _light_world_pos(light_obj.matrix_world).astype(np.float64)
    spot_axis = _world_axis_z_neg(light_obj.matrix_world).astype(np.float64)
    spot_size = float(getattr(light_obj.data, "spot_size", math.pi / 4.0) or 0.0)
    spot_blend = float(np.clip(getattr(light_obj.data, "spot_blend", 0.15) or 0.0, 0.0, 1.0))

    half_outer = spot_size * 0.5
    half_inner = half_outer * (1.0 - spot_blend)
    cos_outer = math.cos(half_outer)
    cos_inner = math.cos(half_inner)

    delta = light_pos.reshape(1, 3) - vert_positions.astype(np.float64)  # (N, 3)
    d2 = np.einsum("ij,ij->i", delta, delta)
    # Near-field floor: same reasoning as evaluate_point -- a SPOT light is
    # modelled as a sphere of radius `shadow_soft_size` (0 by default), so the
    # point-source falloff is only meaningful outside that radius.
    _radius = float(getattr(light_obj.data, "shadow_soft_size", 0.0) or 0.0)
    d2 = np.maximum(d2, max(_radius * _radius, 1e-6))
    d = np.sqrt(d2)
    L = delta / d[:, None]  # vertex → light

    n = vert_normals.astype(np.float64)
    cos_theta_recv = np.maximum(0.0, np.einsum("ij,ij->i", n, L))
    base = energy / (4.0 * math.pi * d2) * cos_theta_recv

    # Spot attenuation: cosine of angle between spot axis (light → scene)
    # and the direction from light to the vertex.
    light_to_vert = -L  # (N, 3) (light → vertex)
    cos_axis = np.einsum("j,ij->i", spot_axis, light_to_vert)
    if cos_inner > cos_outer:
        atten = np.clip((cos_axis - cos_outer) / (cos_inner - cos_outer), 0.0, 1.0)
    else:
        atten = (cos_axis >= cos_outer).astype(np.float64)
    irr = base * atten

    Nv = int(vert_positions.shape[0])
    origins = (vert_positions.astype(np.float32)
               + (_RAY_EPS * vert_normals.astype(np.float32)))
    return (
        irr.reshape(1, Nv),
        origins.reshape(1, Nv, 3).astype(np.float32),
        L.astype(np.float32).reshape(1, Nv, 3),
        d.astype(np.float32).reshape(1, Nv) - _RAY_EPS,
    )


# ---------------------------------------------------------------------------
# Area
# ---------------------------------------------------------------------------


def _area_surface_samples(light_obj, n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate n_samples positions on the area light's surface (world
    coordinates) plus the world-space surface normal (one normal — area
    lights are flat). Returns (positions (S, 3), normal (3,), area_m²).

    Stratified inside the bounding rectangle. For DISK / ELLIPSE we
    reject samples outside the disk and resample (acceptable for small
    rejection ratio ≤ 21 %)."""
    data = light_obj.data
    shape = str(getattr(data, "shape", "SQUARE")).upper()
    sx = float(getattr(data, "size", 1.0) or 0.0)
    sy = float(getattr(data, "size_y", sx) or sx) if shape in ("RECTANGLE", "ELLIPSE") else sx

    if sx <= 1e-9 or sy <= 1e-9:
        return np.zeros((1, 3), dtype=np.float64), np.zeros((3,), dtype=np.float64), 0.0

    # Stratified sample in [-0.5, 0.5]² (local) then transform to world.
    K = max(1, int(n_samples))
    side = math.ceil(math.sqrt(K))
    K_actual = side * side
    grid_u, grid_v = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    jitter_u = rng.random((side, side))
    jitter_v = rng.random((side, side))
    u = (grid_u + jitter_u) / float(side) - 0.5  # in [-0.5, 0.5)
    v = (grid_v + jitter_v) / float(side) - 0.5
    u = u.reshape(-1)
    v = v.reshape(-1)

    if shape in ("DISK", "ELLIPSE"):
        # Reject samples outside the unit ellipse. Up to 21% loss for disk.
        # We're sampling [-0.5, 0.5]² → unit ellipse condition: (2u)² + (2v)² ≤ 1.
        ok = ((2 * u) ** 2 + (2 * v) ** 2) <= 1.0
        u = u[ok]
        v = v[ok]
        if u.size == 0:
            u = np.zeros((1,))
            v = np.zeros((1,))
        K_actual = int(u.size)

    local = np.column_stack([u * sx, v * sy, np.zeros(K_actual)])  # (K, 3)
    m = np.asarray(light_obj.matrix_world, dtype=np.float64)
    local_h = np.column_stack([local, np.ones((K_actual, 1))])
    world = (m @ local_h.T).T[:, :3]

    normal = _world_axis_z_neg(light_obj.matrix_world).astype(np.float64)
    if shape in ("DISK", "ELLIPSE"):
        area = math.pi * (sx * 0.5) * (sy * 0.5)
    else:
        area = sx * sy
    return world.astype(np.float64), normal, float(area)


def evaluate_area(
    light_obj,
    vert_positions: np.ndarray,
    vert_normals: np.ndarray,
    *,
    n_samples: int = 8,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    energy = float(getattr(light_obj.data, "energy", 0.0) or 0.0)
    if rng is None:
        rng = np.random.default_rng(0)
    samples, light_normal, area = _area_surface_samples(light_obj, n_samples, rng)
    if area <= 1e-12 or samples.shape[0] == 0:
        Nv = int(vert_positions.shape[0])
        return (
            np.zeros((1, Nv), dtype=np.float64),
            np.zeros((1, Nv, 3), dtype=np.float32),
            np.zeros((1, Nv, 3), dtype=np.float32),
            np.full((1, Nv), 1e9, dtype=np.float32),
        )

    K = int(samples.shape[0])
    Nv = int(vert_positions.shape[0])
    n = vert_normals.astype(np.float64)
    p = vert_positions.astype(np.float64)

    # Per (sample, vertex): delta = sample - vertex
    s = samples.reshape(K, 1, 3)
    delta = s - p.reshape(1, Nv, 3)  # (K, Nv, 3)
    d2 = np.einsum("kij,kij->ki", delta, delta)
    # Near-field floor: THE root cause of a measured >6000 K localized FEM
    # blow-up (visionsim50/kitchen1: a floor slab's texels sitting a few cm from
    # the base of a large, floor-adjacent area light -- e.g. a floor-to-ceiling
    # window/panel light -- picked up a stratified sample landing within
    # millimetres of the receiver). Each of the K stratified samples stands in
    # for a finite sub-patch of the light's own surface (area/K, not a literal
    # point), so the point-sampled 1/d^2 estimator is only valid at ranges large
    # compared to that sub-patch's own size; a `max(d2, 1e-20)` numerical floor
    # is many orders of magnitude too small to catch a genuinely close receiver.
    # Clamp d^2 to at least the sub-patch's half-linear-size squared so a close
    # receiver saturates at a bounded near-field value instead of diverging.
    _patch_radius = 0.5 * math.sqrt(area / float(K))
    d2 = np.maximum(d2, max(_patch_radius * _patch_radius, 1e-6))
    d = np.sqrt(d2)
    L = delta / d[..., None]  # (K, Nv, 3) unit vector from vertex to sample

    cos_recv = np.maximum(0.0, np.einsum("ij,kij->ki", n, L))
    cos_emit = np.maximum(0.0, np.einsum("j,kij->ki", -light_normal, L))

    # Per-sample contribution: energy * cos_emit * cos_recv / (π * K * d²)
    irr = (energy * cos_emit * cos_recv) / (math.pi * float(K) * d2)

    origins = np.broadcast_to(
        (vert_positions.astype(np.float32) + _RAY_EPS * vert_normals.astype(np.float32)).reshape(1, Nv, 3),
        (K, Nv, 3),
    ).copy()
    return (
        irr.astype(np.float64),
        origins,
        L.astype(np.float32),
        (d - _RAY_EPS).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def evaluate_light(
    light_obj,
    vert_positions: np.ndarray,
    vert_normals: np.ndarray,
    *,
    n_samples_for_area: int = 8,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(irradiance (S, N), origins (S, N, 3), dirs (S, N, 3),
    max_dists (S, N))`` for any light type. S = 1 for sun/point/spot,
    K for area."""
    light_type = str(getattr(light_obj.data, "type", "POINT")).upper()
    if light_type == "SUN":
        return evaluate_sun(light_obj, vert_positions, vert_normals)
    if light_type == "POINT":
        return evaluate_point(light_obj, vert_positions, vert_normals)
    if light_type == "SPOT":
        return evaluate_spot(light_obj, vert_positions, vert_normals)
    if light_type == "AREA":
        return evaluate_area(
            light_obj, vert_positions, vert_normals,
            n_samples=n_samples_for_area, rng=rng,
        )
    # Unknown / unsupported type: contribute zero.
    Nv = int(vert_positions.shape[0])
    return (
        np.zeros((1, Nv), dtype=np.float64),
        np.zeros((1, Nv, 3), dtype=np.float32),
        np.zeros((1, Nv, 3), dtype=np.float32),
        np.full((1, Nv), 1e9, dtype=np.float32),
    )
