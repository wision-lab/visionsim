# Vendored from heat-sim-blender:addon/lib/sh9_sky.py @ e5b4afe
"""SH9 sky-dome prefilter for the Direct Kernel irradiance source.

Implements the Ramamoorthi & Hanrahan 2001 "An Efficient Representation
for Irradiance Environment Maps" technique:

  1. Prefilter the world's background environment into 9 spherical
     harmonic (SH) coefficients per RGB channel — done once per world
     (cached, invalidated on world hash change).
  2. Evaluate per-vertex irradiance E(n) from the SH coefficients with
     a closed-form quadratic in the vertex normal — no rays required.

Two world-shader cases are supported:

  - **Solid background color** (a `Background` node feeding the world
    output, with an RGBA color and a strength multiplier). The whole
    sphere is treated as constant radiance — only the SH00 band is
    nonzero.
  - **Environment texture** (an `Image Texture` or `Environment
    Texture` feeding `Background`, equirectangular layout). We sum
    over pixels weighted by sin(θ) and the SH basis.

Anything more exotic (Sky shaders, mix nodes, color-ramps) falls back
to using the `world.color` as a solid color.

For thermal use, RGB irradiance is collapsed to a scalar via the mean
of the channels — the FEM solver doesn't do spectral transport.

Public API:

    coeffs_rgb = prefilter_world(world)            # (9, 3) float64
    irr = eval_per_vertex(normals, coeffs_rgb)     # (N,) W/m²

The orchestrator (``irradiance_kernel``) caches the (9, 3) coefficients
keyed by ``world_hash(world)`` so re-baking is free for an unchanged
world.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Optional, Tuple

import numpy as np

_log = logging.getLogger("rich")

# Ramamoorthi-Hanrahan diffuse reconstruction constants
# (table 1 of "An Efficient Representation for Irradiance Environment Maps").
_C1 = 0.429043
_C2 = 0.511664
_C3 = 0.743125
_C4 = 0.886227
_C5 = 0.247708

# Real SH basis function coefficients for l ∈ {0, 1, 2}.
# Order: [Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20, Y21, Y22]
_Y_NORM = np.array([
    0.282094791773878,   # Y00 = 1/(2√π)
    0.488602511902920,   # Y1-1 = √(3/(4π)) y
    0.488602511902920,   # Y10  = √(3/(4π)) z
    0.488602511902920,   # Y11  = √(3/(4π)) x
    1.092548430592079,   # Y2-2 = √(15/(4π)) xy
    1.092548430592079,   # Y2-1 = √(15/(4π)) yz
    0.315391565252520,   # Y20  = √(5/(16π))(3z²−1)
    1.092548430592079,   # Y21  = √(15/(4π)) xz
    0.546274215296039,   # Y22  = √(15/(16π))(x²−y²)
], dtype=np.float64)


# ---------------------------------------------------------------------------
# World introspection / hashing
# ---------------------------------------------------------------------------


def world_hash(world) -> str:
    """A short stable hash for the world. Used as cache key. Changes
    when shader nodes, image filepath, or strength change. We hash
    enough state to detect the common edits — a subtle node-graph
    mutation that doesn't touch any of the inspected fields will not
    invalidate the cache (acceptable: the user can press Reset)."""
    if world is None:
        return "no_world"
    h = hashlib.md5()
    h.update((world.name or "").encode("utf-8"))
    color = tuple(float(c) for c in (getattr(world, "color", None) or (0.0, 0.0, 0.0)))
    h.update(repr(color).encode("utf-8"))
    if getattr(world, "use_nodes", False) and world.node_tree is not None:
        for node in world.node_tree.nodes:
            h.update(node.bl_idname.encode("utf-8"))
            h.update((node.name or "").encode("utf-8"))
            for inp in node.inputs:
                if hasattr(inp, "default_value"):
                    try:
                        h.update(repr(tuple(inp.default_value)).encode("utf-8"))
                    except TypeError:
                        h.update(repr(float(inp.default_value)).encode("utf-8"))
            img = getattr(node, "image", None)
            if img is not None:
                h.update((img.filepath or img.name or "").encode("utf-8"))
                h.update(repr((int(img.size[0]), int(img.size[1]))).encode("utf-8"))
    return h.hexdigest()[:16]


def _find_background_node(world):
    """Walk the world node tree for the Background node feeding the
    World Output. Returns the Background node or None."""
    if not getattr(world, "use_nodes", False) or world.node_tree is None:
        return None
    nt = world.node_tree
    out = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputWorld" and n.is_active_output), None)
    if out is None:
        out = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputWorld"), None)
    if out is None:
        return None
    surface_input = out.inputs.get("Surface")
    if surface_input is None or not surface_input.is_linked:
        return None
    src = surface_input.links[0].from_node
    if src.bl_idname == "ShaderNodeBackground":
        return src
    return None


def _find_env_image_node(background_node):
    """If the Background's Color input is fed by an Image/Environment
    Texture, return that node; else None."""
    if background_node is None:
        return None
    color_input = background_node.inputs.get("Color")
    if color_input is None or not color_input.is_linked:
        return None
    src = color_input.links[0].from_node
    if src.bl_idname in ("ShaderNodeTexEnvironment", "ShaderNodeTexImage"):
        return src
    return None


# ---------------------------------------------------------------------------
# Solid color path (no env map)
# ---------------------------------------------------------------------------


def _solid_color_sh(color_rgb: np.ndarray, strength: float) -> np.ndarray:
    """A constant-radiance sky has nonzero only in the SH00 band:
    L00 = (color · strength) · ∫ Y00 dΩ = (color · strength) · √(4π) · Y00 = color · strength · 2√π · 1/(2√π) · 4π / (2√π · 4π)

    Simpler: ∫ L Y00 dΩ over the sphere with constant L = c·s gives
    L00 = c·s · √(4π) (since Y00 = 1/(2√π) and area = 4π)."""
    c = np.asarray(color_rgb, dtype=np.float64).reshape(3) * float(strength)
    coeffs = np.zeros((9, 3), dtype=np.float64)
    coeffs[0] = c * math.sqrt(4.0 * math.pi)
    return coeffs


# ---------------------------------------------------------------------------
# HDRI prefilter (equirectangular)
# ---------------------------------------------------------------------------


def _read_image_pixels(image) -> Optional[np.ndarray]:
    """Return image pixels as (H, W, 4) float32 in linear space, or
    None if the image has no pixel data loaded."""
    if image is None:
        return None
    try:
        w, h = int(image.size[0]), int(image.size[1])
        if w <= 0 or h <= 0:
            return None
        n_channels = int(image.channels) if image.channels else 4
        flat = np.asarray(image.pixels[:], dtype=np.float32)
        if flat.size != w * h * n_channels:
            return None
        arr = flat.reshape(h, w, n_channels)
        # Blender's pixel buffer is bottom-up; flip so row 0 = top.
        return np.flipud(arr)
    except Exception:  # noqa: BLE001
        return None


def _prefilter_equirectangular(pixels_rgba: np.ndarray, strength: float) -> np.ndarray:
    """Sum-quadrature SH9 prefilter for an equirectangular image.

    pixels_rgba: (H, W, ≥3) float linear radiance (W/m²/sr scale unknown,
    inherits image authoring convention). We treat values as relative
    radiance and let ``strength`` apply.

    Sampling matches Blender / Cycles' equirectangular env-texture mapping
    (Z-up world space) verbatim. From Cycles source
    (``direction_to_equirectangular``):

        u = -atan2(d.y, d.x) / (2π) + 0.5    # note the NEGATIVE atan2
        v =  1 - acos(d.z) / π

    Inverting (texel center (j, i) on a top-down image after flipud):

        theta = (j + 0.5) · π / H            → row 0 (top) = world +Z
        phi   = π - (i + 0.5) · 2π / W       → col W/2     = world +X

    Direction:    d = (sin θ cos φ, sin θ sin φ, cos θ)

    The SH basis below uses standard math convention (Y_10 ∝ z), so the
    coefficients can be evaluated against world-space normals directly
    without any axis swap.

    Earlier versions of this prefilter used ``phi = (i+0.5)·2π/W - π``
    (Cycles' phi negated), which silently mirrored the y-component of
    every coefficient. White-sky tests still passed (L00 has no
    direction), but Cycles vs kernel disagreed by exactly the y-flip
    on every HDRI scene. See ``proposals/phase-3-sky-env-texture-debug.md``.
    """
    H, W = pixels_rgba.shape[:2]
    rgb = pixels_rgba[..., :3].astype(np.float64) * float(strength)

    j = np.arange(H, dtype=np.float64)
    i = np.arange(W, dtype=np.float64)
    theta = (j + 0.5) * math.pi / float(H)
    phi = math.pi - (i + 0.5) * 2.0 * math.pi / float(W)

    sin_th = np.sin(theta)            # (H,)
    cos_th = np.cos(theta)            # (H,)
    cos_ph = np.cos(phi)              # (W,)
    sin_ph = np.sin(phi)              # (W,)

    # Z-up direction vector per pixel.
    x = (sin_th[:, None] * cos_ph[None, :])  # (H, W)
    y = (sin_th[:, None] * sin_ph[None, :])  # (H, W)
    z = np.broadcast_to(cos_th[:, None], (H, W))

    Y = np.empty((9, H, W), dtype=np.float64)
    Y[0] = _Y_NORM[0]
    Y[1] = _Y_NORM[1] * y
    Y[2] = _Y_NORM[2] * z
    Y[3] = _Y_NORM[3] * x
    Y[4] = _Y_NORM[4] * x * y
    Y[5] = _Y_NORM[5] * y * z
    Y[6] = _Y_NORM[6] * (3.0 * z * z - 1.0)
    Y[7] = _Y_NORM[7] * x * z
    Y[8] = _Y_NORM[8] * (x * x - y * y)

    # Solid angle per pixel: dΩ = sin θ · dθ · dφ
    dtheta = math.pi / float(H)
    dphi = 2.0 * math.pi / float(W)
    dOmega = (sin_th[:, None] * (dtheta * dphi)).astype(np.float64)  # (H, 1)
    dOmega = np.broadcast_to(dOmega, (H, W))

    coeffs = np.zeros((9, 3), dtype=np.float64)
    weight = (Y * dOmega[None, :, :])  # (9, H, W)
    for c in range(3):
        # einsum gives (9,) per channel
        coeffs[:, c] = np.einsum("ihw,hw->i", weight, rgb[..., c])
    return coeffs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prefilter_world(world) -> np.ndarray:
    """Returns (9, 3) RGB SH coefficients describing the world's
    diffuse irradiance environment. Falls back to ``world.color``
    when the node graph isn't a recognized pattern."""
    if world is None:
        return np.zeros((9, 3), dtype=np.float64)

    bg = _find_background_node(world)
    if bg is not None:
        try:
            strength_input = bg.inputs.get("Strength")
            strength = float(strength_input.default_value) if strength_input is not None else 1.0
        except Exception:  # noqa: BLE001
            strength = 1.0

        env = _find_env_image_node(bg)
        if env is not None and getattr(env, "image", None) is not None:
            # Warn (not fail) if the Strength socket is driven by a node —
            # we read default_value, which is stale in that case.
            try:
                strength_input = bg.inputs.get("Strength")
                if strength_input is not None and strength_input.is_linked:
                    _log.debug(
                        "[HeatSim:Kernel] WARNING: world '%s' has "
                        "a linked Background Strength input; SH9 prefilter "
                        "uses default_value=%s, which may be stale.",
                        world.name, strength,
                    )
            except Exception:
                pass
            pixels = _read_image_pixels(env.image)
            if pixels is not None:
                return _prefilter_equirectangular(pixels, strength)
            # Fall through to color fallback if the image has no data.
            _log.debug(
                "[HeatSim:Kernel] WARNING: world '%s' env texture '%s' has no loaded pixel data; "
                "falling back to solid-color sky. Try image.reload().",
                world.name, getattr(env.image, "name", "?"),
            )

        try:
            color_input = bg.inputs.get("Color")
            color = (
                tuple(float(c) for c in color_input.default_value[:3])
                if color_input is not None and not color_input.is_linked
                else (0.0, 0.0, 0.0)
            )
        except Exception:  # noqa: BLE001
            color = (0.0, 0.0, 0.0)
        return _solid_color_sh(np.array(color), strength)

    # No usable node graph: use the legacy world.color (Blender 5.x still
    # exposes this, defaults to (0.05, 0.05, 0.05)).
    color = tuple(float(c) for c in (getattr(world, "color", None) or (0.0, 0.0, 0.0)))
    return _solid_color_sh(np.array(color), 1.0)


def eval_per_vertex(
    normals: np.ndarray,
    coeffs_rgb: np.ndarray,
    *,
    rgb_to_scalar: str = "mean",
) -> np.ndarray:
    """Evaluate per-vertex irradiance E(n) from SH9 coefficients.

    Uses the Ramamoorthi-Hanrahan closed-form quadratic. The input
    ``coeffs_rgb`` is (9, 3) and the output is (N,) scalar W/m² (the
    thermal solver is monochromatic).

    rgb_to_scalar:
        "mean"      → average of R/G/B (good neutral default).
        "luminance" → 0.2126·R + 0.7152·G + 0.0722·B (Rec. 709).
    """
    n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    # Normalize defensively — vertex normals from Blender can be slightly
    # off unit length on degenerate meshes.
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    n = n / norms

    # Both the prefilter and this eval use the standard math SH basis
    # (Y_10 ∝ z), so the world-space (Z-up) normal components feed in
    # directly: no axis swap required.
    x = n[:, 0]
    y = n[:, 1]
    z = n[:, 2]

    c = np.asarray(coeffs_rgb, dtype=np.float64).reshape(9, 3)
    L00 = c[0]
    L1m1 = c[1]; L10 = c[2]; L11 = c[3]
    L2m2 = c[4]; L2m1 = c[5]; L20 = c[6]; L21 = c[7]; L22 = c[8]

    # E(n) per channel.
    # E = c1·L22·(x²−y²) + c3·L20·z² + c4·L00 − c5·L20
    #     + 2·c1·(L2−2·xy + L21·xz + L2−1·yz)
    #     + 2·c2·(L11·x + L1−1·y + L10·z)
    irr = (
        _C1 * L22[None, :] * (x * x - y * y)[:, None]
        + _C3 * L20[None, :] * (z * z)[:, None]
        + _C4 * L00[None, :]
        - _C5 * L20[None, :]
        + 2.0 * _C1 * (
            L2m2[None, :] * (x * y)[:, None]
            + L21[None, :] * (x * z)[:, None]
            + L2m1[None, :] * (y * z)[:, None]
        )
        + 2.0 * _C2 * (
            L11[None, :] * x[:, None]
            + L1m1[None, :] * y[:, None]
            + L10[None, :] * z[:, None]
        )
    )
    irr = np.maximum(irr, 0.0)  # negative reconstruction lobes get clamped.

    if rgb_to_scalar == "luminance":
        weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
        return (irr @ weights).astype(np.float64)
    # default mean
    return irr.mean(axis=1).astype(np.float64)


def coeffs_have_energy(coeffs_rgb: np.ndarray) -> bool:
    """Cheap test for whether the prefilter produced a usable sky.
    Skips per-vertex SH eval when the world contributes nothing."""
    return bool(np.any(np.abs(coeffs_rgb) > 1e-9))
