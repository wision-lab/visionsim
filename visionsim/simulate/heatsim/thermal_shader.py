"""Thermal AOV + gray-body radiance shader support for visionsim.

Provides four entry points consumed by the thermal render pipeline:

* :func:`setup_temperature_aov`      — register a ``temperature`` value AOV on a view-layer
                                       and append ``Attribute → ShaderNodeOutputAOV`` to
                                       every material; returns the compositor socket name.
* :func:`enter_thermal_scene`        — override materials for the gray-body pass, disable
                                       lights, set a gray world; returns a restore state dict.
* :func:`restore_scene`              — undo :func:`enter_thermal_scene` from the state dict.
* :func:`stamp_default_temperatures` — stamp per-object ``heatsim_default_temperature``
                                       OBJECT-domain custom properties as a shader fallback.

"""

from __future__ import annotations

import logging
from typing import Any

from visionsim.simulate.heatsim.names import ATLAS_COVERAGE_PROP, ATLAS_IMAGE_NAME, ATLAS_UV_LAYER_NAME

try:
    import bpy  # type: ignore
except ImportError:
    bpy = None  # type: ignore

_log = logging.getLogger("rich")

# Stefan-Boltzmann constant (SI, W/m²·K⁴).  Used as a magnitude knob × radiance_scale
# in the shader; the solver uses the same value converted to W/mm².
from visionsim.simulate.heatsim.physics import AMBIENT_TEMPERATURE_K
from visionsim.simulate.heatsim.physics import STEFAN_BOLTZMANN_SI as _SIGMA_SI

# Defaults used when the scene provides no overrides.
_DEFAULT_EMISSIVITY: float = 0.9
_THERMAL_WORLD_NAME: str = "HeatSim_Thermal_World"

# Keys inside the opaque state dict returned by enter_thermal_scene.
# Object types that render geometry and can therefore carry a material patched with the
# temperature AOV chain. Only MESH supports the per-vertex `sim_temperature` attribute; the
# rest rely entirely on the stamped object-level default (see stamp_default_temperatures).
_RENDERABLE_GEOMETRY_TYPES: frozenset[str] = frozenset({"MESH", "CURVE", "SURFACE", "META", "FONT"})

_KEY_WORLD: str = "orig_world"
_KEY_MATERIAL_OVERRIDES: str = "orig_material_overrides"
_KEY_LIGHT_HIDE_RENDER: str = "light_hide_render"
_KEY_LIGHT_HIDE_VIEWPORT: str = "light_hide_viewport"
_KEY_CLAMP: str = "orig_sample_clamp"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_temperature_source_chain(nodes: Any, links: Any, new_node: Any = None, x0: float = -800.0, y0: float = 300.0) -> Any:
    """Return a temperature socket shared by the AOV and radiance shaders.

    A valid vertex attribute overrides the object default. For atlas objects,
    covered texels supply the value. Coverage is gated by both image alpha and
    an object property: a mesh without atlas UVs must not sample another tile at
    the default UV origin.
    """
    _new = new_node or nodes.new

    # -- Per-vertex temperature (zero when attribute absent) --------------------
    temp_attr = _new("ShaderNodeAttribute")
    temp_attr.attribute_name = "sim_temperature"
    temp_attr.attribute_type = "GEOMETRY"
    temp_attr.location = (x0, y0)

    # -- Per-object fallback temperature (stamped by stamp_default_temperatures) -
    default_temp_attr = _new("ShaderNodeAttribute")
    default_temp_attr.attribute_name = "heatsim_default_temperature"
    default_temp_attr.attribute_type = "OBJECT"
    default_temp_attr.location = (x0, y0 + 200.0)

    # is_valid = sim_temperature > 1.0  (a real physical Kelvin value)
    temp_is_valid = _new("ShaderNodeMath")
    temp_is_valid.operation = "GREATER_THAN"
    temp_is_valid.location = (x0 + 200.0, y0 + 100.0)
    temp_is_valid.inputs[1].default_value = 1.0
    links.new(temp_attr.outputs["Fac"], temp_is_valid.inputs[0])

    # delta = sim_temperature - default_T
    temp_delta = _new("ShaderNodeMath")
    temp_delta.operation = "SUBTRACT"
    temp_delta.location = (x0 + 200.0, y0 - 40.0)
    links.new(temp_attr.outputs["Fac"], temp_delta.inputs[0])
    links.new(default_temp_attr.outputs["Fac"], temp_delta.inputs[1])

    # scaled = is_valid × delta  (zero when the per-vertex attr is absent)
    temp_scaled = _new("ShaderNodeMath")
    temp_scaled.operation = "MULTIPLY"
    temp_scaled.location = (x0 + 360.0, y0 + 30.0)
    links.new(temp_is_valid.outputs["Value"], temp_scaled.inputs[0])
    links.new(temp_delta.outputs["Value"], temp_scaled.inputs[1])

    # T_vertex = default_T + scaled  (= sim_temperature when valid, else default_T)
    temp_vertex = _new("ShaderNodeMath")
    temp_vertex.operation = "ADD"
    temp_vertex.location = (x0 + 520.0, y0 + 130.0)
    links.new(default_temp_attr.outputs["Fac"], temp_vertex.inputs[0])
    links.new(temp_scaled.outputs["Value"], temp_vertex.inputs[1])

    # -- Atlas extension: UV -> Image Texture -> (Red, Alpha) --------------------
    atlas_uv_attr = _new("ShaderNodeAttribute")
    atlas_uv_attr.attribute_name = ATLAS_UV_LAYER_NAME
    atlas_uv_attr.attribute_type = "GEOMETRY"
    atlas_uv_attr.location = (x0, y0 - 260.0)

    atlas_tex = _new("ShaderNodeTexImage")
    atlas_tex.image = bpy.data.images.get(ATLAS_IMAGE_NAME) if bpy is not None else None
    atlas_tex.interpolation = "Linear"
    atlas_tex.extension = "CLIP"
    if atlas_tex.image is not None:
        try:
            atlas_tex.image.colorspace_settings.name = "Non-Color"
        except Exception:
            logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
    atlas_tex.location = (x0 + 200.0, y0 - 260.0)
    links.new(atlas_uv_attr.outputs["Vector"], atlas_tex.inputs["Vector"])

    atlas_red = _new("ShaderNodeSeparateColor")
    atlas_red.location = (x0 + 420.0, y0 - 200.0)
    links.new(atlas_tex.outputs["Color"], atlas_red.inputs["Color"])

    # gate = atlas alpha (this texel's own validity) × object-level atlas-participant flag
    atlas_gate_attr = _new("ShaderNodeAttribute")
    atlas_gate_attr.attribute_name = ATLAS_COVERAGE_PROP
    atlas_gate_attr.attribute_type = "OBJECT"
    atlas_gate_attr.location = (x0 + 200.0, y0 - 420.0)

    # Normalize the filtered value by its coverage. Without this, the invalid
    # zero pixels lower temperatures near atlas holes and tile edges.
    atlas_temperature = _new("ShaderNodeMath")
    atlas_temperature.operation = "DIVIDE"
    atlas_temperature.location = (x0 + 550.0, y0 - 210.0)
    links.new(atlas_red.outputs["Red"], atlas_temperature.inputs[0])
    links.new(atlas_tex.outputs["Alpha"], atlas_temperature.inputs[1])

    # Uncovered pixels use the object fallback temperature.
    atlas_alpha_valid = _new("ShaderNodeMath")
    atlas_alpha_valid.operation = "GREATER_THAN"
    atlas_alpha_valid.location = (x0 + 300.0, y0 - 340.0)
    atlas_alpha_valid.inputs[1].default_value = 0.5
    links.new(atlas_tex.outputs["Alpha"], atlas_alpha_valid.inputs[0])

    atlas_gate = _new("ShaderNodeMath")
    atlas_gate.operation = "MULTIPLY"
    atlas_gate.location = (x0 + 420.0, y0 - 380.0)
    links.new(atlas_alpha_valid.outputs["Value"], atlas_gate.inputs[0])
    links.new(atlas_gate_attr.outputs["Fac"], atlas_gate.inputs[1])

    # T_effective = Mix(Factor=gate, A=T_vertex, B=atlas temperature)
    mix = _new("ShaderNodeMix")
    mix.data_type = "FLOAT"
    mix.location = (x0 + 680.0, y0 - 100.0)
    links.new(atlas_gate.outputs["Value"], mix.inputs["Factor"])
    links.new(temp_vertex.outputs["Value"], mix.inputs["A"])
    links.new(atlas_temperature.outputs["Value"], mix.inputs["B"])

    return mix.outputs["Result"]


def _build_emissivity_source_chain(nodes: Any, links: Any, x0: float, y0: float) -> Any:
    """Per-vertex emissivity socket, falling back to ``_DEFAULT_EMISSIVITY``.

    ``adapter._write_emissivity_attr`` stamps an ``emissivity`` POINT attribute on every
    object it writes back (both the per-vertex path and the constant-fill fallback), from
    the sidecar's per-slot presets where one is supplied. A mesh that never went through
    the solve has no such attribute, and Blender returns 0.0 for a missing float
    attribute -- which would mean "perfect mirror, emits nothing". So a value of exactly
    0 is treated as absent and replaced by the default.

    Emissivity is what an LWIR camera actually distinguishes: the preset library spans
    0.05 (polished aluminium) to 0.98 (skin), and at 350 K that is the difference between
    42 and 750 W/m^2 emitted from surfaces at the same physical temperature. Baking one
    constant into the mix factor made every material in the scene radiate identically.

    Returns:
        The output socket (float ``Value``) carrying the effective per-pixel emissivity.
    """
    eps_attr = nodes.new("ShaderNodeAttribute")
    eps_attr.attribute_name = "emissivity"
    eps_attr.attribute_type = "GEOMETRY"
    eps_attr.location = (x0, y0)

    # is_valid = emissivity > 0  (a missing attribute reads as 0.0)
    eps_valid = nodes.new("ShaderNodeMath")
    eps_valid.operation = "GREATER_THAN"
    eps_valid.location = (x0 + 200.0, y0 + 100.0)
    eps_valid.inputs[1].default_value = 0.0
    links.new(eps_attr.outputs["Fac"], eps_valid.inputs[0])

    # delta = emissivity - default
    eps_delta = nodes.new("ShaderNodeMath")
    eps_delta.operation = "SUBTRACT"
    eps_delta.location = (x0 + 200.0, y0 - 40.0)
    links.new(eps_attr.outputs["Fac"], eps_delta.inputs[0])
    eps_delta.inputs[1].default_value = _DEFAULT_EMISSIVITY

    # effective = default + is_valid * delta   (branch-free select)
    eps_eff = nodes.new("ShaderNodeMath")
    eps_eff.operation = "MULTIPLY_ADD"
    eps_eff.location = (x0 + 400.0, y0)
    links.new(eps_delta.outputs["Value"], eps_eff.inputs[0])
    links.new(eps_valid.outputs["Value"], eps_eff.inputs[1])
    eps_eff.inputs[2].default_value = _DEFAULT_EMISSIVITY

    # Clamp to [0, 1]; a sidecar cannot produce an out-of-range value, but a hand-edited
    # attribute could, and a negative mix factor is meaningless.
    eps_clamped = nodes.new("ShaderNodeClamp")
    eps_clamped.location = (x0 + 600.0, y0)
    eps_clamped.inputs["Min"].default_value = 0.0
    eps_clamped.inputs["Max"].default_value = 1.0
    links.new(eps_eff.outputs["Value"], eps_clamped.inputs["Value"])

    atlas_uv = nodes.new("ShaderNodeAttribute")
    atlas_uv.attribute_name = ATLAS_UV_LAYER_NAME
    atlas_uv.attribute_type = "GEOMETRY"
    atlas_tex = nodes.new("ShaderNodeTexImage")
    atlas_tex.image = bpy.data.images.get(ATLAS_IMAGE_NAME) if bpy is not None else None
    atlas_tex.interpolation = "Closest"
    atlas_tex.extension = "CLIP"
    links.new(atlas_uv.outputs["Vector"], atlas_tex.inputs["Vector"])
    atlas_channels = nodes.new("ShaderNodeSeparateColor")
    links.new(atlas_tex.outputs["Color"], atlas_channels.inputs["Color"])

    coverage = nodes.new("ShaderNodeAttribute")
    coverage.attribute_name = ATLAS_COVERAGE_PROP
    coverage.attribute_type = "OBJECT"
    alpha_valid = nodes.new("ShaderNodeMath")
    alpha_valid.operation = "GREATER_THAN"
    alpha_valid.inputs[1].default_value = 0.5
    links.new(atlas_tex.outputs["Alpha"], alpha_valid.inputs[0])
    gate = nodes.new("ShaderNodeMath")
    gate.operation = "MULTIPLY"
    links.new(alpha_valid.outputs["Value"], gate.inputs[0])
    links.new(coverage.outputs["Fac"], gate.inputs[1])
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "FLOAT"
    links.new(gate.outputs["Value"], mix.inputs["Factor"])
    links.new(eps_clamped.outputs["Result"], mix.inputs["A"])
    links.new(atlas_channels.outputs["Green"], mix.inputs["B"])
    return mix.outputs["Result"]


def _build_gray_body_material(radiance_scale: float) -> Any:
    """Create (or return a cached) gray-body emission material for thermal rendering.

    Shader graph — gray-body Kirchhoff (ε + ρ = 1) with Stefan-Boltzmann T⁴ emission:

        T_eff ─→ POWER(4) ─→ MUL(σ) ─→ MUL(radiance_scale) ─→ Emission(Color=1, Str)─┐
                                                                                        ├─ Mix(Fac=1-ε) ─→ Out
                                                               Diffuse(Color=1, R=0) ──┘

    ``T_eff`` is read from per-vertex ``sim_temperature`` (falling back to the per-object
    ``heatsim_default_temperature`` custom property when the attribute is absent).
    Emissivity is read per-vertex from the ``emissivity`` attribute, falling back to
    ``_DEFAULT_EMISSIVITY`` where that attribute is absent.

    Args:
        radiance_scale: Scalar multiplier after σ·T⁴ — tune this to adjust rendered
            radiance brightness relative to physical W/m² magnitudes.

    Returns:
        A ``bpy.types.Material`` configured for gray-body thermal rendering.
    """
    mat_name = f"HeatSim_ThermalShader_vs_{radiance_scale:.6g}"
    existing = bpy.data.materials.get(mat_name)
    if existing is not None:
        return existing

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # -- Temperature source: vertex path + atlas mix (see _build_temperature_source_chain) --
    temp_effective_socket = _build_temperature_source_chain(nodes, links, x0=-800.0, y0=300.0)

    # -- Stefan-Boltzmann T⁴ chain ---------------------------------------------
    temp_pow4 = nodes.new("ShaderNodeMath")
    temp_pow4.operation = "POWER"
    temp_pow4.location = (-100.0, 430.0)
    temp_pow4.inputs[1].default_value = 4.0
    links.new(temp_effective_socket, temp_pow4.inputs[0])

    sigma_mul = nodes.new("ShaderNodeMath")
    sigma_mul.operation = "MULTIPLY"
    sigma_mul.location = (80.0, 430.0)
    sigma_mul.inputs[1].default_value = _SIGMA_SI
    links.new(temp_pow4.outputs["Value"], sigma_mul.inputs[0])

    scale_mul = nodes.new("ShaderNodeMath")
    scale_mul.operation = "MULTIPLY"
    scale_mul.location = (260.0, 430.0)
    scale_mul.inputs[1].default_value = float(radiance_scale)
    links.new(sigma_mul.outputs["Value"], scale_mul.inputs[0])

    # -- Gray-body shader: Mix(Fac=1-ε, Emission, Diffuse) ----------------------
    # Cycles Mix Shader: out = (1-Fac)·A + Fac·B
    # With Fac=1-ε, A=Emission, B=Diffuse: out = ε·σT⁴·scale + (1-ε)·L_in  ✓
    emission = nodes.new("ShaderNodeEmission")
    emission.location = (440.0, 430.0)
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    links.new(scale_mul.outputs["Value"], emission.inputs["Strength"])

    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.location = (440.0, 200.0)
    diffuse.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    diffuse.inputs["Roughness"].default_value = 0.0  # Lambertian

    # Fac = 1 - ε so the Emission slot gets weight ε and Diffuse gets weight 1-ε.
    # ε is per-vertex, not a scene-wide constant -- see _build_emissivity_source_chain.
    eps_socket = _build_emissivity_source_chain(nodes, links, x0=-800.0, y0=-260.0)
    one_minus_eps = nodes.new("ShaderNodeMath")
    one_minus_eps.operation = "SUBTRACT"
    one_minus_eps.location = (440.0, 60.0)
    one_minus_eps.inputs[0].default_value = 1.0
    links.new(eps_socket, one_minus_eps.inputs[1])

    mix_shader = nodes.new("ShaderNodeMixShader")
    mix_shader.location = (640.0, 340.0)
    mix_shader.inputs["Fac"].default_value = 1.0 - _DEFAULT_EMISSIVITY
    links.new(one_minus_eps.outputs["Value"], mix_shader.inputs["Fac"])
    links.new(emission.outputs["Emission"], mix_shader.inputs[1])
    links.new(diffuse.outputs["BSDF"], mix_shader.inputs[2])

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (840.0, 340.0)
    links.new(mix_shader.outputs["Shader"], out.inputs["Surface"])

    mat["heatsim_thermal_radiance_scale"] = float(radiance_scale)
    mat["heatsim_thermal_default_emissivity"] = _DEFAULT_EMISSIVITY
    return mat


def _append_temperature_aov_nodes(mat: Any, aov_name: str) -> None:
    """Append a temperature value chain → ``ShaderNodeOutputAOV(aov_name)`` to *mat*.

    The AOV mirrors the gray-body shader's is-valid/default blend rather than wiring
    ``sim_temperature`` straight through: where the per-vertex ``sim_temperature``
    attribute is missing/invalid (≤ 1 K) the AOV reports the per-object
    ``heatsim_default_temperature`` instead of 0 K, so ``temperature/`` and
    ``thermal_radiance/`` agree for un-simulated meshes.

        T_eff = default_T + (sim_temperature > 1) · (sim_temperature − default_T)

    Idempotent: skips materials that already have an OutputAOV node with the same name.
    """
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Skip if already wired.
    for node in nodes:
        if node.type == "OUTPUT_AOV" and (getattr(node, "aov_name", None) == aov_name or node.name == aov_name):
            return

    # Track nodes we add so we can clean up on a mid-build failure.
    added: list[Any] = []

    def _new(node_type: str) -> Any:
        node = nodes.new(node_type)
        added.append(node)
        return node

    try:
        # -- Temperature source: vertex path + atlas mix (see _build_temperature_source_chain) --
        temp_effective_socket = _build_temperature_source_chain(nodes, links, new_node=_new, x0=-600.0, y0=-400.0)

        # Transparent surfaces can contribute to the AOV multiple times along
        # one ray. Keep the first surface's temperature.
        light_path = _new("ShaderNodeLightPath")
        light_path.location = (-600.0, -700.0)

        first_hit = _new("ShaderNodeMath")
        first_hit.operation = "LESS_THAN"
        first_hit.location = (-400.0, -700.0)
        first_hit.inputs[1].default_value = 0.5
        links.new(light_path.outputs["Transparent Depth"], first_hit.inputs[0])

        gated = _new("ShaderNodeMath")
        gated.operation = "MULTIPLY"
        gated.location = (700.0, -400.0)
        links.new(temp_effective_socket, gated.inputs[0])
        links.new(first_hit.outputs["Value"], gated.inputs[1])
        temp_effective_socket = gated.outputs["Value"]

        aov_node = _new("ShaderNodeOutputAOV")
    except Exception as exc:  # noqa: BLE001
        _log.debug("Could not build temperature AOV chain on %r: %s", mat.name, exc)
        for node in added:
            try:
                nodes.remove(node)
            except Exception:
                logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
        return

    aov_node.name = aov_name
    if hasattr(aov_node, "aov_name"):
        try:
            aov_node.aov_name = aov_name
        except Exception:
            logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
    aov_node.location = (940.0, -400.0)

    sock = aov_node.inputs.get("Value") or (aov_node.inputs[0] if aov_node.inputs else None)
    if sock is not None:
        try:
            links.new(temp_effective_socket, sock)
        except Exception as exc:  # noqa: BLE001
            _log.debug("Could not link AOV socket on %r: %s", mat.name, exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_DEFAULT_SURFACE_MATERIAL_NAME = "HeatSim_Default_Surface"


def _get_default_surface_material() -> Any:
    """A stand-in for Blender's implicit default surface, shared across the scene.

    A freshly created node-based material is already Blender's default look — a
    Principled BSDF at 0.8 grey, roughness 0.5 — which is exactly what Cycles draws
    for a mesh with no material. Assigning it therefore leaves the RGB pass looking
    the same while giving the surface a node tree the temperature AOV can hang off.
    """
    mat = bpy.data.materials.get(_DEFAULT_SURFACE_MATERIAL_NAME)
    if mat is None:
        mat = bpy.data.materials.new(_DEFAULT_SURFACE_MATERIAL_NAME)
        mat.use_nodes = True
    return mat


def _ensure_temperature_material_slots(obj: Any) -> int:
    """Give ``obj`` a material wherever it has none, returning how many were filled.

    A value AOV is emitted by shader nodes, so a mesh with no material — or with an
    empty slot — has nothing to write the ``temperature`` pass and renders as 0 K even
    though the solver produced a temperature for it. kitchen1's ``Vert.005`` is exactly
    this: 36 vertices, a valid ``sim_temperature`` attribute, and no material slot at
    all, so it came out solid black.

    Faces on an empty slot render with the default surface too, so those are filled in
    place rather than only handling the zero-slot case.
    """
    mesh = getattr(obj, "data", None)
    if mesh is None or not hasattr(mesh, "materials"):
        return 0
    try:
        if len(obj.material_slots) == 0:
            mesh.materials.append(_get_default_surface_material())
            return 1
        filled = 0
        for slot in obj.material_slots:
            if slot.material is None:
                slot.material = _get_default_surface_material()
                filled += 1
        return filled
    except Exception as exc:   # noqa: BLE001
        _log.warning("thermal: could not assign a default surface to %r: %s", obj.name, exc)
        return 0


def setup_temperature_aov(scene: Any, view_layer: Any) -> str:
    """Register a ``temperature`` value AOV on *view_layer* and wire it into scene materials.

    For every mesh material in the scene, appends an
    ``Attribute("sim_temperature") → ShaderNodeOutputAOV("temperature")`` chain so that
    Cycles writes raw temperature values into the AOV render pass.

    Args:
        scene: The Blender scene (``bpy.types.Scene``).
        view_layer: The view layer on which to register the AOV (``bpy.types.ViewLayer``).

    Returns:
        The compositor socket name ``"temperature"`` — wire this to the
        ``CompositorNodeOutputFile`` input for the temperature render pass.
    """
    aov_name = "temperature"

    # 1. Register the AOV on the view layer (idempotent).
    existing_names: set[str] = set()
    try:
        existing_names = {a.name for a in view_layer.aovs}
    except Exception:
        logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
    if aov_name not in existing_names:
        try:
            aov = view_layer.aovs.add()
            aov.name = aov_name
            if hasattr(aov, "type"):
                try:
                    aov.type = "VALUE"
                except Exception:  # noqa: BLE001
                    try:
                        aov.type = "FLOAT"
                    except Exception:
                        logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
        except Exception as exc:  # noqa: BLE001
            _log.warning("Could not register temperature AOV on view layer: %s", exc)

    # 2. Append Attribute → OutputAOV chain to every material-using mesh in the scene.
    #    Meshes with no usable material get one first, else they have no shader to carry
    #    the AOV and Cycles renders them as 0 K (see _ensure_temperature_material_slots).
    patched = 0
    filled = 0
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        filled += _ensure_temperature_material_slots(obj)
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or not mat.use_nodes:
                continue
            _append_temperature_aov_nodes(mat, aov_name)
            patched += 1

    _log.debug(
        "setup_temperature_aov: AOV %r registered; %d material slot(s) patched, %d filled with the default surface",
        aov_name, patched, filled,
    )
    return aov_name


def enter_thermal_scene(scene: Any, *, radiance_scale: float) -> dict:
    """Swap *scene* into thermal rendering state for a gray-body radiance pass.

    Overrides view-layer materials with the gray-body emission material,
    hides all light objects from render, and sets a uniform-background thermal world.
    Returns an opaque state dict; pass it to :func:`restore_scene` to undo all changes.

    Args:
        scene: The Blender scene.
        radiance_scale: Multiplier applied after σ·T⁴; controls the brightness of the
            rendered radiance image relative to physical W/m² values.

    Returns:
        An opaque state dict for passing to :func:`restore_scene`.

    Note:
        Self-restoring: the ``state`` dict is populated *as* the scene is mutated,
        and any exception during setup triggers a full :func:`restore_scene`
        rollback before the error is re-raised.  This guarantees the scene is never
        left half-swapped even though the caller has not yet received ``state``.
    """
    state: dict[str, Any] = {}

    # Bind the (mutable) record containers into ``state`` up-front so a rollback
    # during the loops below sees everything recorded so far.
    material_overrides: dict[str, Any] = {}
    hide_render: dict[str, bool] = {}
    hide_viewport: dict[str, bool] = {}
    state[_KEY_MATERIAL_OVERRIDES] = material_overrides
    state[_KEY_LIGHT_HIDE_RENDER] = hide_render
    state[_KEY_LIGHT_HIDE_VIEWPORT] = hide_viewport

    try:
        thermal_mat = _build_gray_body_material(radiance_scale)

        # Render overrides preserve face indices, shared meshes and object-linked slots.
        for view_layer in scene.view_layers:
            material_overrides[view_layer.name] = view_layer.material_override
            view_layer.material_override = thermal_mat

        # -- Save and disable all lights (hide from render + viewport) ----------
        for obj in scene.objects:
            if obj.type == "LIGHT":
                hide_render[obj.name] = bool(obj.hide_render)
                hide_viewport[obj.name] = bool(obj.hide_viewport)
                obj.hide_render = True
                obj.hide_viewport = True


        # RGB scene clamps can truncate the larger radiometric values in this pass.
        # Save both clamp settings for restoration after rendering.
        cy = getattr(scene, "cycles", None)
        if cy is not None:
            saved_clamp: dict[str, Any] = {}
            for attr in ("sample_clamp_direct", "sample_clamp_indirect"):
                if hasattr(cy, attr):
                    saved_clamp[attr] = getattr(cy, attr)
                    try:
                        setattr(cy, attr, 0.0)  # 0 == disabled in Cycles
                    except Exception as exc:   # noqa: BLE001
                        _log.debug("Could not clear cycles.%s: %s", attr, exc)
                        saved_clamp.pop(attr, None)
            if saved_clamp:
                _log.info("thermal: cleared Cycles sample clamps for the radiance pass (%s)",
                          ", ".join(f"{k}={v}" for k, v in saved_clamp.items()))
            state[_KEY_CLAMP] = saved_clamp

        # -- Save world and replace with a uniform gray thermal world -----------
        orig_world = scene.world
        state[_KEY_WORLD] = orig_world.name if orig_world is not None else None

        # Match the gray-body emission units in the reflected world term.
        # Include ambient temperature and radiance_scale in the cached world name.
        _ambient_radiance = _SIGMA_SI * (AMBIENT_TEMPERATURE_K**4) * float(radiance_scale)
        world_name = f"{_THERMAL_WORLD_NAME}_{AMBIENT_TEMPERATURE_K:.6g}K_vs_{radiance_scale:.6g}"
        thermal_world = bpy.data.worlds.get(world_name)
        if thermal_world is None:
            thermal_world = bpy.data.worlds.new(world_name)
            thermal_world.use_nodes = True
            wnodes = thermal_world.node_tree.nodes
            wlinks = thermal_world.node_tree.links
            wnodes.clear()
            bg = wnodes.new("ShaderNodeBackground")
            bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
            bg.inputs["Strength"].default_value = _ambient_radiance
            wout = wnodes.new("ShaderNodeOutputWorld")
            wlinks.new(bg.outputs["Background"], wout.inputs["Surface"])
        scene.world = thermal_world
    except Exception:
        # Roll back every change recorded so far, then re-raise so the caller sees
        # the original failure (with the scene already restored, not half-swapped).
        restore_scene(scene, state)
        raise

    return state


def restore_scene(scene: Any, state: dict) -> None:
    """Restore *scene* to the state captured by :func:`enter_thermal_scene`.

    Args:
        scene: The Blender scene (must be the same scene passed to
            :func:`enter_thermal_scene`).
        state: The opaque dict returned by :func:`enter_thermal_scene`.
    """
    # -- Restore view-layer material overrides --------------------------------
    material_overrides = state.get(_KEY_MATERIAL_OVERRIDES, {})
    for view_layer in scene.view_layers:
        if view_layer.name in material_overrides:
            view_layer.material_override = material_overrides[view_layer.name]

    # -- Restore Cycles sample clamps -------------------------------------------
    cy = getattr(scene, "cycles", None)
    if cy is not None:
        for attr, value in (state.get(_KEY_CLAMP, {}) or {}).items():
            try:
                setattr(cy, attr, value)
            except Exception as exc:   # noqa: BLE001
                _log.debug("Could not restore cycles.%s: %s", attr, exc)

    # -- Restore light visibility -----------------------------------------------
    hide_render: dict[str, bool] = state.get(_KEY_LIGHT_HIDE_RENDER, {})
    hide_viewport: dict[str, bool] = state.get(_KEY_LIGHT_HIDE_VIEWPORT, {})
    for obj in scene.objects:
        if obj.type == "LIGHT":
            if obj.name in hide_render:
                obj.hide_render = hide_render[obj.name]
            if obj.name in hide_viewport:
                obj.hide_viewport = hide_viewport[obj.name]

    # -- Restore world ----------------------------------------------------------
    orig_world_name = state.get(_KEY_WORLD)
    if orig_world_name is not None:
        w = bpy.data.worlds.get(orig_world_name)
        if w is not None:
            scene.world = w
    elif _KEY_WORLD in state:
        # World was None before enter_thermal_scene.
        scene.world = None  # type: ignore[assignment]


def stamp_default_temperatures(scene: Any, *, default_K: float) -> None:
    """Stamp the ``heatsim_default_temperature`` custom property on every mesh object.

    The gray-body emission shader reads this as a fallback temperature wherever the
    per-vertex ``sim_temperature`` attribute is absent (objects not participating in
    the FEM solve, or fluid meshes regenerated each frame by Mantaflow).  Cheap to
    call every frame.

    Args:
        scene: The Blender scene.
        default_K: Fallback temperature in Kelvin to stamp on all mesh objects.
    """
    t = float(default_K)
    for obj in scene.objects:
        # Shared materials can patch non-mesh geometry too. Give those objects a
        # fallback temperature because they cannot carry the mesh point attribute.
        if obj.type not in _RENDERABLE_GEOMETRY_TYPES:
            continue
        obj["heatsim_default_temperature"] = t
