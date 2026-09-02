"""Cycles COLOR albedo bake, ported from the heat-sim-blender addon.

Vendored so VisionSim's Direct-Kernel albedo path (``irradiance_kernel.
_bake_vertex_albedo_via_cycles`` → this module's ``bake_albedo_map``) can
resolve per-vertex reflectivity without depending on the installed addon.

Only the albedo (DIFFUSE/COLOR) bake is ported; the Cycles *irradiance* bake
is intentionally not vendored (VisionSim uses the analytic Direct-Kernel for
irradiance). ``bake_albedo_map`` returns a ``BakedFluxMap`` whose ``.pixels``
is an ``(H, W, 3)`` float64 array — the contract consumed by
``irradiance_kernel._bake_vertex_albedo_via_cycles``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import bpy
import numpy as np

from .constants import ALBEDO_LAYER_NAME, BAKE_UV_LAYER_NAME, CYCLES_LOUT_TO_IRRADIANCE, IRRADIANCE_LAYER_NAME
from .uv_utils import restore_uv_states, snapshot_uv_states


@dataclass
class BakedFluxMap:
    """Container for baked flux data coming from an image texture."""

    pixels: np.ndarray  # shape (H, W, 3) in linear space
    vertex_flux: np.ndarray  # per-vertex luminance * scale
    width: int
    height: int


def _ensure_uv_layer(obj):
    """Return an active UV layer or warn if missing."""
    mesh = obj.data
    if not mesh.uv_layers:
        scene = bpy.context.scene
        view_layer = bpy.context.view_layer
        prev_active = view_layer.objects.active
        prev_selection = [o for o in scene.objects if o.select_get()]
        try:
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            view_layer.objects.active = obj
            if obj.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            # Using 1.6% island margin to prevent texture bleeding artifacts
            bpy.ops.uv.smart_project(island_margin=0.016)
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception as exc:  # keep running but warn
            warnings.warn(f"HeatSim: Failed to auto-unwrap UVs for {obj.name}: {exc}")
        finally:
            # Restore selection and active object/mode. Force OBJECT mode FIRST and
            # guard the operators: if smart_project raised above it left the context
            # in EDIT mode, and an unguarded object.select_all here would poll()-fail
            # and abort the whole scene's bake instead of just skipping this object.
            if getattr(bpy.context, "mode", "OBJECT") != "OBJECT":
                try:
                    bpy.ops.object.mode_set(mode="OBJECT")
                except Exception:
                    pass
            try:
                bpy.ops.object.select_all(action="DESELECT")
            except Exception:
                pass
            for o in prev_selection:
                try:
                    o.select_set(True)
                except Exception:
                    pass
            if prev_active:
                view_layer.objects.active = prev_active

    if not mesh.uv_layers:
        warnings.warn(f"HeatSim: {obj.name} has no UV map; cannot bake irradiance texture.")
        return None
    return mesh.uv_layers.active or mesh.uv_layers[0]


def prepare_object_bake_uv(obj: bpy.types.Object) -> None:
    """
    Ensure `BAKE_UV_LAYER_NAME` exists on the object and is unwrapped for baking.

    This mirrors the shared baking pipeline, but operates on a single object:
    - Create/activate `BAKE_UV_LAYER_NAME`
    - Smart Project unwrap into that layer to get a non-overlapping 0..1 bake UVs
    """
    if obj is None or obj.type != "MESH":
        return
    mesh = obj.data
    # NOTE: test `is None`, not truthiness. `mesh.uv_layers` on a mesh carrying zero UV
    # layers is an EMPTY collection, which is falsy -- so a truthiness check bailed out on
    # exactly the meshes that need a bake UV created, while the very next block
    # (`uv_layers.new(...)` + Smart Project) exists to create one from scratch.
    #
    # The cost was severe and entirely silent. An object with no authored UVs never got
    # HeatSim_Bake_UV, so `_write_atlas_uv_layer` had no source layer, HeatSim_Atlas_UV was
    # never written, and `adapter.build_atlas_plan` demoted the object from the atlas to the
    # per-vertex path. There, if a modifier changed the vertex count (Subsurf, Solidify,
    # ...), per-vertex write-back is structurally impossible, so `write_frame_attributes`
    # constant-filled the whole object at the MEAN of its solved field.
    #
    # Measured on visionsim50/diningroom (289 objects, 85% with no authored UVs):
    # 259 selected for the atlas, 231 demoted for a missing UV layer, 229 then
    # constant-filled -- each rendering as ONE flat value. That is why every chair showed a
    # different uniform temperature, and why officebuilding's floor (Cube.006, base 68 verts
    # vs 408 evaluated) rendered as a flat 330.74 K plateau while its solved field actually
    # spanned 295.8-445.8 K with a 33 K standard deviation.
    if mesh is None or getattr(mesh, "uv_layers", None) is None:
        return
    # Skip degenerate (zero-geometry) meshes: an empty object has no albedo to
    # bake, and bpy.ops.uv.smart_project.poll() fails on it in --background mode
    # (nothing to unwrap), which would otherwise abort the whole scene's bake.
    # Mirrors the empty-mesh guard in irradiance_kernel.py.
    if len(mesh.polygons) == 0 or len(mesh.vertices) == 0:
        return

    # Ensure bake UV layer exists
    if BAKE_UV_LAYER_NAME not in mesh.uv_layers:
        try:
            mesh.uv_layers.new(name=BAKE_UV_LAYER_NAME)
        except Exception:
            return

    # Set bake UV active for unwrap + baking
    try:
        mesh.uv_layers[BAKE_UV_LAYER_NAME].active = True
        mesh.uv_layers[BAKE_UV_LAYER_NAME].active_render = True
    except Exception:
        pass

    ctx = bpy.context
    view_layer = ctx.view_layer
    prev_selection = list(getattr(ctx, "selected_objects", []) or [])
    prev_active = getattr(ctx, "active_object", None)

    try:
        # Establish a clean OBJECT-mode context before any selection operator.
        # object.select_all / mode_set poll() fail in --background when the context
        # is in EDIT mode with no valid active object (which happens on dense scenes
        # once any earlier object's unwrap misbehaves). Point 'active' at the object
        # we are about to bake first, so mode_set has a valid target to switch.
        if view_layer.objects.active is None:
            view_layer.objects.active = obj
        if getattr(bpy.context, "mode", "OBJECT") != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass

        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        view_layer.objects.active = obj

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        # Using 1.6% island margin to prevent texture bleeding artifacts.
        bpy.ops.uv.smart_project(island_margin=0.016)
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception as exc:
        # A per-object unwrap failure (smart_project.poll() and friends can fail in
        # --background on some meshes) must never abort the whole scene's bake. Warn
        # and keep whatever UVs the object already has; the albedo kernel treats a
        # missing/degenerate bake as full absorption, a documented fallback.
        warnings.warn(f"HeatSim: bake-UV prep failed for {obj.name}; using existing UVs. {exc}")
    finally:
        # Restore selection / active / mode best-effort.
        # Crucially, always return to OBJECT mode: this function enters EDIT mode
        # above, and if smart_project raised, control skipped the mode_set back to
        # OBJECT. Leaving the shared context in EDIT makes the *next* object's
        # object.select_all.poll() fail and abort the bake (the backwards
        # ``prev_mode != "OBJECT"`` guard used to skip this reset when the object
        # started, as they all do, in OBJECT mode).
        if getattr(bpy.context, "mode", "OBJECT") != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass
        try:
            bpy.ops.object.select_all(action="DESELECT")
        except Exception:
            pass
        for o in prev_selection:
            try:
                o.select_set(True)
            except Exception:
                pass
        if prev_active and prev_active in getattr(ctx, "visible_objects", []):
            try:
                view_layer.objects.active = prev_active
            except Exception:
                pass


def _ensure_bake_image(base_name: str, name_suffix: str, size: int):
    """Create or reuse a float image that receives baked data."""
    size = max(16, int(size))
    img_name = f"{base_name}_{name_suffix}"
    image = bpy.data.images.get(img_name)
    if image is None:
        image = bpy.data.images.new(img_name, width=size, height=size, alpha=False, float_buffer=True)
    else:
        if image.size[0] != size or image.size[1] != size:
            image.scale(size, size)
    # Prefer linear/non-color spaces; we treat all bake outputs numerically.
    preferred = ("Linear", "Linear Rec.709", "Non-Color")
    for name in preferred:
        if name in image.colorspace_settings.bl_rna.properties["name"].enum_items:
            image.colorspace_settings.name = name
            break
    # Clear previous contents
    if image.pixels:
        image.pixels.foreach_set([0.0] * (len(image.pixels)))
    return image


def _ensure_albedo_image(name_suffix: str, size: int):
    return _ensure_bake_image(ALBEDO_LAYER_NAME, name_suffix, size)


def _ensure_irradiance_image(name_suffix: str, size: int):
    return _ensure_bake_image(IRRADIANCE_LAYER_NAME, name_suffix, size)


def _prepare_image_nodes_for_bake(obj, image, node_name: str, node_label: str):
    """Ensure each material has an image texture node set active for baking."""
    if not obj.material_slots:
        mat = bpy.data.materials.new(name="HeatSim_Irradiance_Bake")
        mat.use_nodes = True
        obj.data.materials.append(mat)

    for slot in obj.material_slots:
        mat = slot.material
        if mat is None:
            continue
        # For shared bake, we might reuse materials if they are shared, but we need to ensure
        # the node exists in all of them.
        if not mat.use_nodes:
            mat.use_nodes = True
        nodes = mat.node_tree.nodes
        img_node = nodes.get(node_name)
        if img_node is None:
            img_node = nodes.new("ShaderNodeTexImage")
            img_node.name = node_name
            img_node.label = node_label
        img_node.image = image
        nodes.active = img_node
        for n in nodes:
            n.select = False
        img_node.select = True


@dataclass
class _BakeMaterialUVOverride:
    """
    Temporary material edits to ensure textures are sampled using the *original* UV map,
    even while the bake target UV is switched to `BAKE_UV_LAYER_NAME`.
    """

    # (obj, slot_index, original_material) entries for slots we temporarily replaced with a copy
    replaced_slots: list[tuple[bpy.types.Object, int, bpy.types.Material | None]]
    # (material, uv_node_name) for UVMap nodes we created
    created_uv_nodes: list[tuple[bpy.types.Material, str]]
    # (material, tex_node_name) for TexImage nodes we connected
    patched_tex_nodes: list[tuple[bpy.types.Material, str]]


def _pick_source_uv_for_object(obj: bpy.types.Object, uv_snapshot_map: dict[int, tuple[str | None, str | None]]) -> str | None:
    """
    Choose the UV map name that represents the object's 'real' texturing UVs.
    Prefer the snapshot's active_render, then active, then any non-bake UV layer.
    """
    state = uv_snapshot_map.get(obj.as_pointer(), (None, None))
    active_name, render_name = state
    if render_name:
        return str(render_name)
    if active_name:
        return str(active_name)
    try:
        mesh = obj.data
        if mesh and getattr(mesh, "uv_layers", None):
            for uv in mesh.uv_layers:
                if uv.name != BAKE_UV_LAYER_NAME:
                    return uv.name
            if len(mesh.uv_layers) > 0:
                return mesh.uv_layers[0].name
    except Exception:
        pass
    return None


def _apply_uv_override_to_material(mat: bpy.types.Material, uv_name: str) -> tuple[str | None, list[str]]:
    """
    Ensure every unlinked Image Texture node samples using a specific UV map.
    Returns (created_uv_node_name, patched_tex_node_names).
    """
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return None, []
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links

    # Create a deterministic UVMap node for this bake override
    uv_node_name = f"HeatSim_OrigUV__{uv_name}"
    uv_node = nodes.get(uv_node_name)
    created = False
    if uv_node is None:
        uv_node = nodes.new("ShaderNodeUVMap")
        uv_node.name = uv_node_name
        uv_node.label = "HeatSim Orig UV (Bake Override)"
        created = True
    try:
        uv_node.uv_map = uv_name
    except Exception:
        # Older Blender versions may differ; keep best-effort.
        pass

    patched = []
    for node in nodes:
        if node.type != "TEX_IMAGE":
            continue
        vec_in = node.inputs.get("Vector")
        if vec_in is None or vec_in.is_linked:
            continue
        # Wire UVMap -> Vector
        try:
            links.new(uv_node.outputs.get("UV"), vec_in)
            patched.append(node.name)
        except Exception:
            # Don't hard-fail on odd node trees
            pass

    if created:
        return uv_node_name, patched
    # Even if node existed, we still might have patched tex nodes
    return uv_node_name, patched


def _install_bake_uv_material_overrides(objects: list[bpy.types.Object], uv_snapshot) -> _BakeMaterialUVOverride:
    """
    Install per-material UVMap overrides so real materials keep sampling their textures
    with the original UV map during shared UV baking.

    Handles the tricky case where multiple objects share a material but have different
    source UV map names by temporarily copying materials per-object.
    """
    uv_snapshot_map: dict[int, tuple[str | None, str | None]] = {
        obj.as_pointer(): state for obj, state in (uv_snapshot or [])
    }

    # Build material usage: mat_ptr -> {uv_names}, plus occurrences to resolve per-object copies.
    mat_uvs: dict[int, set[str]] = {}
    occurrences: list[tuple[bpy.types.Object, int, bpy.types.Material | None, str | None]] = []
    for obj in objects:
        if obj is None or obj.type != "MESH":
            continue
        uv_name = _pick_source_uv_for_object(obj, uv_snapshot_map)
        for slot_idx, slot in enumerate(getattr(obj, "material_slots", []) or []):
            mat = getattr(slot, "material", None)
            if mat is None or not getattr(mat, "use_nodes", False) or mat.node_tree is None:
                continue
            occurrences.append((obj, slot_idx, mat, uv_name))
            if uv_name:
                mat_uvs.setdefault(mat.as_pointer(), set()).add(str(uv_name))

    replaced_slots: list[tuple[bpy.types.Object, int, bpy.types.Material | None]] = []
    created_uv_nodes: list[tuple[bpy.types.Material, str]] = []
    patched_tex_nodes: list[tuple[bpy.types.Material, str]] = []

    # Apply overrides
    for obj, slot_idx, mat, uv_name in occurrences:
        if not uv_name:
            continue
        uv_set = mat_uvs.get(mat.as_pointer(), {str(uv_name)})
        target_mat = mat

        # If the same material is used with multiple different UV map names, copy per-object.
        if len(uv_set) > 1:
            try:
                mat_copy = mat.copy()
                mat_copy.name = f"{mat.name}__HeatSimBake"
                obj.material_slots[slot_idx].material = mat_copy
                replaced_slots.append((obj, slot_idx, mat))
                target_mat = mat_copy
            except Exception:
                # If copying fails, fall back to editing the shared material (may be imperfect).
                target_mat = mat

        uv_node_name, patched = _apply_uv_override_to_material(target_mat, str(uv_name))
        if uv_node_name:
            created_uv_nodes.append((target_mat, uv_node_name))
        for tex_node_name in patched:
            patched_tex_nodes.append((target_mat, tex_node_name))

    return _BakeMaterialUVOverride(
        replaced_slots=replaced_slots,
        created_uv_nodes=created_uv_nodes,
        patched_tex_nodes=patched_tex_nodes,
    )


def _restore_bake_uv_material_overrides(state: _BakeMaterialUVOverride | None) -> None:
    if state is None:
        return

    # Remove UVMap -> TexImage links we created and delete UVMap nodes.
    # Do this before restoring original materials so we can clean up copied ones too.
    for mat, uv_node_name in state.created_uv_nodes:
        try:
            if mat is None or not mat.use_nodes or mat.node_tree is None:
                continue
            nt = mat.node_tree
            nodes = nt.nodes
            links = nt.links
            uv_node = nodes.get(uv_node_name)
            if uv_node is None:
                continue

            # Remove links from this uv node into any TEX_IMAGE Vector inputs
            for link in list(links):
                if (
                    link.from_node == uv_node
                    and link.to_node
                    and link.to_node.type == "TEX_IMAGE"
                    and getattr(link.to_socket, "name", "") == "Vector"
                ):
                    try:
                        links.remove(link)
                    except Exception:
                        pass

            # Remove the uv node itself (if still present)
            try:
                nodes.remove(uv_node)
            except Exception:
                pass
        except Exception:
            pass

    # Restore original materials in slots, and remove temporary copies if possible.
    for obj, slot_idx, original_mat in state.replaced_slots:
        try:
            slot = obj.material_slots[slot_idx]
            tmp = getattr(slot, "material", None)
            slot.material = original_mat
            # Best-effort cleanup of the temp material if it's not used elsewhere
            try:
                if tmp is not None and tmp.users == 0:
                    bpy.data.materials.remove(tmp)
            except Exception:
                pass
        except Exception:
            pass


def _image_pixels_to_rgb(image) -> np.ndarray | None:
    """Convert a Blender image to (H, W, 3) numpy array."""
    w, h = image.size
    raw = np.array(image.pixels[:], dtype=np.float64)
    expected = w * h * 4
    if raw.size != expected:
        warnings.warn(f"HeatSim: Unexpected pixel count for baked image {image.name} ({raw.size} vs {expected}).")
        return None
    rgb = raw.reshape((h * w, 4))[:, :3]
    return rgb.reshape((h, w, 3))


def _bilinear_sample(rgb_pixels: np.ndarray, width: int, height: int, uv: np.ndarray) -> float:
    """Sample luminance from an RGB image at UV coordinates using bilinear filtering."""
    u = float(np.clip(uv[0], 0.0, 1.0))
    v = float(np.clip(uv[1], 0.0, 1.0))
    x = u * (width - 1)
    y = v * (height - 1)

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)

    tx = x - x0
    ty = y - y0

    c00 = rgb_pixels[y0, x0]
    c10 = rgb_pixels[y0, x1]
    c01 = rgb_pixels[y1, x0]
    c11 = rgb_pixels[y1, x1]

    c0 = c00 * (1 - tx) + c10 * tx
    c1 = c01 * (1 - tx) + c11 * tx
    rgb = c0 * (1 - ty) + c1 * ty
    return float(rgb @ np.array((0.2126, 0.7152, 0.0722), dtype=np.float64))


def _image_to_vertex_irradiance(
    loop_vertex_indices: np.ndarray,
    uv_data: np.ndarray,
    rgb_pixels: np.ndarray,
    width: int,
    height: int,
    scale: float,
    vert_count: int,
) -> np.ndarray:
    """Accumulate image luminance to vertices via UVs."""
    accum = np.zeros(vert_count, dtype=np.float64)
    counts = np.zeros(vert_count, dtype=np.int32)
    for loop_idx, vert_idx in enumerate(loop_vertex_indices):
        uv = uv_data[loop_idx]
        flux = _bilinear_sample(rgb_pixels, width, height, uv) * float(scale)
        accum[vert_idx] += flux
        counts[vert_idx] += 1

    counts[counts == 0] = 1
    return accum / counts


def _mesh_to_sample(obj):
    """The mesh a bake's per-vertex reduction must be indexed against.

    Both bakes MUST use this. The solver builds its nodes from
    ``adapter._extract_geometry``, which reads ``obj.evaluated_get(depsgraph).data`` --
    modifiers applied. A bake that reduces against ``obj.data`` instead produces an array
    sized to the pre-modifier vertex count, and ``_combine`` drops a flux array whose
    length does not match the node count: the object then receives NO absorbed flux and
    sits at its initial temperature, warmed only by conduction from its neighbours.

    Measured on one 289-object interior before this was fixed: 229 objects (79%)
    mismatched, and every one landed at +0.332-0.334 K regardless of the flux computed
    for it, while the 60 aligned objects rose a median 13.3 K per unit flux. Two
    instances of the same asset made it unmistakable -- 584 verts/584 nodes rose 33.8 K;
    584 verts/9305 nodes rose 0.333 K on identical material and flux.

    This lives in one function because the fix was originally applied to only one of the
    two near-identical bake bodies, which silently reintroduced the same class of bug on
    the albedo side: a mismatched albedo is discarded in favour of albedo=0, i.e. full
    absorption, overestimating absorbed flux by up to ~4x on a light surface.

    Falls back to ``obj.data`` when the evaluated mesh is unusable (no vertices, or no UV
    layers to sample through).
    """
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        candidate = obj.evaluated_get(depsgraph).data
        if candidate is not None and len(candidate.vertices) > 0 and len(candidate.uv_layers) > 0:
            return candidate
    except Exception:  # pragma: no cover - defensive, mirrors this module's style
        pass
    return obj.data


def bake_albedo_map(scene, obj, texture_size: int) -> BakedFluxMap | None:
    """
    Bake visible diffuse albedo (COLOR pass) for a single object.
    """
    if obj.type != "MESH":
        return None

    # Ensure object has source UVs for its real materials to sample from.
    if _ensure_uv_layer(obj) is None:
        return None

    # Snapshot UV state so we don't permanently change the user's UV selections.
    uv_snapshot = snapshot_uv_states([obj])

    # Ensure the bake UV exists and is unwrapped.
    prepare_object_bake_uv(obj)

    # Record which UV map this bake used (bake UV, matching shared pipeline).
    try:
        obj["heat_sim_albedo_uv"] = BAKE_UV_LAYER_NAME
    except Exception:
        pass

    image = _ensure_albedo_image(obj.name, texture_size)
    _prepare_image_nodes_for_bake(obj, image, "HeatSim_Albedo", "HeatSim Albedo")

    render = scene.render
    bake_settings = render.bake
    prev_engine = render.engine
    prev_settings = (
        getattr(bake_settings, "use_pass_direct", None),
        getattr(bake_settings, "use_pass_indirect", None),
        getattr(bake_settings, "use_pass_color", None),
        getattr(bake_settings, "target", None),
    )

    view_layer = scene.view_layers.active if hasattr(scene.view_layers, "active") else scene.view_layers[0]
    prev_active = view_layer.objects.active
    prev_selection = [o for o in scene.objects if o.select_get()]
    uv_override_state: _BakeMaterialUVOverride | None = None

    try:
        render.engine = "CYCLES"
        if hasattr(bake_settings, "use_pass_direct"):
            bake_settings.use_pass_direct = False
        if hasattr(bake_settings, "use_pass_indirect"):
            bake_settings.use_pass_indirect = False
        if hasattr(bake_settings, "use_pass_color"):
            bake_settings.use_pass_color = True
        if hasattr(bake_settings, "target"):
            bake_settings.target = "IMAGE_TEXTURES"

        # Keep textures sampling from the original UVs while baking into BAKE_UV_LAYER_NAME.
        uv_override_state = _install_bake_uv_material_overrides([obj], uv_snapshot)

        # object.select_all.poll() fails in --background when the context is in
        # EDIT mode; make sure we are in OBJECT mode (with a valid active object)
        # before selecting the bake target.
        if view_layer.objects.active is None:
            view_layer.objects.active = obj
        if getattr(bpy.context, "mode", "OBJECT") != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        view_layer.objects.active = obj

        with bpy.context.temp_override(scene=scene, view_layer=view_layer, active_object=obj, selected_objects=[obj]):
            bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, target="IMAGE_TEXTURES", margin=8, margin_type='EXTEND')

    except Exception as exc:
        warnings.warn(f"HeatSim albedo bake failed for {obj.name}: {exc}")
        return None
    finally:
        render.engine = prev_engine
        if hasattr(bake_settings, "use_pass_direct") and prev_settings[0] is not None:
            bake_settings.use_pass_direct = prev_settings[0]
        if hasattr(bake_settings, "use_pass_indirect") and prev_settings[1] is not None:
            bake_settings.use_pass_indirect = prev_settings[1]
        if hasattr(bake_settings, "use_pass_color") and prev_settings[2] is not None:
            bake_settings.use_pass_color = prev_settings[2]
        if hasattr(bake_settings, "target") and prev_settings[3] is not None:
            bake_settings.target = prev_settings[3]

        _restore_bake_uv_material_overrides(uv_override_state)

        # Guarded: this runs in a finally, so an unhandled poll() failure here
        # (context left in EDIT mode) would mask the real error and abort the
        # whole scene's bake.
        try:
            bpy.ops.object.select_all(action="DESELECT")
        except Exception:
            pass
        for item in prev_selection:
            try:
                item.select_set(True)
            except Exception:
                pass
        view_layer.objects.active = prev_active
        # Restore original UV selections
        restore_uv_states(uv_snapshot)

    rgb_pixels = _image_pixels_to_rgb(image)
    if rgb_pixels is None:
        return None

    mesh = _mesh_to_sample(obj)
    mesh.calc_loop_triangles()
    if len(mesh.loop_triangles) == 0:
        return None

    # Use the bake UV layer for mapping baked pixels to vertices.
    uv_layer = mesh.uv_layers.get(BAKE_UV_LAYER_NAME) or (mesh.uv_layers.active or mesh.uv_layers[0])

    loop_indices = np.zeros((len(mesh.loop_triangles), 3), dtype=np.int32)
    mesh.loop_triangles.foreach_get("loops", loop_indices.ravel())

    uv_data = np.zeros((len(mesh.loops), 2), dtype=np.float64)
    uv_layer.data.foreach_get("uv", uv_data.ravel())

    loop_vertex_indices = np.zeros(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vertex_indices)

    width, height = image.size
    vertex_luma = _image_to_vertex_irradiance(
        loop_vertex_indices,
        uv_data,
        rgb_pixels,
        width,
        height,
        1.0,
        len(mesh.vertices),
    )

    obj["heat_sim_albedo_image"] = image.name
    return BakedFluxMap(
        pixels=rgb_pixels,
        vertex_flux=vertex_luma,
        width=width,
        height=height,
    )


def bake_irradiance_map(scene, obj, texture_size: int, samples: int | None = None) -> BakedFluxMap | None:
    """Bake incident irradiance (DIFFUSE DIRECT+INDIRECT) for a single object.

    The mirror image of :func:`bake_albedo_map`: that bakes COLOR with the light
    passes off (a texture lookup, no light transport); this bakes the light passes
    with COLOR off, so the result is incoming light *independent of surface colour* -
    irradiance, not radiosity.

    Why this exists: the analytic Direct Kernel counts only objects of type ``LIGHT``
    plus a world sky term. A scene lit by *emissive geometry* - the standard way an
    interior is daylit, e.g. visionsim50/classroom's ``dayLight_portal`` material at
    emission strength 20 across 6.17 m2 of windows - therefore receives no thermal flux
    from its actual light source, and the kernel models no indirect bounce either.
    Cycles resolves emissive meshes, bounce, portals and HDRI transport for free.

    Cost is close to the albedo bake already run per object: both are dominated by UV
    setup, texture allocation and BVH build rather than ray tracing. Measured on
    classroom at 512px/128spp: 0.95 s/object COLOR vs 0.97 s/object DIRECT+INDIRECT.

    ``vertex_flux`` is per-vertex irradiance in W/m2; Cycles bakes outgoing radiance so
    it is scaled by ``CYCLES_LOUT_TO_IRRADIANCE`` (= pi).

    NOTE: this is *incident* flux. The solver wants *absorbed* flux, so the caller
    applies (1 - albedo) - see ``adapter._compute_irradiance_cycles``. The Direct
    Kernel returns absorbed flux directly; that is the one contract difference.
    """
    if obj.type != "MESH":
        return None

    # Ensure object has source UVs for its real materials to sample from.
    if _ensure_uv_layer(obj) is None:
        return None

    # Snapshot UV state so we don't permanently change the user's UV selections.
    uv_snapshot = snapshot_uv_states([obj])

    # Ensure the bake UV exists and is unwrapped.
    prepare_object_bake_uv(obj)

    # Record which UV map this bake used (bake UV, matching shared pipeline).
    try:
        obj["heat_sim_flux_uv"] = BAKE_UV_LAYER_NAME
    except Exception:
        pass

    image = _ensure_irradiance_image(obj.name, texture_size)
    _prepare_image_nodes_for_bake(obj, image, "HeatSim_Irradiance", "HeatSim Irradiance")

    render = scene.render
    bake_settings = render.bake
    prev_engine = render.engine
    prev_settings = (
        getattr(bake_settings, "use_pass_direct", None),
        getattr(bake_settings, "use_pass_indirect", None),
        getattr(bake_settings, "use_pass_color", None),
        getattr(bake_settings, "target", None),
    )

    view_layer = scene.view_layers.active if hasattr(scene.view_layers, "active") else scene.view_layers[0]
    prev_active = view_layer.objects.active
    prev_selection = [o for o in scene.objects if o.select_get()]
    uv_override_state: _BakeMaterialUVOverride | None = None

    # Bake sampling is deliberately overridden rather than inherited. The dataset's
    # blends ship `samples=256` with adaptive sampling at threshold 0.05 - five times
    # looser than Blender's 0.01 default - so adaptive terminates texels far below the
    # nominal cap. Measured on visionsim50/diningroom that leaves 9.6-19.2% relative
    # noise per texel; a steady-state surface sits at T ~ (E/(eps*sigma))^(1/4), so that
    # is ~2.4-4.8% in T, i.e. several-Kelvin blotches across the temperature field.
    # Adaptive is switched OFF, not merely tightened: at a matched cap it measured
    # WORSE than fixed sampling (4.17% vs 3.06%) because it still cuts texels short.
    # Denoising is deliberately NOT touched - it provably does nothing for a bake
    # (output identical to four decimals with it on and off), unlike a rendered pass.
    cycles = getattr(scene, "cycles", None)
    prev_sampling = None
    if cycles is not None and samples is not None:
        prev_sampling = (
            getattr(cycles, "samples", None),
            getattr(cycles, "use_adaptive_sampling", None),
        )

    try:
        render.engine = "CYCLES"
        if prev_sampling is not None:
            if hasattr(cycles, "use_adaptive_sampling"):
                cycles.use_adaptive_sampling = False
            if hasattr(cycles, "samples"):
                cycles.samples = int(samples)
        if hasattr(bake_settings, "use_pass_direct"):
            bake_settings.use_pass_direct = True
        if hasattr(bake_settings, "use_pass_indirect"):
            bake_settings.use_pass_indirect = True
        if hasattr(bake_settings, "use_pass_color"):
            bake_settings.use_pass_color = False
        if hasattr(bake_settings, "target"):
            bake_settings.target = "IMAGE_TEXTURES"

        # Keep textures sampling from the original UVs while baking into BAKE_UV_LAYER_NAME.
        uv_override_state = _install_bake_uv_material_overrides([obj], uv_snapshot)

        # object.select_all.poll() fails in --background when the context is in
        # EDIT mode; make sure we are in OBJECT mode (with a valid active object)
        # before selecting the bake target.
        if view_layer.objects.active is None:
            view_layer.objects.active = obj
        if getattr(bpy.context, "mode", "OBJECT") != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        view_layer.objects.active = obj

        with bpy.context.temp_override(scene=scene, view_layer=view_layer, active_object=obj, selected_objects=[obj]):
            bpy.ops.object.bake(type="DIFFUSE", pass_filter={"DIRECT", "INDIRECT"}, target="IMAGE_TEXTURES", margin=8, margin_type='EXTEND')

    except Exception as exc:
        warnings.warn(f"HeatSim irradiance bake failed for {obj.name}: {exc}")
        return None
    finally:
        render.engine = prev_engine
        if prev_sampling is not None:
            if prev_sampling[0] is not None and hasattr(cycles, "samples"):
                cycles.samples = prev_sampling[0]
            if prev_sampling[1] is not None and hasattr(cycles, "use_adaptive_sampling"):
                cycles.use_adaptive_sampling = prev_sampling[1]
        if hasattr(bake_settings, "use_pass_direct") and prev_settings[0] is not None:
            bake_settings.use_pass_direct = prev_settings[0]
        if hasattr(bake_settings, "use_pass_indirect") and prev_settings[1] is not None:
            bake_settings.use_pass_indirect = prev_settings[1]
        if hasattr(bake_settings, "use_pass_color") and prev_settings[2] is not None:
            bake_settings.use_pass_color = prev_settings[2]
        if hasattr(bake_settings, "target") and prev_settings[3] is not None:
            bake_settings.target = prev_settings[3]

        _restore_bake_uv_material_overrides(uv_override_state)

        # Guarded: this runs in a finally, so an unhandled poll() failure here
        # (context left in EDIT mode) would mask the real error and abort the
        # whole scene's bake.
        try:
            bpy.ops.object.select_all(action="DESELECT")
        except Exception:
            pass
        for item in prev_selection:
            try:
                item.select_set(True)
            except Exception:
                pass
        view_layer.objects.active = prev_active
        # Restore original UV selections
        restore_uv_states(uv_snapshot)

    rgb_pixels = _image_pixels_to_rgb(image)
    if rgb_pixels is None:
        return None

    mesh = _mesh_to_sample(obj)
    mesh.calc_loop_triangles()
    if len(mesh.loop_triangles) == 0:
        return None

    # Use the bake UV layer for mapping baked pixels to vertices.
    uv_layer = mesh.uv_layers.get(BAKE_UV_LAYER_NAME) or (mesh.uv_layers.active or mesh.uv_layers[0])

    loop_indices = np.zeros((len(mesh.loop_triangles), 3), dtype=np.int32)
    mesh.loop_triangles.foreach_get("loops", loop_indices.ravel())

    uv_data = np.zeros((len(mesh.loops), 2), dtype=np.float64)
    uv_layer.data.foreach_get("uv", uv_data.ravel())

    loop_vertex_indices = np.zeros(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vertex_indices)

    width, height = image.size
    vertex_luma = _image_to_vertex_irradiance(
        loop_vertex_indices,
        uv_data,
        rgb_pixels,
        width,
        height,
        CYCLES_LOUT_TO_IRRADIANCE,
        len(mesh.vertices),
    )

    obj["heat_sim_irradiance_image"] = image.name
    return BakedFluxMap(
        pixels=rgb_pixels,
        vertex_flux=vertex_luma,
        width=width,
        height=height,
    )
