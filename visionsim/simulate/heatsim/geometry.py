"""Private evaluated meshes shared by thermal baking, sampling and rendering."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Any

import bpy  # type: ignore
import numpy as np
import numpy.typing as npt

from visionsim.simulate.heatsim.material_slots import restore_material_slots, snapshot_material_slots


def evaluated_meshes(objects: list, *, force: bool = False) -> dict[str, Any]:
    """Copy render-visible geometry and triangulate n-gons for thermal UV charts."""
    meshes = {}
    try:
        for obj in objects:
            if (
                not force
                and obj.data.shape_keys is None
                and not any(m.show_render or m.show_viewport for m in obj.modifiers)
                and not any(p.loop_total > 4 for p in obj.data.polygons)
            ):
                continue
            settings = [
                (m, "show_viewport", m.show_viewport, m.show_render)
                for m in obj.modifiers
                if m.show_viewport != m.show_render
            ]
            settings.extend(
                (m, "levels", m.levels, m.render_levels)
                for m in obj.modifiers
                if m.type in {"SUBSURF", "MULTIRES"} and m.levels != m.render_levels
            )
            try:
                for modifier, attr, _, render_value in settings:
                    setattr(modifier, attr, render_value)
                if settings:
                    bpy.context.view_layer.update()
                depsgraph = bpy.context.evaluated_depsgraph_get()
                mesh = bpy.data.meshes.new_from_object(
                    obj.evaluated_get(depsgraph), preserve_all_data_layers=True, depsgraph=depsgraph
                )
                meshes[obj.name] = mesh
                mesh["heatsim_authored_render_uv"] = next((uv.name for uv in mesh.uv_layers if uv.active_render), "")
                # Nonplanar n-gons can fold within one projected UV chart. Give
                # their triangles independent corners on the private thermal mesh.
                if any(p.loop_total > 4 for p in mesh.polygons):
                    import bmesh  # type: ignore[import-not-found]

                    bm = bmesh.new()
                    try:
                        bm.from_mesh(mesh)
                        bmesh.ops.triangulate(
                            bm, faces=[f for f in bm.faces if len(f.verts) > 4], ngon_method="EAR_CLIP"
                        )
                        bm.to_mesh(mesh)
                    finally:
                        bm.free()
                # Preserve Generated-coordinate bounds for authored materials.
                mesh.use_auto_texspace = False
                mesh.texspace_location = obj.data.texspace_location
                mesh.texspace_size = obj.data.texspace_size
                mesh.pop("heatsim_bake_uv_signature", None)
            finally:
                for modifier, attr, original, _ in reversed(settings):
                    setattr(modifier, attr, original)
                if settings:
                    bpy.context.view_layer.update()
    except Exception:
        release_meshes(meshes)
        raise
    return meshes


def refresh_evaluated_meshes(scene: Any, meshes: dict[str, Any]) -> None:
    """Refresh positions while preserving the thermal snapshots' UVs and attributes.

    Call outside ``use_meshes`` so evaluation sees the authored modifiers and
    shape keys. Topology and authored shading attributes must remain stable.
    """
    from visionsim.simulate.heatsim.cache import mesh_attribute_digest
    from visionsim.simulate.heatsim.names import ATLAS_UV_LAYER_NAME, BAKE_UV_LAYER_NAME

    def authored_uvs(mesh):
        layers = []
        for layer in mesh.uv_layers:
            if layer.name in {ATLAS_UV_LAYER_NAME, BAKE_UV_LAYER_NAME}:
                continue
            coordinates = np.empty(len(layer.data) * 2, dtype=np.float32)
            layer.data.foreach_get("uv", coordinates)
            layers.append((layer.name, coordinates.tobytes()))
        return layers

    current = evaluated_meshes([scene.objects[name] for name in meshes], force=True)
    try:
        for name, previous in meshes.items():
            fresh = current[name]
            if (len(previous.vertices), len(previous.loops), len(previous.polygons)) != (
                len(fresh.vertices),
                len(fresh.loops),
                len(fresh.polygons),
            ):
                raise RuntimeError(f"{name!r}: transient thermal mesh topology changed")
            for collection, attr in (
                ("loops", "vertex_index"),
                ("polygons", "loop_total"),
                ("polygons", "material_index"),
            ):
                old_items, new_items = getattr(previous, collection), getattr(fresh, collection)
                old: npt.NDArray[np.int32] = np.empty(len(old_items), dtype=np.int32)
                new: npt.NDArray[np.int32] = np.empty(len(new_items), dtype=np.int32)
                old_items.foreach_get(attr, old)
                new_items.foreach_get(attr, new)
                if not np.array_equal(old, new):
                    raise RuntimeError(f"{name!r}: transient thermal topology or material assignment changed")
            if (
                list(previous.materials) != list(fresh.materials)
                or tuple(previous.texspace_location) != tuple(fresh.texspace_location)
                or tuple(previous.texspace_size) != tuple(fresh.texspace_size)
                or previous.color_attributes.render_color_index != fresh.color_attributes.render_color_index
                or previous.get("heatsim_authored_render_uv") != fresh.get("heatsim_authored_render_uv")
                or authored_uvs(previous) != authored_uvs(fresh)
                or mesh_attribute_digest(previous, include_position=False)
                != mesh_attribute_digest(fresh, include_position=False)
            ):
                raise RuntimeError(
                    f"{name!r}: transient changes to evaluated material slots, UVs or mesh attributes are unsupported"
                )
        # Validate all objects before changing any retained snapshot.
        for name, previous in meshes.items():
            coordinates: npt.NDArray[np.float64] = np.empty(len(previous.vertices) * 3, dtype=np.float64)
            current[name].vertices.foreach_get("co", coordinates)
            previous.vertices.foreach_set("co", coordinates)
            previous.update()
    finally:
        release_meshes(current)


def release_meshes(meshes: dict[str, Any]) -> None:
    """Release snapshots once their render job is finished or preparation failed."""
    for mesh in meshes.values():
        bpy.data.meshes.remove(mesh)
    meshes.clear()


def _restore_object(obj, mesh, slots, modifiers) -> None:
    obj.data = mesh
    restore_material_slots(obj, slots)
    for modifier, render, viewport in modifiers:
        modifier.show_render, modifier.show_viewport = render, viewport


@contextmanager
def use_meshes(scene: Any, meshes: dict[str, Any]):
    """Temporarily install snapshots; never evaluate their modifiers a second time."""
    with ExitStack() as stack:
        stack.callback(bpy.context.view_layer.update)
        for name, mesh in meshes.items():
            obj = scene.objects[name]
            modifiers = [(m, m.show_render, m.show_viewport) for m in obj.modifiers]
            stack.callback(_restore_object, obj, obj.data, snapshot_material_slots(obj), modifiers)
            obj.data = mesh
            for modifier, _, _ in modifiers:
                modifier.show_render = modifier.show_viewport = False
        bpy.context.view_layer.update()
        yield
