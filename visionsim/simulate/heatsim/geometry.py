"""Private evaluated meshes shared by thermal baking, sampling and rendering."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Any

import bpy  # type: ignore

from visionsim.simulate.heatsim.material_slots import restore_material_slots, snapshot_material_slots


def evaluated_meshes(objects: list) -> dict[str, Any]:
    """Copy render-visible geometry and triangulate n-gons for thermal UV charts."""
    meshes = {}
    try:
        for obj in objects:
            if not any(m.show_render or m.show_viewport for m in obj.modifiers) and not any(
                p.loop_total > 4 for p in obj.data.polygons
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
