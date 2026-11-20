"""Blender check loaded by ``TransientGeometryCheckService``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import bpy
import numpy as np

from visionsim.simulate.blender import BlenderService
from visionsim.simulate.heatsim.geometry import evaluated_meshes, refresh_evaluated_meshes, release_meshes
from visionsim.simulate.heatsim.names import ATLAS_UV_LAYER_NAME, BAKE_UV_LAYER_NAME


class TransientGeometryCheckService(BlenderService):
    """Blender check service."""

    def exposed_run(self) -> dict:
        bpy.ops.mesh.primitive_plane_add()
        obj = bpy.context.object
        obj.shape_key_add(name="Basis")
        key = obj.shape_key_add(name="Raised")
        key.data[0].co.z = 0.2
        attr = obj.data.attributes.new("authored_albedo", "FLOAT", "POINT")
        for item in attr.data:
            item.value = 0.1
        obj.data.uv_layers.new(name="Second UV")
        obj.data.color_attributes.new(name="Color One", type="FLOAT_COLOR", domain="POINT")
        obj.data.color_attributes.new(name="Color Two", type="FLOAT_COLOR", domain="POINT")
        obj.data.color_attributes.render_color_index = 0
        original_material = bpy.data.materials.new("Original")
        replacement_material = bpy.data.materials.new("Replacement")
        obj.data.materials.append(original_material)
        meshes = evaluated_meshes([obj])
        try:
            retained = meshes[obj.name]
            for name in ("sim_temperature", "emissivity"):
                retained.attributes.new(name, "FLOAT", "POINT")
            for name in (ATLAS_UV_LAYER_NAME, BAKE_UV_LAYER_NAME):
                retained.uv_layers.new(name=name)
            key.value = 1
            bpy.context.view_layer.update()
            refresh_evaluated_meshes(bpy.context.scene, meshes)
            assert np.isclose(retained.vertices[0].co.z, 0.2)
            # Changing shader values on the same material remains supported.
            original_material.diffuse_color = (0.2, 0.3, 0.4, 1)
            refresh_evaluated_meshes(bpy.context.scene, meshes)

            def rejected():
                try:
                    refresh_evaluated_meshes(bpy.context.scene, meshes)
                except RuntimeError as exc:
                    assert "material slots, UVs or mesh attributes" in str(exc)
                else:
                    raise AssertionError("Animated shading data silently froze at the initial frame")

            attr.data[0].value = 0.9
            rejected()
            attr.data[0].value = 0.1
            uv = obj.data.uv_layers[0].data[0]
            original_uv = tuple(uv.uv)
            uv.uv.x += 0.1
            rejected()
            uv.uv = original_uv
            obj.data.uv_layers[1].active_render = True
            rejected()
            obj.data.uv_layers[0].active_render = True
            obj.data.color_attributes.render_color_index = 1
            rejected()
            obj.data.color_attributes.render_color_index = 0
            obj.data.use_auto_texspace = False
            original_size = tuple(obj.data.texspace_size)
            obj.data.texspace_size.x += 1
            rejected()
            obj.data.texspace_size = original_size
            obj.data.materials[0] = replacement_material
            rejected()
            obj.data.materials[0] = original_material
            refresh_evaluated_meshes(bpy.context.scene, meshes)
        finally:
            release_meshes(meshes)

        return {"ok": True}
