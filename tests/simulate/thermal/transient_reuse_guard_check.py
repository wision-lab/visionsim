"""Blender check loaded by ``TransientReuseGuardCheckService``."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import bpy
import torch

from tests.simulate.thermal import transient_blender_check
from visionsim.simulate.blender import BlenderService
from visionsim.simulate.heatsim.solver import HeatSimFEM
from visionsim.simulate.heatsim.transient import _reuse_blocker


class TransientReuseGuardCheckService(BlenderService):
    """Blender check service."""

    def exposed_run(self, root_path: str) -> dict:
        output = Path(root_path)
        torch.set_num_threads(2)
        output.mkdir(parents=True, exist_ok=True)
        fixture = output / "scene.blend"
        transient_blender_check.generated_scene("vertex", fixture)
        self.exposed_initialize(str(fixture), str(output / "renders"))
        scene = self.scene
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 2
        scene.cycles.device = "CPU"
        assert _reuse_blocker(scene) is None
        config = {
            "mode": "TRANSIENT",
            "device": "cpu",
            "render_domain": "VERTEX",
            "cache_root": str(output / "cache"),
            "simulation_start_frame": 1,
            "simulation_end_frame": 2,
            "irradiance_bake_mode": "ONCE",
            "irradiance_scale": 0,
            "bake_samples": 1,
            "irradiance_texture_size": 16,
        }

        def callback(scene):
            pass

        bpy.app.handlers.frame_change_pre.append(callback)
        try:
            assert "callbacks" in _reuse_blocker(scene)
            first = self.exposed_prepare_transient(config, render_outputs=False)
            second = self.exposed_prepare_transient(config, render_outputs=False)
            assert first != second, "Untracked callback state must bypass automatic reuse"
            with patch.object(HeatSimFEM, "simulate_for_pose", side_effect=AssertionError("Worker recomputed")):
                assert self.exposed_prepare_transient(config, trajectory=second, render_outputs=False) == second
        finally:
            bpy.app.handlers.frame_change_pre.remove(callback)
        assert _reuse_blocker(scene) is None
        obj = scene.objects["HotBall"]
        other_scene = bpy.data.scenes.new("Other scene")
        external = bpy.data.objects.new("External dependency", None)
        other_scene.collection.objects.link(external)
        assert "outside" in _reuse_blocker(scene)
        bpy.data.objects.remove(external, do_unlink=True)
        bpy.data.scenes.remove(other_scene)
        curve = obj.driver_add("location", 0)
        curve.driver.expression = "frame / 10"
        assert "drivers" in _reuse_blocker(scene)
        obj.driver_remove("location", 0)
        armature = bpy.data.objects.new("Rig", bpy.data.armatures.new("Rig"))
        scene.collection.objects.link(armature)
        assert "ARMATURE" in _reuse_blocker(scene)
        bpy.data.objects.remove(armature, do_unlink=True)
        modifier = obj.modifiers.new("External motion", "MESH_CACHE")
        assert "MESH_CACHE" in _reuse_blocker(scene)
        obj.modifiers.remove(modifier)
        group = bpy.data.node_groups.new("Motion simulation", "GeometryNodeTree")
        modifier = obj.modifiers.new("Procedural motion", "NODES")
        modifier.node_group = group
        assert _reuse_blocker(scene) is None, "Ordinary geometry nodes should remain reusable"
        simulation = group.nodes.new("GeometryNodeSimulationOutput")
        assert "geometry-node" in _reuse_blocker(scene)
        group.nodes.remove(simulation)
        group.nodes.new("GeometryNodeImportOBJ")
        assert "geometry-node" in _reuse_blocker(scene)
        obj.modifiers.remove(modifier)
        for kind in ("DYNAMIC_PAINT", "OCEAN"):
            modifier = obj.modifiers.new(kind, kind)
            assert kind in _reuse_blocker(scene)
            obj.modifiers.remove(modifier)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.rigidbody.object_add()
        assert "rigid-body" in _reuse_blocker(scene)
        scene.rigidbody_world.enabled = False
        assert _reuse_blocker(scene) is None

        return {"ok": True}
