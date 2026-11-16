"""Isolated Blender check for live transient identity and unchanged attachment.

Run with Blender --background --factory-startup --python this_file -- OUTPUT.
"""

import json
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import bpy
import torch

from visionsim.simulate.blender import BlenderService
from visionsim.simulate.config import ThermalConfig
from visionsim.simulate.heatsim import adapter, cache, thermal_shader


def run(output):
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    fixture = output / "scene.blend"
    helpers = runpy.run_path(str(Path(__file__).with_name("transient_blender_check.py")))
    helpers["generated_scene"]("vertex", fixture)
    service = BlenderService()
    service.exposed_initialize(str(fixture), str(output / "renders"))
    scene = service.scene
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 2
    scene.cycles.device = "CPU"
    config = {"render_domain": "VERTEX", "cache_root": str(output / "cache")}
    values = vars(ThermalConfig(device="cpu"))
    defaults, solver, _, assignment = service._thermal_config(
        **{
            key: values[key]
            for key in (
                "device",
                "initial_temperature_K",
                "thermal_diffusivity_mm2_s",
                "density_kg_m3",
                "specific_heat_J_kgK",
                "emissivity",
                "irradiance_scale",
                "sim_time_s",
                "timestep_s",
            )
        }
    )
    objects = adapter.gather_meshes(scene)
    plan = adapter.build_atlas_plan(scene, objects, config)
    with adapter.use_atlas_geometry(scene, plan):
        initial = adapter._combine(objects, {}, defaults, solver, assignment, plan)

    original = cache.transient_scene_digest(scene, initial)
    assert cache.transient_scene_digest(scene, initial) == original
    ball = scene.objects["HotBall"]
    ball.heat_sim_material.dirichlet_temperature_K += 1
    assert cache.transient_scene_digest(scene, initial) != original
    ball.heat_sim_material.dirichlet_temperature_K -= 1
    assert cache.transient_scene_digest(scene, initial) == original
    mat = ball.data.materials[0]
    texture = mat.node_tree.nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.new("Authored texture", width=2, height=2)
    mat.node_tree.links.new(texture.outputs["Color"], mat.node_tree.nodes.get("Principled BSDF").inputs["Base Color"])
    textured = cache.transient_scene_digest(scene, initial)
    texture.image.pixels[0] = 0.75
    assert cache.transient_scene_digest(scene, initial) != textured, "Image pixel edits were not detected"
    mat.node_tree.nodes.remove(texture)
    ball.location.x = 0.2
    ball.keyframe_insert(data_path="location", index=0, frame=9)
    scene.frame_set(1)
    assert cache.transient_scene_digest(scene, initial) != original, "Future key edit was not detected"
    # Reload the pristine scene and simulate; generated bake UVs and AOVs must
    # not change its authored-input fingerprint.
    service.exposed_initialize(str(fixture), str(output / "renders"))
    scene = service.scene
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 2
    scene.cycles.device = "CPU"
    config.update(
        mode="TRANSIENT",
        device="cpu",
        simulation_start_frame=1,
        simulation_end_frame=2,
        irradiance_bake_mode="ONCE",
        irradiance_scale=0,
        bake_samples=1,
        irradiance_texture_size=16,
        radiance=True,
    )
    trajectory = service.exposed_prepare_transient(config)
    scene.frame_set(1)
    after = cache.transient_scene_digest(scene, initial)
    assert after == original, (original, after)
    thermal_shader.setup_temperature_aov(scene, service.view_layer)
    assert cache.transient_scene_digest(scene, initial) == after
    # Actual same-process attachment also exercises manifest comparison when
    # the caller enables scene_digest in transient.prepare.
    assert service.exposed_prepare_transient(config, trajectory=trajectory, render_outputs=False) == trajectory
    scene.objects["HotBall"].heat_sim_material.dirichlet_temperature_K += 1
    try:
        service.exposed_prepare_transient(config, trajectory=trajectory, render_outputs=False)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("Stale source temperatures were accepted during attachment")
    print("TRANSIENT_CACHE_VALIDATION", json.dumps({"digest": after, "trajectory": trajectory}))


if __name__ == "__main__":
    run(Path(sys.argv[sys.argv.index("--") + 1]))
