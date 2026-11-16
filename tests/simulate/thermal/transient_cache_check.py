"""Isolated Blender check for live transient identity and unchanged attachment.

Run with Blender --background --factory-startup --python this_file -- OUTPUT.
"""

import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

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
    collection = bpy.data.collections.new("Visibility check")
    scene.collection.children.link(collection)
    collection.objects.link(ball)
    before = cache.transient_scene_digest(scene, initial)
    collection.hide_render = not collection.hide_render
    assert cache.transient_scene_digest(scene, initial) != before, "Collection visibility edits were not detected"
    bpy.data.collections.remove(collection)
    before = cache.transient_scene_digest(scene, initial)
    scene.cycles.max_bounces += 1
    assert cache.transient_scene_digest(scene, initial) != before, "Cycles bounce edits were not detected"
    scene.cycles.max_bounces -= 1
    mat = ball.data.materials[0]
    texture = mat.node_tree.nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.new("Authored texture", width=2, height=2)
    mat.node_tree.links.new(texture.outputs["Color"], mat.node_tree.nodes.get("Principled BSDF").inputs["Base Color"])
    textured = cache.transient_scene_digest(scene, initial)
    texture.image.pixels[0] = 0.75
    assert cache.transient_scene_digest(scene, initial) != textured, "Image pixel edits were not detected"
    textured = cache.transient_scene_digest(scene, initial)
    texture.image.colorspace_settings.name = "Non-Color"
    assert cache.transient_scene_digest(scene, initial) != textured, "Image color space edits were not detected"
    ball.data.uv_layers.new(name="Authored UV 2")
    ball.data.uv_layers[0].active_render = True
    textured = cache.transient_scene_digest(scene, initial)
    ball.data.uv_layers["Authored UV 2"].active_render = True
    assert cache.transient_scene_digest(scene, initial) != textured, "Active render UV edits were not detected"
    mat.node_tree.nodes.remove(texture)
    base_color = mat.node_tree.nodes.get("Principled BSDF").inputs["Base Color"]
    for node_type, socket_name, changed in (
        ("ShaderNodeRGB", "Color", (0.8, 0.2, 0.1, 1)),
        ("ShaderNodeValue", "Value", 0.8),
    ):
        constant = mat.node_tree.nodes.new(node_type)
        mat.node_tree.links.new(constant.outputs[socket_name], base_color)
        before = cache.transient_scene_digest(scene, initial)
        constant.outputs[socket_name].default_value = changed
        assert cache.transient_scene_digest(scene, initial) != before, f"{node_type} output edit was not detected"
        mat.node_tree.nodes.remove(constant)
    ramp = mat.node_tree.nodes.new("ShaderNodeValToRGB")
    mat.node_tree.links.new(ramp.outputs["Color"], base_color)
    before = cache.transient_scene_digest(scene, initial)
    ramp.color_ramp.elements[0].color = (1, 0, 0, 1)
    assert cache.transient_scene_digest(scene, initial) != before, "Color ramp edits were not detected"
    mat.node_tree.nodes.remove(ramp)
    curves = mat.node_tree.nodes.new("ShaderNodeRGBCurve")
    mat.node_tree.links.new(curves.outputs["Color"], base_color)
    before = cache.transient_scene_digest(scene, initial)
    curves.mapping.curves[0].points.new(0.3, 0.7)
    curves.mapping.update()
    assert cache.transient_scene_digest(scene, initial) != before, "RGB curve edits were not detected"
    mat.node_tree.nodes.remove(curves)
    colors = ball.data.color_attributes.new(name="Authored colors", type="FLOAT_COLOR", domain="POINT")
    before = cache.transient_scene_digest(scene, initial)
    colors.data[0].color = (1, 0, 0, 1)
    assert cache.transient_scene_digest(scene, initial) != before, "Mesh color edits were not detected"
    ball.data.color_attributes.new(name="Second colors", type="FLOAT_COLOR", domain="POINT")
    before = cache.transient_scene_digest(scene, initial)
    ball.data.color_attributes.render_color_index = 1
    assert cache.transient_scene_digest(scene, initial) != before, "Render color selection was not detected"
    ball.data.use_auto_texspace = False
    before = cache.transient_scene_digest(scene, initial)
    ball.data.texspace_location.x += 0.1
    assert cache.transient_scene_digest(scene, initial) != before, "Generated-coordinate bounds were not detected"
    noise = mat.node_tree.nodes.new("ShaderNodeTexNoise")
    mat.node_tree.links.new(noise.outputs["Color"], base_color)
    before = cache.transient_scene_digest(scene, initial)
    noise.texture_mapping.translation.x += 0.2
    assert cache.transient_scene_digest(scene, initial) != before, "Texture mapping edit was not detected"
    before = cache.transient_scene_digest(scene, initial)
    noise.color_mapping.brightness += 0.2
    assert cache.transient_scene_digest(scene, initial) != before, "Texture color mapping edit was not detected"
    mat.node_tree.nodes.remove(noise)
    ball.location.x = 0.2
    ball.keyframe_insert(data_path="location", index=0, frame=9)
    scene.frame_set(1)
    assert cache.transient_scene_digest(scene, initial) != original, "Future key edit was not detected"
    track = ball.animation_data.nla_tracks.new()
    strip = track.strips.new("Animated influence", 1, ball.animation_data.action)
    strip.use_animated_influence = True
    strip.influence = 0.5
    strip.keyframe_insert(data_path="influence", frame=1)
    before = cache.transient_scene_digest(scene, initial)
    strip.fcurves[0].keyframe_points[0].co.y = 0.8
    assert cache.transient_scene_digest(scene, initial) != before, "NLA influence edits were not detected"
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
        simulation_end_frame=3,
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
    # Reuse must avoid both lighting bakes and solver calls, including when
    # attaching a forced run on another rendering worker.
    from visionsim.simulate.heatsim.solver import HeatSimFEM

    def reuse(expected, **options):
        with (
            patch.object(adapter, "_compute_irradiance", side_effect=AssertionError("Unexpected bake")),
            patch.object(HeatSimFEM, "simulate_for_pose", side_effect=AssertionError("Unexpected solve")),
        ):
            assert service.exposed_prepare_transient({**config, **options}, render_outputs=False) == expected

    reuse(trajectory)
    forced = service.exposed_prepare_transient({**config, "recompute": True}, render_outputs=False)
    assert forced != trajectory
    with patch.object(HeatSimFEM, "simulate_for_pose", side_effect=AssertionError("Worker recomputed")):
        assert (
            service.exposed_prepare_transient({**config, "recompute": True}, trajectory=forced, render_outputs=False)
            == forced
        )
    # Prefer the newest completed run, including one produced with True.
    reuse(forced)
    for key, value in (
        ("substeps_per_frame", 2),
        ("laplacian_update_every_steps", 2),
        ("irradiance_bake_mode", "EVERY_STEP"),
        ("irradiance_bake_every_n_frames", 2),
        ("simulation_end_frame", 2),
    ):
        changed = service.exposed_prepare_transient({**config, key: value}, render_outputs=False)
        assert changed != forced, key
        reuse(changed, **{key: value})
    # A complete manifest with a damaged middle frame must not be reused.
    (Path(trajectory) / "manifest.json").unlink()
    (Path(forced) / "frame-2" / "temperatures.npz").write_bytes(b"broken archive")
    trajectory = service.exposed_prepare_transient(config, render_outputs=False)
    assert trajectory != forced
    reuse(trajectory)
    scene.objects["HotBall"].heat_sim_material.dirichlet_temperature_K += 1
    try:
        service.exposed_prepare_transient(config, trajectory=trajectory, render_outputs=False)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("Stale source temperatures were accepted during attachment")
    material_changed = service.exposed_prepare_transient(config, render_outputs=False)
    assert material_changed != trajectory
    reuse(material_changed)
    scene.render.fps += 1
    timing_changed = service.exposed_prepare_transient(config, render_outputs=False)
    assert timing_changed != material_changed
    reuse(timing_changed)
    manifest_path = Path(timing_changed) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["preview_range"]
    manifest_path.write_text(json.dumps(manifest))
    repaired = service.exposed_prepare_transient(config, render_outputs=False)
    assert repaired != timing_changed
    reuse(repaired)
    print("TRANSIENT_CACHE_VALIDATION", json.dumps({"digest": after, "trajectory": trajectory}))


if __name__ == "__main__":
    run(Path(sys.argv[sys.argv.index("--") + 1]))
