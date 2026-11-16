"""Actual Blender checks, invoked in a private process without package installation.

Run with Blender --background --factory-startup --python this_file -- CASE OUTPUT.
All scene copies, caches and renders go to OUTPUT; source assets are never saved.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import bpy
import Imath
import numpy as np
import OpenEXR
import torch
from mathutils import Vector

from visionsim.simulate.blender import BlenderService
from visionsim.simulate.heatsim import adapter, cache, irradiance, register


def material(obj, temperature=295.0, pinned=False):
    props = obj.heat_sim_material
    props.initial_temperature_K = temperature
    props.thermal_diffusivity_mm2_s = 200.0
    props.emissivity = 0.9
    props.thermal_role = "DIRICHLET_SOURCE" if pinned else "FEM_PARTICIPANT"
    props.dirichlet_temperature_K = temperature
    obj.heat_simulation_enabled = True
    mat = bpy.data.materials.new(obj.name + " RGB")
    mat.use_nodes = True
    mat.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = (0.3, 0.3, 0.3, 1.0)
    obj.data.materials.append(mat)


def keyframe_location(obj, start, end, last):
    obj.location = start
    obj.keyframe_insert(data_path="location", frame=1)
    obj.location = end
    obj.keyframe_insert(data_path="location", frame=last)


def generated_scene(case, path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if not hasattr(bpy.types.Object, "heat_sim_material"):
        register()
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 9
    scene.render.fps = 8
    scene.world = bpy.data.worlds.new("Dark environment")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes.get("Background").inputs["Strength"].default_value = 0
    nx, ny = 41, 17
    verts = [(x, y, 0.0) for y in np.linspace(-0.04, 0.04, ny) for x in np.linspace(-0.1, 0.1, nx)]
    faces = [
        (j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i)
        for j in range(ny - 1)
        for i in range(nx - 1)
    ]
    mesh = bpy.data.meshes.new("Floor mesh")
    mesh.from_pydata(verts, [], faces)
    floor = bpy.data.objects.new("Floor", mesh)
    scene.collection.objects.link(floor)
    material(floor)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.012)
    ball = bpy.context.object
    ball.name = "HotBall"
    material(ball, 360.0, pinned=case != "finite")
    keyframe_location(ball, (-0.065, 0, 0.012), (0.065, 0, 0.012), 9)
    ball.rotation_euler = (0, 0, 0)
    ball.keyframe_insert(data_path="rotation_euler", frame=1)
    ball.rotation_euler.y = 0.13 / 0.012
    ball.keyframe_insert(data_path="rotation_euler", frame=9)
    if case == "texel":
        floor.rotation_euler.z = 0
        floor.keyframe_insert(data_path="rotation_euler", frame=1)
        floor.rotation_euler.z = 0.1
        floor.keyframe_insert(data_path="rotation_euler", frame=9)
    if case == "deform":
        floor.shape_key_add(name="Basis")
        key = floor.shape_key_add(name="Bend")
        for vertex in key.data:
            vertex.co.z += 0.003 * (vertex.co.y / 0.04) ** 2
        key.value = 0
        key.keyframe_insert(data_path="value", frame=1)
        key.value = 1
        key.keyframe_insert(data_path="value", frame=9)
    if case == "lighting":
        ball.hide_render = True
        ball.hide_viewport = True
        bpy.ops.mesh.primitive_cube_add(size=0.035)
        occluder = bpy.context.object
        occluder.name = "MovingOccluder"
        material(occluder)
        occluder.heat_simulation_enabled = False
        keyframe_location(occluder, (-0.055, 0, 0.04), (0.055, 0, 0.04), 9)
        bpy.ops.object.light_add(type="SUN", location=(0, 0, 1))
        bpy.context.object.data.energy = 500
        bpy.context.object.data.angle = 0.01
    bpy.ops.object.camera_add(location=(0, -0.23, 0.2))
    camera = bpy.context.object
    camera.rotation_euler = (Vector((0, 0, 0)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "PERSP"
    camera.data.lens = 40
    camera.data.clip_start = 0.001
    scene.camera = camera
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


class ValidationService(BlenderService):
    """RPC-only observation helpers; production setup and rendering stay unchanged."""

    def exposed_make_fixture(self, path):
        torch.set_num_threads(2)
        generated_scene("vertex", Path(str(path)))

    def exposed_initialize(self, *args, **kwargs):
        result = super().exposed_initialize(*args, **kwargs)
        torch.set_num_threads(2)
        self.scene.render.threads_mode = "FIXED"
        self.scene.render.threads = 2
        return result

    def exposed_transient_state(self, frame):
        self.scene.frame_set(frame)
        self._apply_transient_frame()
        values = cache.read_temperatures(self._thermal_transient["root"], f"frame-{frame}")
        field = np.asarray(values["Floor"])
        if not self._thermal_atlas_plan.texels:
            attr = self.scene.objects["Floor"].data.attributes["sim_temperature"]
            actual = np.empty(len(attr.data), np.float32)
            attr.data.foreach_get("value", actual)
            assert np.allclose(actual, field[0], atol=5e-5)
        return json.dumps(
            {
                "root": str(self._thermal_transient["root"]),
                "atlas_root": str(self._thermal_transient.get("atlas_root", "")),
                "stats": self._thermal_transient["manifest"]["stats"],
                "floor_digest": hashlib.sha256(field.tobytes()).hexdigest(),
                "module": str(Path(sys.modules[BlenderService.__module__].__file__).resolve()),
            }
        )


def run(case, output):
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    asset = Path("/data/sriram/blender_files/test_ball_rolling.blend")
    source = asset if case == "asset" else output / "fixture.blend"
    if case != "asset":
        generated_scene(case, source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    service = BlenderService()
    service.exposed_initialize(str(source), str(output / "renders"))
    scene = service.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 4
    scene.cycles.use_denoising = False
    scene.render.resolution_x = 96
    scene.render.resolution_y = 64
    scene.render.resolution_percentage = 100
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 2
    end = 60 if case == "asset" else 9
    floor_name = "Plane" if case == "asset" else "Floor"
    if case == "asset":
        scene.frame_set(1)
        for obj in scene.objects:
            if obj.type == "MESH":
                obj.heat_simulation_enabled = True
                props = obj.heat_sim_material
                props.initial_temperature_K = 295.0
                props.thermal_diffusivity_mm2_s = 200.0
                props.emissivity = 0.9
                props.thermal_role = "FEM_PARTICIPANT"
        for name in ("Cube", "Sphere"):
            props = scene.objects[name].heat_sim_material
            props.initial_temperature_K = 360.0
            props.dirichlet_temperature_K = 360.0
            props.thermal_role = "DIRICHLET_SOURCE"
    flux_snapshots = []
    bake_uv_snapshots = {}
    original_uv = irradiance.prepare_object_bake_uv

    def prepare_uv(obj):
        original_uv(obj)
        if obj.data.get("heatsim_bake_uv_locked", False):
            layer = obj.data.uv_layers[irradiance.BAKE_UV_LAYER_NAME]
            uv = np.empty(len(layer.data) * 2, np.float32)
            layer.data.foreach_get("uv", uv)
            previous = bake_uv_snapshots.setdefault(obj.name, uv.copy())
            assert np.array_equal(previous, uv), f"Atlas UV moved during rebake: {obj.name}"

    irradiance.prepare_object_bake_uv = prepare_uv
    original_flux = adapter._compute_irradiance

    def flux(*args, **kwargs):
        result = original_flux(*args, **kwargs)
        if floor_name in {obj.name for obj in result}:
            flux_snapshots.append(np.asarray(next(v for obj, v in result.items() if obj.name == floor_name)).copy())
        return result

    adapter._compute_irradiance = flux
    previous_cache_override = scene.get("heatsim_bake_cache_root")
    original_publish_uv = cache.publish_bake_uv

    def publish_uv(path, *args, **kwargs):
        assert Path(path).is_relative_to(output / "cache"), "Bake UV cache escaped the isolated cache root"
        return original_publish_uv(path, *args, **kwargs)

    cache.publish_bake_uv = publish_uv
    config = {
        "mode": "TRANSIENT",
        "device": "cpu",
        "simulation_start_frame": 1,
        "simulation_end_frame": end,
        "substeps_per_frame": 2,
        "laplacian_update_every_steps": 2,
        "irradiance_bake_mode": "EVERY_N_FRAMES" if case in {"lighting", "texel", "deform"} else "ONCE",
        "irradiance_bake_every_n_frames": 4,
        "irradiance_scale": 1.0 if case == "lighting" else 0.0,
        "render_domain": "TEXEL" if case in {"texel", "deform"} else "VERTEX",
        "atlas_texel_density": 100000.0,
        "atlas_tile_min": 16,
        "atlas_tile_max": 64,
        "bake_samples": 8 if case == "lighting" else 1,
        "irradiance_texture_size": 64,
        "cache_root": str(output / "cache"),
        "radiance": True,
        "preview": True,
    }
    service.exposed_configure_thermal(config)
    preview_settings = service._outputs["previews/temperature"][0].format.view_settings
    assert preview_settings.view_transform == "Standard"
    assert preview_settings.gamma == 1.0 and preview_settings.exposure == 0.0
    assert scene.get("heatsim_bake_cache_root") == previous_cache_override
    assert list((output / "cache" / "bake_uv").glob("*.npz")), "UV bake cache ignored cache_root"
    state = service._thermal_transient
    root = Path(state["root"])
    stats = state["manifest"]["stats"]
    assert stats == {
        "steps": (end - 1) * 2,
        "bakes": 3 if case in {"lighting", "texel", "deform"} else 1,
        "laplacian_rebuilds": end,
    }, stats
    for obj in scene.objects:
        if obj.type == "MESH":
            assert "heatsim_bake_uv_locked" not in obj.data
    for snapshot in service._thermal_atlas_plan.meshes.values():
        assert "heatsim_bake_uv_locked" not in snapshot
    if case in {"texel", "deform"}:
        assert floor_name in bake_uv_snapshots, "No locked atlas rebake was checked"
    history = []
    hot_history = []
    for frame in range(1, end + 1):
        arrays = cache.read_temperatures(root, f"frame-{frame}")
        assert arrays is not None, frame
        for values in arrays.values():
            assert np.isfinite(values).all()
        history.append(arrays[floor_name][0])
        if "HotBall" in arrays:
            hot_history.append(arrays["HotBall"][0])
    floor_history = np.stack(history)
    assert np.allclose(floor_history[0], 295), floor_history[0].min()
    rise = floor_history[-1] - floor_history[0]
    assert rise.max() > 1e-5, float(rise.max())
    if case != "lighting":
        hot_names = ("Sphere",) if case == "asset" else ("HotBall",)
        for name in hot_names:
            if case == "finite":
                assert np.allclose(hot_history[0], 360)
                assert np.mean(hot_history[-1]) < 359.99, "Finite moving hot body did not cool"
                assert np.min(hot_history[-1]) >= 294.99
            else:
                assert np.allclose(arrays[name], 360), (name, arrays[name].min())
        early = floor_history[min(3, end - 1)] - floor_history[0]
        peak = int(np.argmax(early))
        assert early[peak] > 1e-6
        assert floor_history[-1, peak] > 295.000001, "Earlier floor patch lost its temperature history"
    else:
        assert len(flux_snapshots) == 3, len(flux_snapshots)
        assert np.max(np.abs(flux_snapshots[-1] - flux_snapshots[0])) > 1, "Moving shadow did not update floor heating"
    initial_samples = (
        service._thermal_atlas_plan.texels[floor_name]["position_mm"].copy()
        if config["render_domain"] == "TEXEL"
        else None
    )
    # Render out of order; repeat playback without reinserting frame metadata.
    rendered_fields = {}
    for frame in [end, 1, end]:
        if frame in rendered_fields:
            scene.frame_set(frame)
            service._apply_transient_frame()
        else:
            service.exposed_render_frame(frame, allow_skips=False)
        if config["render_domain"] == "VERTEX":
            attr = scene.objects[floor_name].data.attributes["sim_temperature"]
            values = np.empty(len(attr.data), np.float32)
            attr.data.foreach_get("value", values)
            assert np.allclose(values, floor_history[frame - 1], atol=5e-5)
            if frame in rendered_fields:
                assert np.array_equal(rendered_fields[frame], values)
            rendered_fields[frame] = values
        else:
            rendered_fields[frame] = True
            if frame == end and case == "deform":
                moved = service._thermal_atlas_plan.texels[floor_name]["position_mm"]
                assert np.max(moved[:, 2] - initial_samples[:, 2]) > 1.0, "Evaluated shape-key samples did not move"
    if case == "deform":
        floor = scene.objects[floor_name]
        modifier = floor.modifiers.new("Unsupported topology change", "SUBSURF")
        modifier.levels = modifier.render_levels = 1
        try:
            service._apply_transient_frame()
        except RuntimeError as error:
            assert "topology" in str(error).lower(), error
        else:
            raise AssertionError("Changed evaluated topology silently reused old temperatures")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before, "Source blend was modified"
    for modality in ("temperature", "thermal_radiance"):
        files = sorted((output / "renders" / modality).rglob("*.exr"))
        rendered = []
        for filename in files:
            image = OpenEXR.InputFile(str(filename))
            try:
                channel = next(iter(image.header()["channels"]))
                pixels = np.frombuffer(image.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)
            finally:
                image.close()
            assert np.isfinite(pixels).all(), filename
            assert np.any(pixels > (294 if modality == "temperature" else 0)), filename
            rendered.append(pixels)
        assert len(rendered) == 2
        assert not np.array_equal(*rendered), f"Rendered {modality} image did not change over time"
    report = {
        "case": case,
        "source": str(source),
        "source_sha256": before,
        "frames": end,
        "floor_max_rise_K": float(rise.max()),
        "floor_mean_rise_K": float(rise.mean()),
        "hot_body_mean_final_K": float(np.mean(hot_history[-1])) if hot_history else None,
        "stats": state["manifest"].get("stats"),
        "renders": [str(p) for p in (output / "renders").rglob("*.exr")],
    }
    assert len(report["renders"]) >= 4, report
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print("TRANSIENT_VALIDATION", json.dumps(report), flush=True)
    return service


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1 :]
    run(args[0], Path(args[1]))
