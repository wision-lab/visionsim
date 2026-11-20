"""Dynamic Blender regression checks using isolated imports and output paths."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.parametrize("case", ["vertex", "finite", "texel", "deform", "lighting"])
def test_transient_blender(case, executable, tmp_path):
    from tests.simulate.thermal.heatsim_test_service import call_service
    from visionsim.simulate.blender import BlenderClient

    helper = Path(__file__).with_name("transient_blender_check.py")
    with BlenderClient.spawn(executable=executable, timeout=300, service=f"{helper}:TransientCaseService") as client:
        report = call_service(client, "run", case, str(tmp_path))

    assert report["floor_max_rise_K"] > 1e-5
    assert len(report["renders"]) >= 4

    import Imath
    import numpy as np
    import OpenEXR

    for modality in ("temperature", "thermal_radiance"):
        files = sorted((tmp_path / "renders" / modality).rglob("*.exr"))
        assert len(files) == 2, files
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
        assert not np.array_equal(*rendered), f"Rendered {modality} image did not change over time"


@pytest.mark.slow
@pytest.mark.parametrize("render_domain", ["VERTEX", "TEXEL"])
def test_transient_rpc_workers(render_domain, tmp_path):
    from dataclasses import asdict

    from visionsim.simulate.blender import BlenderClients
    from visionsim.simulate.config import RenderConfig, ThermalConfig
    from visionsim.simulate.job import render_job

    executable = os.environ.get("VSIM_BLENDER") or shutil.which("blender")
    if executable is None:
        pytest.skip("Blender is unavailable")
    helper = Path(__file__).with_name("transient_blender_check.py")
    repo = helper.resolve().parents[3]
    source = tmp_path / "rpc-fixture.blend"
    assignments = tmp_path / "materials.json"
    assignments.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "defaults": {"preset": "plaster"},
                "materials": {},
            }
        )
    )
    thermal = ThermalConfig(
        mode="TRANSIENT",
        device="cpu",
        render_domain=render_domain,
        atlas_texel_density=100000,
        atlas_tile_min=16,
        atlas_tile_max=64,
        simulation_start_frame=1,
        simulation_end_frame=3,
        substeps_per_frame=2,
        laplacian_update_every_steps=2,
        irradiance_bake_mode="ONCE",
        irradiance_scale=0,
        bake_samples=1,
        irradiance_texture_size=32,
        cache_root=tmp_path / "cache",
        assignments=assignments,
        preview=False,
    )
    with BlenderClients.spawn(
        jobs=2,
        timeout=60,
        executable=executable,
        factory_startup=True,
        service=f"{helper}:ValidationService",
        log=tmp_path / "logs",
        env={"PYTHONPATH": str(repo), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
    ) as clients:
        clients[0].make_fixture(str(source))
        clients[0].initialize(str(source), str(tmp_path / "once"))
        clients[0].prepare_transient(asdict(thermal), render_outputs=False)
        once = json.loads(clients[0].transient_state(3))
        assert once["stats"] == {"steps": 4, "bakes": 1, "laplacian_rebuilds": 3}
        thermal.irradiance_bake_mode = "EVERY_N_FRAMES"
        thermal.irradiance_bake_every_n_frames = 1
        config = RenderConfig(
            include_frames=False,
            include_thermal=True,
            thermal=thermal,
            height=48,
            width=64,
            device_type="cpu",
            max_samples=2,
            use_denoising=False,
            use_motion_blur=False,
            allow_skips=False,
        )
        config_file = tmp_path / "render.yaml"
        config.to_file(config_file)
        config = RenderConfig.from_file(config_file)
        render_job(clients, str(source), str(tmp_path / "renders"), config, frame_start=1, frame_end=3, frame_step=2)
        states = [json.loads(client.transient_state(3)) for client in clients]
        if render_domain == "TEXEL":
            assert states[0]["atlas_root"] != states[1]["atlas_root"], "Workers share a writable atlas"
            assert all(Path(state["atlas_root"]).is_dir() for state in states)
        comparable = [{k: v for k, v in state.items() if k != "atlas_root"} for state in states]
        assert comparable[0] == comparable[1], "Render workers did not attach identical thermal history"
        assert states[0]["stats"] == {"steps": 4, "bakes": 3, "laplacian_rebuilds": 3}
        assert states[0]["module"] == str(repo / "visionsim/simulate/blender.py")
        assert len(list((tmp_path / "cache").glob("transient-*"))) == 2
        assert len(list((tmp_path / "renders/temperature").rglob("*.exr"))) == 2
        (tmp_path / "rpc-report.json").write_text(json.dumps(states, indent=2))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("helper_name", "service_name"),
    [
        ("transient_cache_check.py", "TransientCacheCheckService"),
        ("transient_geometry_check.py", "TransientGeometryCheckService"),
        ("transient_reuse_guard_check.py", "TransientReuseGuardCheckService"),
    ],
)
def test_transient_scene_fingerprint_blender(helper_name, service_name, executable, tmp_path):
    from tests.simulate.thermal.heatsim_test_service import call_service
    from visionsim.simulate.blender import BlenderClient

    helper = Path(__file__).with_name(helper_name)
    with BlenderClient.spawn(executable=executable, timeout=300, service=f"{helper}:{service_name}") as client:
        if service_name == "TransientGeometryCheckService":
            result = call_service(client, "run")
        else:
            result = call_service(client, "run", str(tmp_path))
    assert result["ok"] is True


@pytest.mark.slow
def test_transient_prepares_with_unfingerprintable_image(executable, tmp_path):
    from tests.simulate.thermal.heatsim_test_service import SERVICE_PATH, call_service
    from visionsim.simulate.blender import BlenderClient

    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        result = call_service(client, "prepare_tiled_transient", str(tmp_path))
    assert result["has_digest"] is False


@pytest.mark.slow
def test_rolling_ball_transient(executable, tmp_path):
    from tests.simulate.thermal.heatsim_test_service import SERVICE_PATH, call_service
    from visionsim.simulate.blender import BlenderClient

    with BlenderClient.spawn(executable=executable, timeout=300, service=SERVICE_PATH) as client:
        report = call_service(client, "transient_rolling_ball", str(tmp_path))

    assert report["stats"]["bakes"] == 1
    assert report["floor_max_rise_K"] > 1e-5
    assert report["temperature_renders"] == 2
    assert report["thermal_radiance_renders"] == 2


@pytest.mark.slow
def test_static_render_from_config_file(tmp_path):
    import hashlib

    import Imath
    import numpy as np
    import OpenEXR

    from visionsim.simulate.blender import BlenderClient
    from visionsim.simulate.config import RenderConfig, ThermalConfig
    from visionsim.simulate.job import frame_job

    executable = os.environ.get("VSIM_BLENDER") or shutil.which("blender")
    if executable is None:
        pytest.skip("Blender is unavailable")
    helper = Path(__file__).with_name("transient_blender_check.py")
    repo = helper.resolve().parents[3]
    source = tmp_path / "static.blend"
    config = RenderConfig(
        include_thermal=True,
        previews=False,
        height=48,
        width=64,
        device_type="cpu",
        max_samples=2,
        use_denoising=False,
        use_motion_blur=False,
        allow_skips=False,
        thermal=ThermalConfig(
            mode="STATIC",
            device="cpu",
            render_domain="VERTEX",
            sim_time_s=0.1,
            timestep_s=0.05,
            irradiance_scale=0,
            bake_samples=1,
            irradiance_texture_size=32,
            cache_root=tmp_path / "cache",
            radiance=True,
        ),
    )
    config_file = tmp_path / "static.yaml"
    config.to_file(config_file)
    config = RenderConfig.from_file(config_file)
    root = tmp_path / "renders"
    with BlenderClient.spawn(
        executable=executable,
        timeout=60,
        factory_startup=True,
        service=f"{helper}:ValidationService",
        log=tmp_path / "logs",
        env={"PYTHONPATH": str(repo), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
    ) as client:
        client.make_fixture(str(source))
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        frame_job(client, str(source), str(root), config, frame_number=1)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    for modality in ("temperature", "thermal_radiance"):
        files = list((root / modality).rglob("*.exr"))
        assert len(files) == 1
        exr = OpenEXR.InputFile(str(files[0]))
        try:
            header = exr.header()
            window = header["dataWindow"]
            assert (window.max.x - window.min.x + 1, window.max.y - window.min.y + 1) == (64, 48)
            channel = "V" if "V" in header["channels"] else next(iter(header["channels"]))
            values = np.frombuffer(exr.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT)), np.float32)
            assert np.isfinite(values).all()
            assert values.max() > 0
            if modality == "temperature":
                assert 350 < values.max() < 361
        finally:
            exr.close()
    assert not list((root / "previews").rglob("*.png")), "Global previews=False was ignored"
