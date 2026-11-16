"""Dynamic Blender regression checks using isolated imports and output paths."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.parametrize("case", ["vertex", "finite", "texel", "deform", "lighting", "asset"])
def test_transient_blender(case, tmp_path):
    # Do not use the suite's executable fixture here: it installs the checkout
    # editable into Blender's shared Python. The helper controls sys.path itself.
    executable = os.environ.get("VSIM_BLENDER") or shutil.which("blender")
    if executable is None:
        pytest.skip("Blender is unavailable")
    if case == "asset" and not Path("/data/sriram/blender_files/test_ball_rolling.blend").is_file():
        pytest.skip("Optional rolling-ball asset is unavailable")
    helper = Path(__file__).with_name("transient_blender_check.py")
    result = subprocess.run(
        [
            executable,
            "--background",
            "--factory-startup",
            "--threads",
            "2",
            "--python-exit-code",
            "1",
            "--python",
            str(helper),
            "--",
            case,
            str(tmp_path),
        ],
        env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    (tmp_path / "blender.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["floor_max_rise_K"] > 1e-5
    assert len(report["renders"]) >= 4


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
def test_transient_scene_fingerprint_blender(tmp_path):
    executable = os.environ.get("VSIM_BLENDER") or shutil.which("blender")
    if executable is None:
        pytest.skip("Blender is unavailable")
    helper = Path(__file__).with_name("transient_cache_check.py")
    result = subprocess.run(
        [
            executable,
            "--background",
            "--factory-startup",
            "--threads",
            "2",
            "--python-exit-code",
            "1",
            "--python",
            str(helper),
            "--",
            str(tmp_path),
        ],
        env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    (tmp_path / "blender.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    assert "TRANSIENT_CACHE_VALIDATION" in result.stdout
