from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from importlib.metadata import Distribution
from pathlib import Path

import psutil
import pytest

from visionsim.simulate import install_dependencies
from visionsim.simulate.blender import BlenderClient

SCENE = Path(__file__).parent / "test_files" / "scenes" / "cube.blend"


def pytest_addoption(parser):
    parser.addoption(
        "--executable", type=str, default=None, help="Path to Blender executable. Defaults to one found on $PATH."
    )


@pytest.fixture(scope="session")
def executable(pytestconfig):
    executable_path = pytestconfig.getoption("--executable")

    direct_url = Distribution.from_name("visionsim").read_text("direct_url.json")
    pkg_is_editable = json.loads(direct_url).get("dir_info", {}).get("editable", False)

    if not pkg_is_editable:
        warnings.warn(RuntimeWarning("Package visionsim should be installed as editable for development!"))

    if any("blender" in proc.name().lower() for proc in psutil.process_iter()):
        # Note: If there's a previous BlenderServer that's running, we might connect to that
        #   one instead, and it might be running with an outdated visionsim version!
        # TODO: How to fix this race condition in the general case?
        raise RuntimeError(
            "At least on Blender instance is already running, please close all instances "
            "to ensure we do not connect to a stale one."
        )

    install_dependencies(executable=executable_path, editable=True)

    # Resolve to a real path. `--executable` defaults to None, and CI invokes a bare
    # `pytest`, so tests that interpolate this into an argv (subprocess.run([str(exe), ...]))
    # would otherwise spawn the literal string "None" and fail with FileNotFoundError.
    # Tests that pass it to BlenderClient.spawn() tolerate None because that resolves
    # against $PATH itself; argv-interpolating tests do not.
    resolved = executable_path or shutil.which("blender")
    if resolved is None:
        pytest.skip("No Blender executable: pass --executable=/path/to/blender or put `blender` on PATH.")
    return resolved


def _load_cube_scene(client, tmpdir: Path) -> None:
    """Load the test cube scene into ``tmpdir``, rescaled/shortened and at a low resolution."""
    client.initialize(SCENE.resolve(), tmpdir.resolve())
    client.move_keyframes(scale=1 / 5)
    client.set_animation_range(10, 15)
    client.set_resolution(50, 50)


@pytest.fixture(scope="session")
def cube_dataset(tmp_path_factory, executable) -> Path:
    # Note: If this fails and you're using flatpak, it might be because
    #   the application doesn't have read/write access to /tmp!
    # Note: This fixture also renders the thermal modality (prepare_thermal +
    #   include_thermal), so a regression in the thermal pipeline will surface
    #   across all cube_dataset-based tests, not only the thermal-specific ones.
    tmpdir = tmp_path_factory.mktemp("renders")
    log_dir = tmp_path_factory.mktemp("logs")

    with BlenderClient.spawn(
        executable=executable, timeout=30, log=sys.stdout if os.getenv("CI") == "true" else log_dir
    ) as client:
        _load_cube_scene(client, tmpdir)
        client.include_composites()
        client.include_frames()
        client.include_depths()
        client.include_normals()
        client.include_flows()
        client.include_segmentations()
        client.include_materials()
        client.include_diffuse_pass()
        client.include_specular_pass()
        client.include_points()
        client.prepare_thermal(device="cpu", bake_samples=4, irradiance_texture_size=64)
        client.include_thermal(radiance=True, preview=True)
        client.render_animation()
        client.save_file(tmpdir / "cube_out.blend")
    return tmpdir


def _blender_version(executable: str | os.PathLike | None) -> tuple[int, int, int]:
    """Query the ``(major, minor, patch)`` version of the blender installation under test."""
    cmd = [str(executable)] if executable else ["blender"]
    proc = subprocess.run([*cmd, "--version"], capture_output=True, text=True, check=False)
    match = re.search(r"Blender\s+(\d+)\.(\d+)\.(\d+)", proc.stdout)
    if not match:
        pytest.skip(f"Could not determine blender version from: {proc.stdout.strip()!r}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def _render_playblast(tmp_path_factory, executable, *, video: bool) -> Path:
    """Render a playblast of the test scene once, for a session-scoped fixture to hand out.

    Note: non-background blender opens a real window and the playblast needs a GL context, so
      these tests require a display. Wrap the whole pytest invocation in `xvfb-run` if you have
      none (see the `render_playblast` CLI docstring for the exact incantation).
    """
    if not os.environ.get("DISPLAY"):
        pytest.skip("playblast rendering requires a display/GL context")
    if video and (shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None):
        pytest.skip("playblast video encoding requires ffmpeg/ffprobe")
    if _blender_version(executable) < (4, 2, 0):
        pytest.skip("playblast rendering requires blender >= 4.2")

    tmpdir = tmp_path_factory.mktemp("playblast_video" if video else "playblasts")
    log_dir = tmp_path_factory.mktemp("logs")

    with BlenderClient.spawn(
        executable=executable,
        timeout=30,
        log=sys.stdout if os.getenv("CI") == "true" else log_dir,
        background=False,
    ) as client:
        _load_cube_scene(client, tmpdir)
        client.render_playblast(video=video)
    return tmpdir


@pytest.fixture(scope="session")
def playblast_dataset(tmp_path_factory, executable) -> Path:
    return _render_playblast(tmp_path_factory, executable, video=False)


@pytest.fixture(scope="session")
def playblast_video_dataset(tmp_path_factory, executable) -> Path:
    return _render_playblast(tmp_path_factory, executable, video=True)
