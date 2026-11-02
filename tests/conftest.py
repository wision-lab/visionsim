from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from importlib.metadata import Distribution
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import pytest

from visionsim.simulate import install_dependencies
from visionsim.simulate.blender import BlenderClient

SCENE = Path(__file__).parent / "test_files" / "scenes" / "cube.blend"


def _write_dataset(root: Path, frames: int = 1, *, depths: list[float] | None = None, size: int = 32) -> Path:
    """Write a minimal Blender-free dataset: images, a ``transforms.json`` and depths.

    The image is non-uniform and non-zero, because the emulators scale the reflection
    by the albedo: a black frame would decode to an all-zero measurement array and make
    the finiteness and preview assertions vacuous.

    Args:
        root: Directory to write the dataset to, created if needed.
        frames: Number of frames to write. Ignored when ``depths`` is given, which
            fixes the count at one frame per depth.
        depths: Depth of each frame, in metres, written as ``depth_NNNN.npy`` beside its
            frame and referenced from it. None for a dataset with images only.
        size: Side of the square images, and of the map saved for each depth.

    Returns:
        ``root``.
    """
    root.mkdir(parents=True, exist_ok=True)
    image: npt.NDArray[np.uint8] = np.arange(size * size * 3, dtype=np.uint8).reshape(size, size, 3) + 1
    entries = []
    for i in range(len(depths) if depths is not None else frames):
        iio.imwrite(root / f"{i:04}.png", image)
        entry = {"file_path": f"{i:04}.png", "transform_matrix": np.eye(4).tolist()}
        if depths is not None:
            np.save(root / f"depth_{i:04}.npy", np.full((size, size), depths[i], np.float32)[None])
            entry["depth_file_path"] = f"depth_{i:04}.npy"
        entries.append(entry)
    (root / "transforms.json").write_text(
        json.dumps(
            {
                "camera_model": "OPENCV",
                "fl_x": float(size),
                "fl_y": float(size),
                "cx": size / 2,
                "cy": size / 2,
                "h": size,
                "w": size,
                "frames": entries,
            }
        )
    )
    return root


@pytest.fixture
def write_dataset():
    """Expose :func:`_write_dataset` as a fixture, which is how the tests take it."""
    # A fixture cannot take arguments - pytest reads its parameters as fixture names - so
    # this hands back :func:`_write_dataset` for tests to call with their own paths.
    return _write_dataset


def pytest_addoption(parser):
    parser.addoption(
        "--executable", type=str, default=None, help="Path to Blender executable. Defaults to one found on $PATH."
    )


@pytest.fixture(scope="session")
def executable(pytestconfig):
    # Resolve first, in the same order as `inv test` (--executable, $VSIM_BLENDER, $PATH;
    # docs/source/development.rst). `--executable` defaults to None and CI runs a bare
    # `pytest`, so tests that interpolate this into an argv (subprocess.run([str(exe), ...]))
    # would spawn the literal "None"; tests that hand it to BlenderClient.spawn() accept None
    # because that resolves against $PATH. This runs before install_dependencies(), which
    # raises on a missing executable and would otherwise hide the skip below.
    executable_path = pytestconfig.getoption("--executable")
    resolved = executable_path or os.environ.get("VSIM_BLENDER") or shutil.which("blender")
    if resolved is None:
        pytest.skip("No Blender executable: pass --executable=/path/to/blender or put `blender` on PATH.")

    # direct_url.json is absent for installs from an index (a plain `pip install visionsim`),
    # so a missing file is itself proof of a non-editable install.
    direct_url = Distribution.from_name("visionsim").read_text("direct_url.json")
    pkg_is_editable = json.loads(direct_url).get("dir_info", {}).get("editable", False) if direct_url else False

    if not pkg_is_editable:
        # Blender is told to install visionsim editable from this interpreter's import
        # path, so a non-editable install would have it import a frozen site-packages
        # copy while the client runs the working tree - tests would pass against stale code.
        raise RuntimeError(
            "visionsim must be installed as editable for development, otherwise Blender "
            "imports a stale copy of the package and the tests no longer reflect your "
            "working tree. Install with `pip install -e . --group dev` (or `uv sync`)."
        )

    # Hand install_dependencies() the resolved path: it would otherwise fall back to a bare
    # `blender` on $PATH and fail under a $VSIM_BLENDER-only configuration.
    install_dependencies(executable=resolved, editable=True)

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
