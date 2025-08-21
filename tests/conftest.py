import json
import warnings
from importlib.metadata import Distribution
from pathlib import Path

import psutil
import pytest

from visionsim.simulate import install_dependencies
from visionsim.simulate.blender import BlenderClient


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
    return executable_path


@pytest.fixture(scope="session")
def cube_dataset(tmp_path_factory, executable):
    # Note: If this fails and you're using flatpak, it might be because
    #   the application doesn't have read/write access to /tmp!
    tmpdir = tmp_path_factory.mktemp("renders")
    log_dir = tmp_path_factory.mktemp("logs")
    scene = Path(__file__).parent / "test_files" / "scenes" / "cube.blend"

    with BlenderClient.spawn(executable=executable, timeout=30, log_dir=log_dir) as client:
        client.initialize(scene.resolve(), tmpdir.resolve())
        client.move_keyframes(scale=1 / 5)
        client.set_animation_range(10, 15)
        client.set_resolution(50, 50)
        client.include_depths()
        client.include_normals()
        client.include_flows()
        client.include_segmentations()
        transforms = client.render_animation()

        with open(tmpdir / "transforms.json", "w") as f:
            json.dump(transforms, f, indent=2)

        client.save_file(tmpdir / "cube_out.blend")
    return tmpdir
