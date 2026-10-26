from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Literal

import pytest
import tyro
import yaml

from visionsim.cli import main
from visionsim.cli._config_file import bind_config_file
from visionsim.simulate.config import FlowsConfig, RenderConfig
from visionsim.utils.config_file import ConfigIO


@dataclass
class ToyConfig(ConfigIO):
    jobs: int = 1
    mode: Literal["a", "b"] = "a"
    tags: list[str] | None = None
    offset: tuple[float, float, float] | None = None


@dataclass
class PlainConfig:
    jobs: int = 1


CAPTURED: list[ToyConfig] = []
OBSERVED: dict[str, bool] = {}


def toy_command(input_dir: Path, /, config: ToyConfig, flag: bool = False) -> None:
    CAPTURED.append(config)


def observed_command(input_dir: Path, /, config: ToyConfig) -> None:
    OBSERVED["dump_exists"] = config.dump is not None and Path(config.dump).exists()
    CAPTURED.append(config)


def render_command(input_dir: Path, /, config: RenderConfig) -> None:
    pass


def plain_command(input_dir: Path, /, config: PlainConfig) -> None:
    pass


@pytest.fixture(autouse=True)
def _clear_captured():
    CAPTURED.clear()
    OBSERVED.clear()


def test_load_nested_render_config(tmp_path):
    path = tmp_path / "render.yaml"
    path.write_text(
        dedent(
            """
            jobs: 4
            include-depths: true
            previews: false
            executable: /opt/blender
            frames:
              file-format: PNG
              bit-depth: 16
            flows:
              direction: both
            """
        )
    )

    config = RenderConfig.from_file(path)

    assert config.jobs == 4
    assert config.include_depths is True
    assert config.previews is False
    assert config.executable == Path("/opt/blender")
    assert isinstance(config.flows, FlowsConfig)
    assert config.flows.direction == "both"
    assert config.frames.bit_depth == 16


def test_load_json(tmp_path):
    path = tmp_path / "render.json"
    path.write_text('{"jobs": 3, "flows": {"direction": "backward"}}')

    config = RenderConfig.from_file(path)

    assert config.jobs == 3
    assert config.flows.direction == "backward"


def test_load_bytes_size(tmp_path):
    path = tmp_path / "render.yaml"
    path.write_text("max_job_vram: 4GB\n")

    assert RenderConfig.from_file(path).max_job_vram == 4 * 1024**3


@pytest.mark.parametrize(
    "content, match",
    [
        ("unknown: 1\n", "unknown option"),
        ("load: other.yaml\n", "command-line options"),
    ],
)
def test_load_rejects_invalid_structure(tmp_path, content, match):
    path = tmp_path / "render.yaml"
    path.write_text(content)

    with pytest.raises(ValueError, match=match):
        RenderConfig.from_file(path)


@pytest.mark.parametrize(
    "content",
    [
        "jobs: many\n",
        "previews: 1\n",
        "jobs: 1\nflows: nope\n",
        "flows:\n  direction: sideways\n",
        "camera_offset: [1, 2]\n",
    ],
)
def test_load_leaves_value_parsing_to_tyro(tmp_path, content, capsys):
    path = tmp_path / "render.yaml"
    path.write_text(content)

    with pytest.raises(SystemExit):
        RenderConfig.from_file(path)

    captured = capsys.readouterr()
    assert captured.err


def test_load_tricky_values(tmp_path):
    path = tmp_path / "render.yaml"
    path.write_text(
        dedent(
            """
            flows:
              preview: false
            use_motion_blur: false
            camera_offset: [-1, 2, 3]
            executable: -weird
            addons: []
            """
        )
    )

    config = RenderConfig.from_file(path)

    assert config.flows.preview is False
    assert config.use_motion_blur is False
    assert config.camera_offset == (-1.0, 2.0, 3.0)
    assert config.executable == Path("-weird")
    assert config.addons == []


def test_load_rejects_non_mapping_root(tmp_path):
    path = tmp_path / "render.yaml"
    path.write_text("- jobs\n")

    with pytest.raises(TypeError, match="mapping"):
        RenderConfig.from_file(path)


def test_dump_roundtrip(tmp_path):
    config = RenderConfig(jobs=2, include_all=True, flows=FlowsConfig(direction="both"))
    first = tmp_path / "first.yaml"
    second = tmp_path / "nested" / "second.yaml"

    config.to_file(first)
    loaded = RenderConfig.from_file(first)
    loaded.to_file(second)

    assert loaded == config
    assert second.exists()
    assert second.read_text() == first.read_text()


def test_dump_omits_io_options(tmp_path):
    path = tmp_path / "dump.yaml"

    ToyConfig(load=Path("in.yaml"), dump=Path("out.yaml")).to_file(path)

    data = yaml.safe_load(path.read_text())
    assert set(data) == {"jobs", "mode", "tags", "offset"}


def test_load_override_precedence(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("jobs: 4\nmode: b\n")
    argv = ["input", "--config.load", str(path), "--config.jobs", "8"]

    tyro.cli(bind_config_file(toy_command, argv), args=argv, console_outputs=False)

    assert CAPTURED[-1].jobs == 8
    assert CAPTURED[-1].mode == "b"
    assert CAPTURED[-1].load == path


def test_load_equals_form(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("jobs: 6\n")
    argv = ["input", f"--config.load={path}"]

    tyro.cli(bind_config_file(toy_command, argv), args=argv, console_outputs=False)

    assert CAPTURED[-1].jobs == 6


def test_dump_is_written_before_command_runs(tmp_path):
    path = tmp_path / "dump.yaml"
    argv = ["input", "--config.dump", str(path)]

    tyro.cli(bind_config_file(observed_command, argv), args=argv, console_outputs=False)

    assert OBSERVED["dump_exists"] is True
    assert path.exists()
    assert CAPTURED[-1].jobs == 1


def test_load_and_dump_round_trip(tmp_path):
    source = tmp_path / "source.yaml"
    source.write_text("jobs: 4\nmode: b\n")
    target = tmp_path / "effective.yaml"
    argv = ["input", "--config.load", str(source), "--config.dump", str(target), "--config.jobs", "8"]

    tyro.cli(bind_config_file(toy_command, argv), args=argv, console_outputs=False)

    data = yaml.safe_load(target.read_text())
    assert data["jobs"] == 8
    assert data["mode"] == "b"
    assert "load" not in data and "dump" not in data


def test_no_config_flags_leaves_command_untouched():
    assert bind_config_file(toy_command, ["input"]) is toy_command
    assert bind_config_file(toy_command, ["--help"]) is toy_command
    assert bind_config_file(plain_command, ["input", "--config.load", "x.yaml"]) is plain_command


def test_help_short_circuits_before_loading(tmp_path):
    missing = tmp_path / "missing.yaml"

    assert bind_config_file(toy_command, ["input", "--help", "--config.load", str(missing)]) is toy_command


def test_dump_wrapper_preserves_signature_and_docstring(tmp_path):
    wrapped = bind_config_file(observed_command, ["input", "--config.dump", str(tmp_path / "dump.yaml")])

    assert inspect.signature(wrapped) == inspect.signature(observed_command)
    assert wrapped.__doc__ == observed_command.__doc__


def test_nested_config_io_option_is_not_available():
    with pytest.raises(SystemExit):
        tyro.cli(render_command, args=["input", "--config.frames.load", "x.yaml"], console_outputs=False)


def test_help_lists_config_file_options(capsys):
    with pytest.raises(SystemExit):
        tyro.cli(render_command, args=["--help"])

    help_text = capsys.readouterr().out
    assert "--config.load" in help_text
    assert "--config.dump" in help_text


def test_main_help_lists_config_file_options(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["visionsim", "blender.render-animation", "--help"])

    with pytest.raises(SystemExit):
        main()

    help_text = capsys.readouterr().out
    assert "--config.load" in help_text
    assert "--config.dump" in help_text


def test_resolved_expands_include_all_without_mutating():
    config = RenderConfig(include_all=True)

    resolved = config.resolved()

    assert resolved.include_depths is True
    assert resolved.include_thermal is True
    assert config.include_depths is False
    assert resolved.resolved() == resolved


def test_resolved_gates_previews():
    config = RenderConfig(include_depths=True, previews=False)

    resolved = config.resolved()

    assert resolved.depths.preview is False
    assert config.depths.preview is True
