"""Check that render_job sends one validated thermal configuration to Blender."""

from __future__ import annotations

import logging
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from visionsim.simulate.config import RenderConfig, ThermalConfig
from visionsim.simulate.job import render_job


class _RecordingClient:
    """Fake Blender client: records every method call as ``(name, args, kwargs)``.

    Every attribute access resolves to a no-op recorder, so ``render_job`` can run
    end-to-end without a real Blender process while we inspect the call sequence.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str):
        def _record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return _record


def test_render_job_dispatches_one_thermal_config():
    client = _RecordingClient()
    config = RenderConfig(include_thermal=True)

    render_job(client, "scene.blend", "out", config=config, dry_run=True)

    names = [name for name, _, _ in client.calls]
    assert names.count("configure_thermal") == 1, names

    expected = asdict(config.thermal)
    _name, args, kwargs = client.calls[names.index("configure_thermal")]
    assert args == (expected,) and kwargs == {}


def test_render_job_skips_thermal_when_disabled():
    client = _RecordingClient()
    config = RenderConfig(include_thermal=False)

    render_job(client, "scene.blend", "out", config=config, dry_run=True)

    names = [name for name, _, _ in client.calls]
    assert "configure_thermal" not in names, names


def test_render_job_resolves_include_all_and_previews():
    client = _RecordingClient()
    config = RenderConfig(include_all=True, previews=False)

    render_job(client, "scene.blend", "out", config=config, dry_run=True)

    names = [name for name, _, _ in client.calls]
    assert "include_depths" in names
    assert "configure_thermal" in names
    _name, _args, kwargs = client.calls[names.index("include_flows")]
    assert kwargs["preview"] is False
    assert config.include_depths is False


def test_render_job_logs_resolved_config(caplog):
    config = RenderConfig(include_all=True)

    with caplog.at_level(logging.DEBUG, logger="visionsim.simulate.job"):
        render_job(_RecordingClient(), "scene.blend", "out", config=config, dry_run=True)

    assert "Resolved configuration" in caplog.text
    assert "include_depths: true" in caplog.text


@pytest.mark.parametrize("mode", ["STATIC", "TRANSIENT"])
def test_precompute_and_render_resolve_identical_thermal_settings(tmp_path, monkeypatch, mode):
    from visionsim.cli.blender import heatsim_solve

    source = tmp_path / "scene.blend"
    source.touch()
    clients = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = clients
    monkeypatch.setattr("visionsim.simulate.blender.BlenderClients.spawn", lambda **kwargs: context)
    monkeypatch.setattr("visionsim.cli._run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    config = RenderConfig(include_thermal=True, previews=False, thermal=ThermalConfig(mode=mode))
    heatsim_solve(source, config)
    precomputed = clients.heatsim_solve_config.call_args.args[0]

    render_client = _RecordingClient()
    render_job(render_client, str(source), str(tmp_path / "renders"), config, dry_run=True)
    rendered = next(args[0] for name, args, _ in render_client.calls if name == "configure_thermal")
    assert precomputed == rendered
    assert precomputed["preview"] is False
    assert config.thermal.preview is True
