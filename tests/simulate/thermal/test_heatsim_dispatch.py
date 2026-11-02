"""Check that render_job sends one validated thermal configuration to Blender."""

from __future__ import annotations

from dataclasses import asdict

from visionsim.simulate.config import RenderConfig
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
