"""Transient preparation lifecycle tests."""

from __future__ import annotations

from tests.simulate.thermal.transient_prepare_fakes import install_fakes, run_prepare
from visionsim.simulate.heatsim import cache


def test_prune_transient_removes_completed_trajectories(tmp_path, monkeypatch):
    """Explicit pruning bounds completed trajectories without automatic deletes."""
    install_fakes(monkeypatch, blocker="animation drivers with external dependencies")
    run_prepare(tmp_path)
    run_prepare(tmp_path)
    assert len(list(tmp_path.glob("transient-*/manifest.json"))) == 2
    for trajectory in tmp_path.glob("transient-*"):
        playback = trajectory / "playback-test" / "atlas_test"
        playback.mkdir(parents=True)
        (playback / "atlas_temperature.exr").write_bytes(b"x")
    assert len(list(tmp_path.rglob("atlas_temperature.exr"))) == 2

    assert len(cache.prune_transient_trajectories(tmp_path, keep=1)) == 1
    assert len(list(tmp_path.glob("transient-*"))) == 1
    assert len(list(tmp_path.rglob("atlas_temperature.exr"))) == 1

    assert len(cache.prune_transient_trajectories(tmp_path)) == 1
    assert not list(tmp_path.glob("transient-*"))
    assert not list(tmp_path.rglob("atlas_temperature.exr"))
