"""Transient trajectory reuse contract."""

from __future__ import annotations

import pytest

from tests.simulate.thermal.transient_prepare_fakes import install_fakes, run_prepare


@pytest.mark.parametrize("change", [{"preview": False}, {"radiance": False}, {"sim_time_s": 2.0}])
def test_transient_reuse_ignores_render_only_settings(tmp_path, monkeypatch, change):
    """A completed trajectory should be reused when only render-only settings change."""
    install_fakes(monkeypatch)
    run_prepare(tmp_path)
    run_prepare(tmp_path, change)

    trajectories = list(tmp_path.glob("transient-*/manifest.json"))
    assert len(trajectories) == 1
