"""Timing, validation, and frame-store contracts for animated heat."""

import numpy as np
import pytest

from visionsim.simulate.config import ThermalConfig
from visionsim.simulate.heatsim import cache
from visionsim.simulate.heatsim.transient import bake_due, read_frame


@pytest.mark.parametrize("substeps", [1, 2, 4])
def test_bake_times_are_independent_of_thermal_substeps(substeps):
    config = ThermalConfig(mode="TRANSIENT", substeps_per_frame=substeps, irradiance_bake_every_n_frames=3)
    times = [step / substeps for step in range(10 * substeps + 1) if bake_due(step, config)]
    assert times == [0, 3, 6, 9]


def test_once_and_every_step():
    assert [i for i in range(12) if bake_due(i, ThermalConfig(irradiance_bake_mode="ONCE"))] == [0]
    assert all(bake_due(i, ThermalConfig(irradiance_bake_mode="EVERY_STEP")) for i in range(12))


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "ANIMATED"},
        {"substeps_per_frame": 0},
        {"substeps_per_frame": 1.5},
        {"laplacian_update_every_steps": False},
        {"irradiance_bake_every_n_frames": -1},
        {"irradiance_bake_mode": "ADAPTIVE"},
        {"simulation_start_frame": 5, "simulation_end_frame": 4},
    ],
)
def test_invalid_transient_configuration(options):
    with pytest.raises(ValueError):
        ThermalConfig(**options)


def test_frame_history_is_random_access_and_missing_states_fail(tmp_path):
    session = {"root": tmp_path, "manifest": {"frame_start": 7, "frame_end": 9, "counts": {"ball": 2}}}
    for frame in (7, 9):
        cache.write_temperatures(tmp_path, f"frame-{frame}", {"ball": np.full((1, 2), 290 + frame)}, {})
    for frame in (9, 7, 9):
        np.testing.assert_array_equal(read_frame(session, frame)["ball"], np.full((1, 2), 290 + frame))
    with pytest.raises(RuntimeError, match="Missing or invalid"):
        read_frame(session, 8)
    with pytest.raises(ValueError, match="outside"):
        read_frame(session, 10)
