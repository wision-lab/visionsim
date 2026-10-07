"""Tests for iToF measurement simulation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from tests.emulate.itof import D_MAX, PERIOD, codes
from visionsim.emulate.itof import compute_correlation_function, simulate_measurements
from visionsim.emulate.itof.coding import make_conv_sinusoidal_codes, make_delta_sinusoidal_codes

# ──────────────────────────── Correlation function ─────────────────────────


def test_correlation_shape():
    mod, ref = make_conv_sinusoidal_codes(4, 500)
    assert compute_correlation_function(mod[0], ref[0], 1e-9).shape == (500,)
    assert compute_correlation_function(mod[0], ref[0], 1e-9, n_depths=100).shape == (100,)


def test_correlation_delta_mod_reproduces_reference_code():
    """An impulse modulation correlates to the reference code itself."""
    _, ref = make_delta_sinusoidal_codes(3, 200)
    mod = np.zeros(200)
    mod[0] = 1.0
    time_resolution = 1e-9
    assert compute_correlation_function(mod, ref[0], time_resolution) == pytest.approx(200 * ref[0] * time_resolution)


def test_correlation_periodic_continuation():
    """More bins than the period repeat the correlation function."""
    mod, ref = make_conv_sinusoidal_codes(4, 300)
    one_period = compute_correlation_function(mod[0], ref[0], 1e-9)
    two_periods = compute_correlation_function(mod[0], ref[0], 1e-9, n_depths=600)
    assert two_periods[:300] == pytest.approx(one_period)
    assert two_periods[300:] == pytest.approx(one_period)
    # the sample at max_depth equals the sample at depth 0
    with_wrap = compute_correlation_function(mod[0], ref[0], 1e-9, n_depths=301)
    assert with_wrap[-1] == pytest.approx(with_wrap[0])


def test_correlation_non_negative_for_sinusoidal_codes():
    mod, ref = make_conv_sinusoidal_codes(4, 500)
    assert np.all(compute_correlation_function(mod[0], ref[0], 1e-9) >= 0)


def test_correlation_mismatched_code_lengths():
    with pytest.raises(ValueError, match="same length"):
        compute_correlation_function(np.ones(4), np.ones(5), 1e-9)


@pytest.mark.parametrize("n_depths", [0, -5])
def test_correlation_invalid_n_depths(n_depths: int):
    with pytest.raises(ValueError, match="positive number of depth bins"):
        compute_correlation_function(np.ones(4), np.ones(4), 1e-9, n_depths=n_depths)


# ─────────────────────────── Measurement simulation ────────────────────────


def test_simulation_output_shape_1d_and_2d():
    mod, ref = codes("convSin", 4, 501)
    depths_1d = np.array([0.3, 0.7, 1.2])
    assert simulate_measurements(depths_1d, np.full(3, 0.5), mod, ref, PERIOD).shape == (4, 3)
    depths_2d = np.full((8, 6), 0.5)
    assert simulate_measurements(depths_2d, np.full((8, 6), 0.5), mod, ref, PERIOD).shape == (4, 8, 6)


@pytest.mark.parametrize("kwargs", [{"exposure_time": 2.5}, {"light_power": 3.0}, {"albedo": None}])
def test_simulation_linear_terms(kwargs: dict):
    mod, ref = codes("convSin", 4, 501)
    depths = np.array([0.3, 0.7])
    albedo = np.array([0.4, 0.9])
    baseline = simulate_measurements(depths, albedo, mod, ref, PERIOD)
    if "albedo" in kwargs:
        scaled = simulate_measurements(depths, albedo * 2, mod, ref, PERIOD)
        assert scaled == pytest.approx(2 * baseline)
    else:
        scaled = simulate_measurements(depths, albedo, mod, ref, PERIOD, **kwargs)
        assert scaled == pytest.approx(next(iter(kwargs.values())) * baseline)


def test_simulation_inverse_square_falloff_and_phase_wrap():
    """A depth one period deeper has the same phase but a 1/d**2 falloff."""
    mod, ref = codes("convSin", 4, 1001)
    depths = np.array([0.25, 0.25 + D_MAX, 0.5, 0.5 + D_MAX])
    measurements = simulate_measurements(depths, np.full(4, 0.8), mod, ref, PERIOD)
    for i in (0, 2):
        assert measurements[:, i + 1] / measurements[:, i] == pytest.approx((depths[i] / depths[i + 1]) ** 2, rel=1e-6)


def test_simulation_ambient_term_is_depth_independent():
    mod, ref = codes("convSin", 4, 501)
    depths = np.array([0.3, 1.1])
    ambient_only = simulate_measurements(depths, np.full(2, 0.5), mod, ref, PERIOD, ambient_power=2.0, light_power=0.0)
    assert ambient_only[:, 0] == pytest.approx(ambient_only[:, 1])
    assert ambient_only == pytest.approx(
        2 * simulate_measurements(depths, np.full(2, 0.5), mod, ref, PERIOD, ambient_power=1.0, light_power=0.0)
    )


def test_simulation_matches_analytic_model():
    mod, ref = codes("deltaSin", 4, 501)
    depths = np.array([0.4, 1.0])
    albedos = np.array([0.7, 0.3])
    exposure_time, ambient_power, light_power = 0.5, 2.0, 3.0
    measurements = simulate_measurements(
        depths,
        albedos,
        mod,
        ref,
        PERIOD,
        exposure_time=exposure_time,
        ambient_power=ambient_power,
        light_power=light_power,
    )
    n = mod.shape[1]
    time_resolution = PERIOD / n
    period = PERIOD
    distances = np.linspace(0, D_MAX, n, endpoint=False)
    for i in range(mod.shape[0]):
        correlation = compute_correlation_function(mod[i], ref[i], time_resolution, n_depths=n)
        kappa = ref[i].sum() * time_resolution
        expected = (exposure_time / period) * (
            light_power * (albedos / depths**2) * PchipInterpolator(distances, correlation)(depths % D_MAX)
            + ambient_power * kappa * albedos
        )
        assert measurements[i] == pytest.approx(expected)


def test_simulation_non_negative_without_ambient():
    mod, ref = codes("convSin", 4, 501)
    measurements = simulate_measurements(np.array([0.2, 0.9]), np.full(2, 0.5), mod, ref, PERIOD)
    assert np.all(measurements >= 0)


def test_simulation_zero_depth_is_finite():
    mod, ref = codes("convSin", 4, 501)
    measurements = simulate_measurements(np.array([0.0, 1e-9]), np.full(2, 0.5), mod, ref, PERIOD)
    assert np.all(np.isfinite(measurements))


def test_simulation_mismatched_code_lengths():
    mod, _ = codes("convSin", 4, 501)
    _, ref = codes("convSin", 4, 502)
    with pytest.raises(ValueError, match="same length"):
        simulate_measurements(np.array([0.5]), np.array([0.5]), mod, ref, PERIOD)
