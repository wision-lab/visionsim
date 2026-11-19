"""Tests for scene-level dToF transient generation and emulation."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
import scipy.constants
import torch

from tests.emulate.dtof import make_generator
from visionsim.emulate.dtof import calculate_transients, emulate_dtof, make_irf

# ───────────────────────────── Transients ─────────────────────────────


def test_transient_impulse_bin_and_amplitude():
    depth = np.array([[2.5]])
    albedo = np.array([[0.5]])
    tr = calculate_transients(depth, albedo, n_tbins=100, max_depth=10.0, signal_photons=2.0, ambient_photons=0.0)
    assert tr.shape == (1, 100)
    assert tr[0, 25].item() == pytest.approx(2.0 * 0.5)
    assert tr[0].sum() == pytest.approx(1.0)


def test_transient_ambient_scene_calibration():
    depth = np.full((1, 2), 5.0)
    albedo = np.zeros((1, 2))
    ambient = np.array([[1.0, 3.0]])  # mean-normalized weights: 0.5 and 1.5
    tr = calculate_transients(
        depth, albedo, ambient, n_tbins=10, max_depth=10.0, signal_photons=1.0, ambient_photons=4.0
    )
    assert tr.shape == (2, 10)
    # Albedo zero: ambient only, flat across the period, scene-average ambient_photons.
    assert (tr[0] == tr[0, 0]).all() and (tr[1] == tr[1, 0]).all()
    assert tr[0].sum().item() == pytest.approx(4.0 * 0.5)
    assert tr[1].sum().item() == pytest.approx(4.0 * 1.5)


def test_transient_scalar_ambient_broadcasts_over_scene():
    depth = np.full((2, 2), 5.0)
    albedo = np.full((2, 2), 0.5)
    uniform = calculate_transients(
        depth, albedo, None, n_tbins=8, max_depth=10.0, signal_photons=1.0, ambient_photons=2.0
    )
    for scalar in (0.7, 3.0):
        tr = calculate_transients(
            depth, albedo, scalar, n_tbins=8, max_depth=10.0, signal_photons=1.0, ambient_photons=2.0
        )
        assert torch.equal(tr, uniform)
    with pytest.raises(ValueError, match="positive mean"):
        calculate_transients(depth, albedo, 0.0, n_tbins=8, max_depth=10.0)


def test_transient_depth_wraps_over_max_depth():
    depth = np.array([[12.8]])  # aliases to 2.8 m within the 10 m period
    albedo = np.array([[1.0]])
    tr = calculate_transients(depth, albedo, n_tbins=10, max_depth=10.0, signal_photons=1.0, ambient_photons=0.0)
    assert tr[0, 2].item() == pytest.approx(1.0)
    assert tr[0].sum() == pytest.approx(1.0)


def test_transient_nonfinite_depth_is_ambient_only():
    depth = np.array([[np.inf, 0.0, -1.0]])
    albedo = np.ones((1, 3))
    tr = calculate_transients(depth, albedo, n_tbins=8, max_depth=10.0, signal_photons=3.0, ambient_photons=2.0)
    assert tr.shape == (3, 8)
    assert (tr == tr[:, :1]).all()
    assert tr[0].sum().item() == pytest.approx(2.0)


def test_transient_validation():
    with pytest.raises(ValueError, match="at least 1-D"):
        calculate_transients(np.zeros(()), np.zeros(()), n_tbins=4, max_depth=10.0)
    with pytest.raises(ValueError, match="same shape"):
        calculate_transients(np.zeros((2, 2)), np.zeros((3, 3)), n_tbins=4, max_depth=10.0)
    with pytest.raises(ValueError, match="non-negative"):
        calculate_transients(np.zeros((1, 1)), np.zeros((1, 1)), n_tbins=4, max_depth=10.0, signal_photons=-1.0)
    with pytest.raises(ValueError, match="positive mean"):
        calculate_transients(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), n_tbins=4, max_depth=10.0)
    with pytest.raises(ValueError, match="n_tbins"):
        calculate_transients(np.zeros((1, 1)), np.zeros((1, 1)), n_tbins=0, max_depth=10.0)
    with pytest.raises(ValueError, match="max_depth"):
        calculate_transients(np.zeros((1, 1)), np.zeros((1, 1)), n_tbins=4, max_depth=0.0)


# ───────────────────────────── Pulse kernel ─────────────────────────────


def test_make_irf_normalized_and_symmetric():
    irf = make_irf(2.0)
    assert irf.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(irf, irf.flip(-1))
    assert irf.numel() % 2 == 1
    assert irf.argmax() == irf.numel() // 2
    with pytest.raises(ValueError, match="pulse_width_bins"):
        make_irf(0.0)


# ───────────────────────────── Emulation ─────────────────────────────


def test_emulate_dtof_mc_shape_and_integer_counts():
    depth = np.array([[1.0, 2.0], [3.0, 4.0]])
    albedo = np.full((2, 2), 0.5)
    ewh = emulate_dtof(
        depth,
        albedo,
        n_tbins=32,
        max_depth=10.0,
        signal_photons=0.4,
        ambient_photons=0.2,
        n_pulses=500,
        n_bins=8,
        dead_time_bins=4,
        model="mc",
        pulse_width_bins=0.0,
        generator=make_generator(13),
    )
    assert ewh.shape == (2, 2, 8)
    assert np.all(ewh >= 0)
    assert np.all(ewh == np.floor(ewh))


def test_emulate_dtof_forward_sync_matches_analytic_counts():
    depth = np.array([[2.5]])
    albedo = np.array([[1.0]])
    ewh = emulate_dtof(
        depth,
        albedo,
        n_tbins=100,
        max_depth=10.0,
        signal_photons=0.5,
        ambient_photons=0.0,
        n_pulses=1000,
        dead_time_bins=7,  # ignored in synchronous mode
        free_running=False,
        model="forward",
        pulse_width_bins=0.0,
    )
    # Delta pulse: all signal in bin 25 with single-hit detection probability 1 - exp(-0.5).
    assert ewh.shape == (1, 1, 100)
    assert ewh[0, 0, 25] == pytest.approx(1000 * (1 - math.exp(-0.5)), rel=1e-5)
    assert ewh[0, 0, :25].sum() == 0 and ewh[0, 0, 26:].sum() == 0


def test_emulate_dtof_forward_async_zero_dead_time_preserves_totals():
    depth = np.array([[1.0, 2.0]])
    albedo = np.full((1, 2), 0.5)
    ewh = emulate_dtof(
        depth,
        albedo,
        n_tbins=32,
        max_depth=10.0,
        signal_photons=0.6,
        ambient_photons=0.3,
        n_pulses=100,
        n_bins=4,
        dead_time_bins=0,
        model="forward",
        pulse_width_bins=0.0,
    )
    assert ewh.shape == (1, 2, 4)
    # No dead time: every arrival is an expected detection.
    per_pixel_flux = 0.6 * 0.5 + 0.3
    assert ewh.sum() == pytest.approx(100 * 2 * per_pixel_flux, rel=1e-5)


def test_emulate_dtof_mc_matches_forward_at_low_flux():
    depth = np.array([[2.0, 4.0]])
    albedo = np.full((1, 2), 0.25)
    kwargs = {
        "n_tbins": 64,
        "max_depth": 10.0,
        "signal_photons": 0.1,
        "ambient_photons": 0.05,
        "n_pulses": 50000,
        "free_running": False,
        "pulse_width_bins": 0.0,
    }
    mc = emulate_dtof(depth, albedo, model="mc", generator=make_generator(14), **kwargs)
    fwd = emulate_dtof(depth, albedo, model="forward", **kwargs)
    assert mc.sum() == pytest.approx(fwd.sum(), rel=0.05)
    assert np.allclose(mc / mc.sum(), fwd / fwd.sum(), atol=0.02)


def test_emulate_dtof_reproducible_with_generator():
    depth = np.array([[1.0, 2.0]])
    albedo = np.full((1, 2), 0.5)
    kwargs = {"n_tbins": 32, "max_depth": 10.0, "n_pulses": 200, "dead_time_bins": 4}
    # A shared generator object advances between calls; identical seeds reproduce instead.
    first = emulate_dtof(depth, albedo, generator=make_generator(15), **kwargs)
    second = emulate_dtof(depth, albedo, generator=make_generator(15), **kwargs)
    assert np.array_equal(first, second)


def test_emulate_dtof_validation():
    depth = np.zeros((1, 1))
    albedo = np.zeros((1, 1))
    base = {"n_tbins": 64, "max_depth": 10.0}
    with pytest.raises(ValueError, match="model must be one of"):
        emulate_dtof(depth, albedo, model="coincident", **base)
    with pytest.raises(ValueError, match="n_pulses must be > 0"):
        emulate_dtof(depth, albedo, n_pulses=0, **base)
    with pytest.raises(ValueError, match="exact multiple"):
        emulate_dtof(depth, albedo, n_bins=10, **base)
    with pytest.raises(ValueError, match="dead_time_bins must be a non-negative"):
        emulate_dtof(depth, albedo, dead_time_bins=-2, **base)
    with pytest.raises(ValueError, match="dead_time_bins must be a non-negative"):
        emulate_dtof(depth, albedo, dead_time_bins=2.5, **base)
    with pytest.raises(ValueError, match="pulse_width_bins"):
        emulate_dtof(depth, albedo, pulse_width_bins=-1.0, **base)


# ─────────────────────────── ToF geometry resolution ───────────────────────────

C = scipy.constants.c
PERIOD_10M = 2 * 10.0 / C  # spans 10 m of round-trip range


def test_transient_period_derives_max_depth():
    depth = np.array([[2.55]])
    albedo = np.array([[1.0]])
    by_range = calculate_transients(depth, albedo, n_tbins=100, max_depth=10.0, signal_photons=1.0, ambient_photons=0.0)
    by_period = calculate_transients(
        depth, albedo, n_tbins=100, period=PERIOD_10M, signal_photons=1.0, ambient_photons=0.0
    )
    torch.testing.assert_close(by_period, by_range, rtol=1e-4, atol=1e-9)


def test_transient_bin_width_resolves_geometry():
    depth = np.array([[2.55]])
    albedo = np.array([[1.0]])
    reference = calculate_transients(depth, albedo, n_tbins=10, max_depth=10.0, signal_photons=1.0, ambient_photons=0.0)
    from_width = calculate_transients(
        depth, albedo, bin_width=1.0, max_depth=10.0, signal_photons=1.0, ambient_photons=0.0
    )
    from_bins = calculate_transients(depth, albedo, bin_width=1.0, n_tbins=10, signal_photons=1.0, ambient_photons=0.0)
    torch.testing.assert_close(from_width, reference, rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(from_bins, reference, rtol=1e-6, atol=1e-9)


def test_transient_overconstrained_consistent_warns(caplog):
    depth = np.array([[2.55]])
    albedo = np.array([[1.0]])
    with caplog.at_level(logging.WARNING):
        tr = calculate_transients(
            depth, albedo, n_tbins=10, max_depth=10.0, period=PERIOD_10M, signal_photons=1.0, ambient_photons=0.0
        )
    reference = calculate_transients(depth, albedo, n_tbins=10, max_depth=10.0, signal_photons=1.0, ambient_photons=0.0)
    assert "both max_depth and period" in caplog.text
    torch.testing.assert_close(tr, reference, rtol=1e-6, atol=1e-9)


def test_transient_overconstrained_inconsistent_raises():
    depth = np.array([[2.55]])
    albedo = np.array([[1.0]])
    with pytest.raises(ValueError, match="give only one of them"):
        calculate_transients(depth, albedo, n_tbins=10, max_depth=10.0, period=2 * 15.0 / C)
    with pytest.raises(ValueError, match="give only one of them"):
        calculate_transients(depth, albedo, n_tbins=10, bin_width=1.0, max_depth=11.0)


def test_transient_underconstrained_raises():
    depth = np.array([[2.55]])
    albedo = np.array([[1.0]])
    with pytest.raises(ValueError, match="max_depth or period"):
        calculate_transients(depth, albedo, n_tbins=10)
    with pytest.raises(ValueError, match="n_tbins or bin_width"):
        calculate_transients(depth, albedo, max_depth=10.0)
    with pytest.raises(ValueError, match="positive integer"):
        calculate_transients(depth, albedo, n_tbins=0, max_depth=10.0)


def test_transient_bin_width_must_divide_span():
    depth = np.array([[2.55]])
    albedo = np.array([[1.0]])
    with pytest.raises(ValueError, match="integer bin count"):
        calculate_transients(depth, albedo, bin_width=3.0, max_depth=10.0)


def test_transient_min_depth_wraps_forward():
    depth = np.array([[3.0, 1.0]])
    albedo = np.ones((1, 2))
    tr = calculate_transients(
        depth, albedo, n_tbins=8, min_depth=2.0, max_depth=10.0, signal_photons=1.0, ambient_photons=0.0
    )
    # The 8 m window starts at 2 m: the 3 m return lands in bin 1, the 1 m return arrives
    # before the window and wraps to its end.
    assert tr[0, 1].item() == pytest.approx(1.0)
    assert tr[1, 7].item() == pytest.approx(1.0)


def test_emulate_dtof_period_derives_max_depth():
    depth = np.array([[2.55]])
    albedo = np.array([[1.0]])
    kwargs = {"n_pulses": 100, "model": "forward", "dead_time_bins": 0, "pulse_width_bins": 0.0}
    by_range = emulate_dtof(depth, albedo, n_tbins=100, max_depth=10.0, **kwargs)
    by_period = emulate_dtof(depth, albedo, n_tbins=100, period=PERIOD_10M, **kwargs)
    assert np.allclose(by_period, by_range, rtol=1e-4, atol=1e-9)


def test_transient_accepts_1d_and_2d_inputs():
    depths = np.array([2.0, 4.0])
    albedos = np.full(2, 0.5)
    kwargs = {"n_tbins": 32, "max_depth": 10.0, "signal_photons": 1.0, "ambient_photons": 0.0}
    tr_1d = calculate_transients(depths, albedos, **kwargs)
    tr_2d = calculate_transients(depths.reshape(1, 2), albedos.reshape(1, 2), **kwargs)
    # Flat pixel rows: input shape only changes the emulator output, not the transients.
    torch.testing.assert_close(tr_1d, tr_2d)


def test_emulate_dtof_output_keeps_spatial_shape():
    depths = np.array([[1.0, 2.0], [3.0, 4.0]])
    albedos = np.full((2, 2), 0.5)
    kwargs = {
        "n_tbins": 32,
        "max_depth": 10.0,
        "signal_photons": 0.4,
        "ambient_photons": 0.2,
        "n_pulses": 50,
        "model": "forward",
        "pulse_width_bins": 0.0,
    }
    ewh = emulate_dtof(depths, albedos, **kwargs)
    ewh_1d = emulate_dtof(depths.reshape(-1), albedos.reshape(-1), **kwargs)
    assert ewh.shape == (2, 2, 32)
    assert ewh_1d.shape == (4, 32)
    assert np.allclose(ewh.reshape(-1, 32), ewh_1d)
