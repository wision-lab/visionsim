"""Tests for scene-level dToF transient generation and emulation.

Tests cover single-pixel photon arrival rates (transients) and the scene-level readout.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
import scipy.constants
import torch

from tests.emulate.dtof import make_generator
from visionsim.emulate.dtof import (
    apply_irf,
    calculate_transients,
    emulate_dtof,
    forward_free_running,
    forward_sync,
    make_irf,
    simulate_ewh,
)

# ──────── Stage 1: arrival rates ────────

# ──────── Transients ────────


def test_transient_impulse_bin_and_amplitude():
    depth = np.array([[2.5]])
    albedo = np.array([[0.5]])
    tr = calculate_transients(
        depth,
        albedo,
        n_tbins=100,
        max_depth=10.0,
        signal_photons=2.0,
        ambient_photons=0.0,
        reference_depth=None,
    )
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
    with pytest.raises(ValueError, match="reference_depth"):
        calculate_transients(np.zeros((1, 1)), np.zeros((1, 1)), n_tbins=4, max_depth=10.0, reference_depth=0.0)
    with pytest.raises(ValueError, match="positive mean"):
        calculate_transients(depth, albedo, 0.0, n_tbins=8, max_depth=10.0)


def test_transient_depth_wraps_over_max_depth():
    depth = np.array([[12.8]])  # aliases to 2.8 m within the 10 m period
    albedo = np.array([[1.0]])
    tr = calculate_transients(
        depth,
        albedo,
        n_tbins=10,
        max_depth=10.0,
        signal_photons=1.0,
        ambient_photons=0.0,
        reference_depth=None,
    )
    assert tr[0, 2].item() == pytest.approx(1.0)
    assert tr[0].sum() == pytest.approx(1.0)


def test_transient_nonfinite_depth_is_ambient_only():
    depth = np.array([[np.inf, 0.0, -1.0]])
    albedo = np.ones((1, 3))
    tr = calculate_transients(depth, albedo, n_tbins=8, max_depth=10.0, signal_photons=3.0, ambient_photons=2.0)
    assert tr.shape == (3, 8)
    assert (tr == tr[:, :1]).all()
    assert tr[0].sum().item() == pytest.approx(2.0)


def test_transient_signal_falls_off_with_inverse_square_depth():
    # 12.8 m aliases into bin 2 of the 10 m window but is attenuated by its true distance.
    depth = np.array([[2.0, 4.0, 12.8]])
    tr = calculate_transients(
        depth, np.ones((1, 3)), n_tbins=10, max_depth=10.0, signal_photons=1.0, ambient_photons=0.0, reference_depth=2.0
    )
    assert tr.max(dim=1).values.tolist() == pytest.approx([1.0, 0.25, (2.0 / 12.8) ** 2])
    # Ambient does not fall off, and None disables the signal falloff.
    ambient = calculate_transients(
        depth, np.zeros((1, 3)), n_tbins=10, max_depth=10.0, signal_photons=1.0, ambient_photons=2.0
    )
    assert torch.allclose(ambient, torch.full_like(ambient, 0.2))
    flat = calculate_transients(
        depth, np.ones((1, 3)), n_tbins=10, max_depth=10.0, ambient_photons=0.0, reference_depth=None
    )
    assert flat.max(dim=1).values.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_transient_tiny_depth_is_floored_to_one_bin():
    tr = calculate_transients(
        np.array([[1e-6]]), np.ones((1, 1)), n_tbins=100, max_depth=10.0, ambient_photons=0.0, reference_depth=1.0
    )
    assert tr.max().item() == pytest.approx((1 / 0.1) ** 2)  # bin width is 0.1 m


def test_transient_min_depth_wraps_forward():
    depth = np.array([[3.0, 1.0]])
    albedo = np.ones((1, 2))
    tr = calculate_transients(
        depth,
        albedo,
        n_tbins=8,
        min_depth=2.0,
        max_depth=10.0,
        signal_photons=1.0,
        ambient_photons=0.0,
        reference_depth=None,
    )
    # The 8 m window starts at 2 m: the 3 m return lands in bin 1, the 1 m return arrives
    # before the window and wraps to its end.
    assert tr[0, 1].item() == pytest.approx(1.0)
    assert tr[1, 7].item() == pytest.approx(1.0)


def test_transient_accepts_1d_and_2d_inputs():
    depths = np.array([2.0, 4.0])
    albedos = np.full(2, 0.5)
    kwargs = {"n_tbins": 32, "max_depth": 10.0, "signal_photons": 1.0, "ambient_photons": 0.0}
    tr_1d = calculate_transients(depths, albedos, **kwargs)
    tr_2d = calculate_transients(depths.reshape(1, 2), albedos.reshape(1, 2), **kwargs)
    # Flat pixel rows: input shape only changes the emulator output, not the transients.
    torch.testing.assert_close(tr_1d, tr_2d)


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


# ──────── Pulse kernel ────────


@pytest.mark.parametrize("depth", [0.05, 5.0, 9.95])  # the first and last bins exercise the window edges
def test_transient_pulse_width_in_seconds_conserves_flux(depth):
    bin_time = 2 * 10.0 / C / 100  # 100 bins spanning 10 m of range
    kwargs = {"n_tbins": 100, "max_depth": 10.0, "signal_photons": 3.0, "ambient_photons": 2.0, "reference_depth": None}
    tr = calculate_transients(np.array([depth]), np.array([0.8]), pulse_width=4 * bin_time, **kwargs)
    assert tr.shape == (1, 100)
    assert tr.sum().item() == pytest.approx(3.0 * 0.8 + 2.0, rel=1e-5)
    assert tr.argmax().item() == int(depth * 10)
    # A pulse of 4 bin durations is the same as a 4-bin kernel.
    unshaped = calculate_transients(np.array([depth]), np.array([0.8]), **kwargs)
    assert torch.allclose(tr, apply_irf(unshaped, 4.0), atol=1e-6)


@pytest.mark.parametrize("width", [2.0, 4.0, 10.0])
def test_make_irf_matches_requested_width(width):
    irf = make_irf(width).double()
    x = torch.arange(irf.numel(), dtype=torch.float64) - irf.numel() // 2
    sigma = width / (2 * math.sqrt(2 * math.log(2)))
    assert ((x**2) * irf).sum().item() == pytest.approx(sigma**2, rel=0.05)


def test_transient_pulse_narrower_than_one_bin_raises():
    bin_time = 2 * 10.0 / C / 100
    kwargs = {"n_tbins": 100, "max_depth": 10.0}
    depth, albedo = np.array([2.5]), np.array([1.0])
    calculate_transients(depth, albedo, pulse_width=bin_time, **kwargs)  # exactly one bin is allowed
    with pytest.raises(ValueError, match="at least one time bin"):
        calculate_transients(depth, albedo, pulse_width=0.5 * bin_time, **kwargs)
    with pytest.raises(ValueError, match="non-negative"):
        calculate_transients(depth, albedo, pulse_width=-bin_time, **kwargs)


def test_apply_irf_is_circular_with_requested_width():
    # A spike in the last bin leaks its tail into the first bins instead of losing it.
    x = torch.zeros(1, 400)
    x[0, 399] = 1.0
    assert apply_irf(x, 0.0) is x
    y = apply_irf(x, 12.0)[0]
    assert y.sum().item() == pytest.approx(1.0, rel=1e-5)
    assert y[:5].sum().item() > 0.1
    # A kernel longer than the window still conserves flux.
    z = torch.rand(2, 5)
    assert torch.allclose(apply_irf(z, 12.0).sum(-1), z.sum(-1), rtol=1e-5)
    # Measured FWHM matches the request.
    mid = torch.zeros(1, 400)
    mid[0, 200] = 1.0
    assert (apply_irf(mid, 10.0)[0] > apply_irf(mid, 10.0).max() / 2).sum().item() == pytest.approx(10, abs=1.5)


def test_make_irf_normalized_and_symmetric():
    irf = make_irf(2.0)
    assert irf.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(irf, irf.flip(-1))
    assert irf.numel() % 2 == 1
    assert irf.argmax() == irf.numel() // 2
    with pytest.raises(ValueError, match="pulse_width_bins"):
        make_irf(0.0)


# ──────── ToF geometry resolution ────────


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


# ──────── Scene-level readout: emulate_dtof ────────
# A scene is a loop over independent pixels, so these use tiny scenes and check the wiring:
# transients, per-pixel readout, rebinning, reshaping, seeding. The per-pixel physics is verified
# in test_histogrammers.py.

MODELS = ["mc", "forward"]
SCENE = {"n_tbins": 32, "max_depth": 10.0, "signal_photons": 0.4, "ambient_photons": 0.2, "pulse_width": 0.0}


def _single_pixel_reference(depth, albedo, model, *, n_pulses, n_bins, dead_time_bins, free_running, generator=None):
    """The scene-level result of a 1-pixel scene, built directly from the single-pixel kernels."""
    transient = calculate_transients(
        depth, albedo, None, **{k: v for k, v in SCENE.items() if k != "n_tbins"}, n_tbins=32
    )[0]
    if model == "mc":
        return simulate_ewh(
            transient, n_pulses=n_pulses, n_bins=n_bins, dead_time_bins=dead_time_bins,
            free_running=free_running, generator=generator,
        ).numpy()  # fmt: skip
    forward = forward_free_running(transient, dead_time_bins) if free_running else forward_sync(transient)
    return (n_pulses * forward).reshape(n_bins, -1).sum(dim=-1).float().numpy()


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("free_running", [True, False], ids=["free-running", "sync"])
def test_emulate_dtof_single_pixel_matches_kernels(model, free_running):
    depth, albedo = np.array([[2.5]]), np.array([[0.7]])
    kwargs = {"n_pulses": 400, "n_bins": 8, "dead_time_bins": 4, "free_running": free_running}
    out = emulate_dtof(depth, albedo, model=model, generator=make_generator(20), **SCENE, **kwargs)
    ref = _single_pixel_reference(depth, albedo, model, generator=make_generator(20), **kwargs)
    assert out.shape == (1, 1, 8)
    np.testing.assert_array_equal(out[0, 0], ref)


def test_emulate_dtof_mc_shape_and_integer_counts():
    depth = np.array([[1.0, 2.0], [3.0, 4.0]])
    ewh = emulate_dtof(
        depth, np.full((2, 2), 0.5), n_pulses=500, n_bins=8, dead_time_bins=4, model="mc", generator=make_generator(13), **SCENE
    )  # fmt: skip
    assert ewh.shape == (2, 2, 8)
    assert np.all(ewh >= 0)
    assert np.all(ewh == np.floor(ewh))


def test_emulate_dtof_reproducible_with_generator():
    depth, albedo = np.array([[1.0, 2.0]]), np.full((1, 2), 0.5)
    kwargs = {"n_pulses": 200, "dead_time_bins": 4, **SCENE}
    # A shared generator object advances between calls; identical seeds reproduce instead.
    first = emulate_dtof(depth, albedo, generator=make_generator(15), **kwargs)
    second = emulate_dtof(depth, albedo, generator=make_generator(15), **kwargs)
    assert np.array_equal(first, second)


def test_emulate_dtof_forward_sync_matches_analytic_counts():
    depth, albedo = np.array([[2.5]]), np.array([[1.0]])
    ewh = emulate_dtof(
        depth, albedo, n_tbins=100, max_depth=10.0, signal_photons=0.5, ambient_photons=0.0, reference_depth=None,
        n_pulses=1000, dead_time_bins=7, free_running=False, model="forward", pulse_width=0.0,
    )  # fmt: skip
    # Delta pulse: all signal in bin 25 with single-hit detection probability 1 - exp(-0.5);
    # dead_time_bins is ignored in synchronous mode.
    assert ewh.shape == (1, 1, 100)
    assert ewh[0, 0, 25] == pytest.approx(1000 * (1 - math.exp(-0.5)), rel=1e-5)
    assert ewh[0, 0, :25].sum() == 0 and ewh[0, 0, 26:].sum() == 0


def test_emulate_dtof_forward_zero_dead_time_preserves_totals():
    depth, albedo = np.array([[1.0, 2.0]]), np.full((1, 2), 0.5)
    ewh = emulate_dtof(
        depth, albedo, n_tbins=32, max_depth=10.0, signal_photons=0.6, ambient_photons=0.3, reference_depth=None,
        n_pulses=100, n_bins=4, dead_time_bins=0, model="forward", pulse_width=0.0,
    )  # fmt: skip
    assert ewh.shape == (1, 2, 4)
    # No dead time: every arrival is an expected detection.
    assert ewh.sum() == pytest.approx(100 * 2 * (0.6 * 0.5 + 0.3), rel=1e-5)


@pytest.mark.parametrize("free_running", [True, False], ids=["free-running", "sync"])
def test_emulate_dtof_mc_matches_forward_at_low_flux(free_running):
    depth, albedo = np.array([[2.0, 4.0]]), np.full((1, 2), 0.25)
    kwargs = {
        "n_tbins": 64, "max_depth": 10.0, "signal_photons": 0.1, "ambient_photons": 0.05,
        "n_pulses": 50000, "dead_time_bins": 3, "free_running": free_running, "pulse_width": 0.0,
    }  # fmt: skip
    mc = emulate_dtof(depth, albedo, model="mc", generator=make_generator(14), **kwargs)
    fwd = emulate_dtof(depth, albedo, model="forward", **kwargs)
    assert mc.sum() == pytest.approx(fwd.sum(), rel=0.05)
    assert np.allclose(mc / mc.sum(), fwd / fwd.sum(), atol=0.02)


def test_emulate_dtof_validation():
    depth, albedo = np.zeros((1, 1)), np.zeros((1, 1))
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
    with pytest.raises(ValueError, match="pulse_width"):
        emulate_dtof(depth, albedo, pulse_width=-1.0, **base)


@pytest.mark.parametrize("model", MODELS)
def test_emulate_dtof_period_derives_max_depth(model):
    depth, albedo = np.array([[2.55]]), np.array([[1.0]])
    kwargs = {"n_pulses": 100, "model": model, "dead_time_bins": 0, "pulse_width": 0.0}
    by_range = emulate_dtof(depth, albedo, n_tbins=100, max_depth=10.0, generator=make_generator(16), **kwargs)
    by_period = emulate_dtof(depth, albedo, n_tbins=100, period=PERIOD_10M, generator=make_generator(16), **kwargs)
    assert np.allclose(by_period, by_range, rtol=1e-4, atol=1e-9)


@pytest.mark.parametrize("model", MODELS)
def test_emulate_dtof_output_keeps_spatial_shape(model):
    depths = np.array([[1.0, 2.0], [3.0, 4.0]])
    albedos = np.full((2, 2), 0.5)
    kwargs = {"n_pulses": 50, "model": model, **SCENE}
    ewh = emulate_dtof(depths, albedos, generator=make_generator(17), **kwargs)
    ewh_1d = emulate_dtof(depths.reshape(-1), albedos.reshape(-1), generator=make_generator(17), **kwargs)
    assert ewh.shape == (2, 2, 32)
    assert ewh_1d.shape == (4, 32)
    assert np.allclose(ewh.reshape(-1, 32), ewh_1d)
