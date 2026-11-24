"""Tests for dToF photon sampling, dead-time and histogramming kernels.

Stage 2 (Monte-Carlo and histogram utilities) is active; stage 3 (forward models and MC agreement)
is commented out until that stage is reviewed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.stats
import torch

from tests.emulate.dtof import N_TBINS, make_generator
from visionsim.emulate.dtof.histogrammers import (
    _sample_occupied_cells,
    simulate_ewh,
    simulate_photon_timestamps,
    timestamps_to_histogram,
)

# Used only by the commented-out stage 3 tests below:
# from visionsim.emulate.dtof.histogrammers import batch_distorted_transient_async, batch_distorted_transient_sync

# Statistical tests use fixed seeds (deterministic); thresholds come from the statistics, not hand tuning.
Z_MAX = 4.0  # bound on |z| for count tests
P_MIN = 1e-3  # floor on chi-square p-values
N_PULSES = 10_000


# ──────── Stage 2: utilities and Monte-Carlo behavior ────────


def test_occupied_cells_shape_and_range():
    phi = torch.zeros(N_TBINS)
    phi[10] = 2.0
    cells, counts = _sample_occupied_cells(phi, 50, generator=make_generator())
    assert cells.dtype == counts.dtype == torch.int64
    assert (counts > 0).all() and (cells.diff() > 0).all()  # sorted, unique, non-empty cells
    assert cells.max() < 50 * N_TBINS
    # A zero-rate bin never receives photons.
    assert (cells % N_TBINS == 10).all()
    assert all(t.numel() == 0 for t in _sample_occupied_cells(torch.zeros(N_TBINS), 50))


def test_timestamps_validation_and_zero_rate():
    phi = torch.zeros(4)
    with pytest.raises(ValueError, match="1-D"):
        simulate_photon_timestamps(phi[None], 10)
    with pytest.raises(ValueError, match="n_pulses"):
        simulate_photon_timestamps(phi, 0)
    with pytest.raises(ValueError, match="dead_time_bins"):
        simulate_photon_timestamps(phi, 10, dead_time_bins=-1)
    with pytest.raises(ValueError, match="dead_time_bins"):
        simulate_photon_timestamps(phi, 10, dead_time_bins=1.5)
    assert simulate_photon_timestamps(phi, 10, generator=make_generator()).shape == (0, 2)


def test_sync_timestamps_single_hit_per_cycle():
    phi = torch.full((N_TBINS,), 0.3)
    ts = simulate_photon_timestamps(phi, 400, free_running=False, generator=make_generator(1))
    cycles = ts[:, 0]
    # Synchronous operation records at most one photon per cycle.
    assert cycles.unique().numel() == cycles.numel()
    assert cycles.numel() > 300


def test_sync_ignores_dead_time():
    phi = torch.full((N_TBINS,), 0.2)
    armed = simulate_photon_timestamps(phi, 200, dead_time_bins=0, free_running=False, generator=make_generator(2))
    blocked = simulate_photon_timestamps(phi, 200, dead_time_bins=32, free_running=False, generator=make_generator(2))
    assert torch.equal(armed, blocked)


def test_free_running_no_dead_time_detects_every_arrival():
    phi = torch.full((N_TBINS,), 0.5)
    ts = simulate_photon_timestamps(phi, 300, dead_time_bins=0, generator=make_generator(7))
    _, counts = _sample_occupied_cells(phi, 300, generator=make_generator(7))
    assert ts.shape[0] == counts.sum()


def test_histogram_counts_and_rebin():
    ts = torch.tensor([[0, 0], [0, 10], [1, 20], [5, 63]])
    hist = timestamps_to_histogram(ts, N_TBINS)
    assert hist.shape == (N_TBINS,)
    assert hist.sum() == 4
    assert hist[0] == 1 and hist[10] == 1 and hist[20] == 1 and hist[63] == 1

    rebinned = timestamps_to_histogram(ts, N_TBINS, 8)
    assert rebinned.shape == (8,)
    assert rebinned[0] == 1 and rebinned[1] == 1 and rebinned[2] == 1 and rebinned[7] == 1

    empty = timestamps_to_histogram(torch.zeros((0, 2), dtype=torch.int64), N_TBINS, 8)
    assert empty.shape == (8,) and empty.sum() == 0

    with pytest.raises(ValueError, match="must be \\(n, 2\\)"):
        timestamps_to_histogram(ts[:, 0], N_TBINS)
    with pytest.raises(ValueError, match="exceed"):
        timestamps_to_histogram(ts, N_TBINS, 128)
    with pytest.raises(ValueError, match="exact multiple"):
        timestamps_to_histogram(ts, N_TBINS, 10)


def test_ewh_rebin_preserves_counts():
    phi = torch.full((N_TBINS,), 0.3)
    full = simulate_ewh(phi, n_pulses=200, n_bins=N_TBINS, dead_time_bins=4, generator=make_generator(8))
    rebinned = simulate_ewh(phi, n_pulses=200, n_bins=N_TBINS // 4, dead_time_bins=4, generator=make_generator(8))
    assert full.shape == (N_TBINS,) and rebinned.shape == (N_TBINS // 4,)
    assert full.sum() == rebinned.sum()


# ──────── Stage 2: Monte-Carlo against independent ground truth ────────


def _coates(phi: np.ndarray) -> np.ndarray:
    """First-photon-wins detection probability per bin of one synchronous cycle."""
    p = 1 - np.exp(-phi)
    survival = np.concatenate([[1.0], np.cumprod(1 - p)[:-1]])
    return p * survival


def _chi_square_p(observed: np.ndarray, probs: np.ndarray, n: int) -> float:
    """p-value of per-bin counts (plus a no-detection category) against ``probs``; sparse bins are pooled."""
    observed = np.append(observed, n - observed.sum())
    expected = n * np.append(probs, 1 - probs.sum())
    keep = expected >= 5
    o = np.append(observed[keep], observed[~keep].sum())
    e = np.append(expected[keep], expected[~keep].sum())
    nonzero = e > 0
    o, e = o[nonzero], e[nonzero]
    return float(scipy.stats.chi2.sf(((o - e) ** 2 / e).sum(), len(o) - 1))


@pytest.mark.parametrize("shape", ["flat-low", "flat-high", "spike"])
def test_mc_sync_matches_coates(shape):
    phi = {"flat-low": torch.full((N_TBINS,), 0.003), "flat-high": torch.full((N_TBINS,), 0.3)}.get(
        shape, torch.full((N_TBINS,), 0.02)
    )
    if shape == "spike":
        phi[20] = 1.5
    ts = simulate_photon_timestamps(phi, N_PULSES, free_running=False, generator=make_generator(0))
    counts = np.bincount(ts[:, 1].numpy(), minlength=N_TBINS)
    assert _chi_square_p(counts, _coates(phi.numpy()), N_PULSES) > P_MIN


@pytest.mark.parametrize("phi", [0.002, 0.5], ids=["low-flux", "bright"])
@pytest.mark.parametrize("dead_time", [1, 5, 16, 100])  # 100 exceeds n_tbins
def test_mc_free_running_matches_renewal_rate(phi, dead_time):
    # With flat flux, a detection is followed by the first occupied bin at or after dead_time bins later:
    # the gap is dead_time + Geometric(p), a renewal process with a known mean and variance.
    p = 1 - math.exp(-phi)
    mean_gap, var_gap = dead_time + (1 - p) / p, (1 - p) / p**2
    n_bins = N_PULSES * N_TBINS
    expected, sd = n_bins / mean_gap, math.sqrt(n_bins * var_gap / mean_gap**3)
    ts = simulate_photon_timestamps(
        torch.full((N_TBINS,), phi), N_PULSES, dead_time_bins=dead_time, generator=make_generator(1)
    )
    assert abs(ts.shape[0] - expected) / sd < Z_MAX


@pytest.mark.parametrize("phi", [0.002, 0.5], ids=["low-flux", "bright"])
def test_mc_zero_dead_time_is_poisson(phi):
    ts = simulate_photon_timestamps(torch.full((N_TBINS,), phi), N_PULSES, dead_time_bins=0, generator=make_generator(2))
    per_cycle = torch.bincount(ts[:, 0], minlength=N_PULSES).double()
    lam = N_TBINS * phi
    assert abs(per_cycle.mean().item() - lam) / math.sqrt(lam / N_PULSES) < Z_MAX
    # Poisson counts have variance equal to their mean (Fano factor 1).
    fano_sd = math.sqrt((1 / lam + 2) / N_PULSES)
    assert abs(per_cycle.var().item() / per_cycle.mean().item() - 1) / fano_sd < Z_MAX


@pytest.mark.parametrize("dead_time", [1, 10, 64, 100])  # 64 equals and 100 exceeds n_tbins
def test_dead_time_convention_exact_spacing(dead_time):
    # Saturating flux occupies every bin, so the detector re-arms exactly dead_time bins after each
    # detection, across cycle boundaries too: detections are at least dead_time bins apart.
    phi = torch.full((N_TBINS,), 50.0)
    ts = simulate_photon_timestamps(phi, 300, dead_time_bins=dead_time, generator=make_generator(3))
    g = ts[:, 0] * N_TBINS + ts[:, 1]
    assert g[0] == 0 and (g.diff() == dead_time).all()
    assert torch.equal(ts, simulate_photon_timestamps(phi, 300, dead_time_bins=dead_time, generator=make_generator(3)))
    # Realistic flux never violates the minimum spacing.
    dim = simulate_photon_timestamps(
        torch.full((N_TBINS,), 0.4), 300, dead_time_bins=dead_time, generator=make_generator(4)
    )
    assert (dim[:, 0] * N_TBINS + dim[:, 1]).diff().min() >= dead_time


# ──────── Stage 3: forward models and MC agreement ────────

# def test_sync_forward_impulse_detection_probability():
#     phi = torch.zeros(1, N_TBINS)
#     phi[0, 5] = 0.7
#     out = batch_distorted_transient_sync(phi)
#     assert out[0, 5].item() == pytest.approx(1 - math.exp(-0.7), abs=1e-6)
#     assert out[0, :5].sum() == 0 and out[0, 6:].sum() == 0
#
#
# def test_sync_forward_flat_geometric_detection():
#     c = 0.05
#     phi = torch.full((1, N_TBINS), c)
#     out = batch_distorted_transient_sync(phi)
#     expected = (1 - math.exp(-c)) * np.exp(-c * np.arange(N_TBINS))
#     torch.testing.assert_close(out[0].double(), torch.from_numpy(expected), rtol=1e-4, atol=1e-8)
#
#
# def test_sync_forward_sums_to_single_hit_probability():
#     phi = torch.rand(4, N_TBINS, generator=make_generator(9)) * 0.3
#     sums = batch_distorted_transient_sync(phi).sum(dim=-1)
#     expected = 1 - torch.exp(-phi.sum(dim=-1))
#     torch.testing.assert_close(sums, expected, rtol=1e-4, atol=1e-6)
#
#
# def test_async_forward_zero_dead_time_is_undistorted():
#     phi = torch.rand(3, N_TBINS, generator=make_generator(10))
#     assert torch.equal(batch_distorted_transient_async(phi, 0), phi)
#
#
# def test_async_forward_total_rate_matches_nonparalyzable_formula():
#     phi = torch.full((1, N_TBINS), 0.2)  # 12.8 photons per cycle
#     dt = 16
#     out = batch_distorted_transient_async(phi, dt)
#     expected_rate = 12.8 / (1 + 12.8 * dt / N_TBINS)
#     assert out.sum().item() == pytest.approx(expected_rate, rel=1e-3)
#
#
# def test_async_forward_dead_time_suppresses_rate():
#     phi = torch.full((1, N_TBINS), 0.5)
#     clean = batch_distorted_transient_async(phi, 0).sum()
#     dead = batch_distorted_transient_async(phi, N_TBINS // 2).sum()
#     assert dead < 0.5 * clean
#
#
# def test_mc_sync_matches_coates_forward():
#     phi = torch.full((N_TBINS,), 0.02)
#     phi[30] += 0.4
#     n_pulses = 4000
#     ewh = simulate_ewh(phi, n_pulses=n_pulses, free_running=False, generator=make_generator(11))
#     forward = n_pulses * batch_distorted_transient_sync(phi[None])[0]
#     assert ewh.sum() == pytest.approx(forward.sum().item(), rel=0.1)
#     torch.testing.assert_close(ewh / ewh.sum(), forward / forward.sum(), rtol=0.0, atol=0.01)
#
#
# def test_mc_free_running_matches_async_forward():
#     phi = torch.full((N_TBINS,), 0.05)
#     phi[20] += 0.5
#     dt = 12
#     n_pulses = 3000
#     ewh = simulate_ewh(phi, n_pulses=n_pulses, dead_time_bins=dt, generator=make_generator(12))
#     forward = n_pulses * batch_distorted_transient_async(phi[None], dt)[0]
#     assert ewh.sum() == pytest.approx(forward.sum().item(), rel=0.15)
#     torch.testing.assert_close(ewh / ewh.sum(), forward / forward.sum(), rtol=0.0, atol=0.02)
