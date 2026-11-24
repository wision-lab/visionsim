"""Tests for dToF photon sampling, dead-time and histogramming kernels.

All tests here belong to stage 2 (Monte-Carlo) or stage 3 (forward models) and are commented out
until those stages are reviewed.
"""

# from __future__ import annotations
#
# import math
#
# import numpy as np
# import pytest
# import torch
#
# from tests.emulate.dtof import N_TBINS, make_generator
# from visionsim.emulate.dtof.histogrammers import (
#     _sample_photon_arrivals,
#     batch_distorted_transient_async,
#     batch_distorted_transient_sync,
#     simulate_ewh,
#     simulate_photon_timestamps,
#     timestamps_to_histogram,
# )

# ──────── Stage 2: Monte-Carlo and histogram utilities ────────

# def test_arrival_counts_shape_and_validation():
#     phi = torch.zeros(N_TBINS)
#     phi[10] = 2.0
#     counts = _sample_photon_arrivals(phi, 50, generator=make_generator())
#     assert counts.shape == (50, N_TBINS)
#     assert counts.dtype == torch.int64
#     # A zero-rate bin never receives photons.
#     assert counts[:, :10].sum() == 0 and counts[:, 11:].sum() == 0
#     with pytest.raises(ValueError, match="1-D"):
#         _sample_photon_arrivals(phi[None], 5)
#     with pytest.raises(ValueError, match="n_pulses"):
#         _sample_photon_arrivals(phi, 0)
#
#
# def test_timestamps_validation_and_zero_rate():
#     phi = torch.zeros(4)
#     with pytest.raises(ValueError, match="1-D"):
#         simulate_photon_timestamps(phi[None], 10)
#     with pytest.raises(ValueError, match="n_pulses"):
#         simulate_photon_timestamps(phi, 0)
#     with pytest.raises(ValueError, match="dead_time_bins"):
#         simulate_photon_timestamps(phi, 10, dead_time_bins=-1)
#     with pytest.raises(ValueError, match="dead_time_bins"):
#         simulate_photon_timestamps(phi, 10, dead_time_bins=1.5)
#     assert simulate_photon_timestamps(phi, 10, generator=make_generator()).shape == (0, 2)
#
#
# def test_sync_timestamps_single_hit_per_cycle():
#     phi = torch.full((N_TBINS,), 0.3)
#     ts = simulate_photon_timestamps(phi, 400, free_running=False, generator=make_generator(1))
#     cycles = ts[:, 0]
#     # Synchronous operation records at most one photon per cycle.
#     assert cycles.unique().numel() == cycles.numel()
#     assert cycles.numel() > 300
#
#
# def test_sync_ignores_dead_time():
#     phi = torch.full((N_TBINS,), 0.2)
#     armed = simulate_photon_timestamps(phi, 200, dead_time_bins=0, free_running=False, generator=make_generator(2))
#     blocked = simulate_photon_timestamps(phi, 200, dead_time_bins=32, free_running=False, generator=make_generator(2))
#     assert torch.equal(armed, blocked)
#
#
# def test_sync_first_hit_is_impulse_bin():
#     phi = torch.zeros(N_TBINS)
#     phi[40] = 0.5
#     ts = simulate_photon_timestamps(phi, 500, free_running=False, generator=make_generator(3))
#     assert ts.numel() > 0
#     assert (ts[:, 1] == 40).all()
#
#
# def test_sparse_sampling_all_hits_in_lit_bin():
#     phi = torch.zeros(N_TBINS)
#     phi[7] = 0.2  # dim flux takes the sparse sampling path
#     ts = simulate_photon_timestamps(phi, 200, dead_time_bins=0, generator=make_generator(4))
#     assert ts.shape[0] > 0 and (ts[:, 1] == 7).all()
#
#
# def test_free_running_respects_dead_time():
#     phi = torch.full((N_TBINS,), 0.4)
#     ts = simulate_photon_timestamps(phi, 500, dead_time_bins=8, generator=make_generator(5))
#     assert ts.shape[0] > 0
#     g = ts[:, 0] * N_TBINS + ts[:, 1]
#     assert (g.diff() >= 8).all()
#
#
# def test_dead_time_reduces_free_running_counts():
#     phi = torch.full((N_TBINS,), 0.5)
#     free = simulate_photon_timestamps(phi, 400, dead_time_bins=0, generator=make_generator(6))
#     dead = simulate_photon_timestamps(phi, 400, dead_time_bins=16, generator=make_generator(6))
#     assert dead.shape[0] < free.shape[0]
#
#
# def test_free_running_no_dead_time_detects_every_arrival():
#     phi = torch.full((N_TBINS,), 0.5)
#     ts = simulate_photon_timestamps(phi, 300, dead_time_bins=0, generator=make_generator(7))
#     arrivals = _sample_photon_arrivals(phi, 300, generator=make_generator(7)).sum()
#     assert ts.shape[0] == arrivals
#
#
# def test_histogram_counts_and_rebin():
#     ts = torch.tensor([[0, 0], [0, 10], [1, 20], [5, 63]])
#     hist = timestamps_to_histogram(ts, N_TBINS)
#     assert hist.shape == (N_TBINS,)
#     assert hist.sum() == 4
#     assert hist[0] == 1 and hist[10] == 1 and hist[20] == 1 and hist[63] == 1
#
#     rebinned = timestamps_to_histogram(ts, N_TBINS, 8)
#     assert rebinned.shape == (8,)
#     assert rebinned[0] == 1 and rebinned[1] == 1 and rebinned[2] == 1 and rebinned[7] == 1
#
#     empty = timestamps_to_histogram(torch.zeros((0, 2), dtype=torch.int64), N_TBINS, 8)
#     assert empty.shape == (8,) and empty.sum() == 0
#
#     with pytest.raises(ValueError, match="must be \\(n, 2\\)"):
#         timestamps_to_histogram(ts[:, 0], N_TBINS)
#     with pytest.raises(ValueError, match="exceed"):
#         timestamps_to_histogram(ts, N_TBINS, 128)
#     with pytest.raises(ValueError, match="exact multiple"):
#         timestamps_to_histogram(ts, N_TBINS, 10)
#
#
# def test_ewh_rebin_preserves_counts():
#     phi = torch.full((N_TBINS,), 0.3)
#     full = simulate_ewh(phi, n_pulses=200, n_bins=N_TBINS, dead_time_bins=4, generator=make_generator(8))
#     rebinned = simulate_ewh(phi, n_pulses=200, n_bins=N_TBINS // 4, dead_time_bins=4, generator=make_generator(8))
#     assert full.shape == (N_TBINS,) and rebinned.shape == (N_TBINS // 4,)
#     assert full.sum() == rebinned.sum()


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
