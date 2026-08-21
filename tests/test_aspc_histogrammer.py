"""Stage 3 — Histogrammer dead-time / pile-up forward models.

3a: distribution invariants (normalization, zero-flux robustness) — deterministic.

Functions under test (histogrammers.py):
  * batch_distorted_transient_sync  — batched single-hit Coates (first-photon-wins)
  * batch_distorted_transient_async — batched free-running (power iteration)
  * calculate_distorted_transient       — per-pixel free-running (eigendecomposition)
  * calculate_distorted_transient_sync  — per-pixel synchronous multi/single-hit
"""

import pytest
import torch

from visionsim.emulate.aspc.histogrammers import (
    HistConfig,
    Histogrammer,
    HistogrammerEDH,
    batch_distorted_transient_async,
    batch_distorted_transient_sync,
    calculate_distorted_transient,
    calculate_distorted_transient_sync,
)

B = 12  # histogram bins


def _phi(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=g) + 0.1  # strictly positive arrival rates


def _com(x):
    """Center of mass (mean bin index) of a 1-D distribution."""
    idx = torch.arange(x.shape[-1], dtype=x.dtype)
    return float((idx * x).sum() / x.sum())


# =========================================================================== #
# 3a-1 — closed-form outputs are proper probability distributions (sum to 1)   #
# =========================================================================== #
class TestNormalization:
    def test_batch_sync_rows_sum_to_one(self):
        out = batch_distorted_transient_sync(_phi((5, B)), 0, B)
        assert torch.allclose(out.sum(-1), torch.ones(5), atol=1e-5)

    def test_batch_async_deadtime_rows_sum_to_one(self):
        out = batch_distorted_transient_async(_phi((5, B)), 3, B)
        assert torch.allclose(out.sum(-1), torch.ones(5), atol=1e-5)

    def test_batch_async_no_deadtime_rows_sum_to_one(self):
        out = batch_distorted_transient_async(_phi((5, B)), 0, B)
        assert torch.allclose(out.sum(-1), torch.ones(5), atol=1e-5)

    def test_per_pixel_async_sums_to_one(self):
        out = calculate_distorted_transient(_phi((B,)), 3, B)
        assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-5)

    def test_per_pixel_sync_sums_to_one(self):
        out = calculate_distorted_transient_sync(_phi((B,)), B, B)
        assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-5)

    def test_distributions_are_non_negative(self):
        assert batch_distorted_transient_sync(_phi((5, B)), 0, B).min() >= -1e-8
        assert batch_distorted_transient_async(_phi((5, B)), 3, B).min() >= -1e-8


# =========================================================================== #
# 3a-2 — zero-flux robustness (all-zero input -> all-zero output, no NaN/Inf)  #
# =========================================================================== #
class TestZeroFlux:
    def test_batch_sync_zero_flux(self):
        out = batch_distorted_transient_sync(torch.zeros(4, B), 0, B)
        assert torch.count_nonzero(out) == 0
        assert torch.isfinite(out).all()

    def test_batch_async_zero_flux(self):
        out = batch_distorted_transient_async(torch.zeros(4, B), 3, B)
        assert torch.count_nonzero(out) == 0
        assert torch.isfinite(out).all()

    def test_batch_sync_mixed_zero_and_nonzero_rows(self):
        phi = _phi((3, B))
        phi[1] = 0.0  # middle row is a dead pixel
        out = batch_distorted_transient_sync(phi, 0, B)
        assert torch.count_nonzero(out[1]) == 0
        assert torch.isclose(out[0].sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(out[2].sum(), torch.tensor(1.0), atol=1e-5)

    def test_batch_async_mixed_zero_and_nonzero_rows(self):
        phi = _phi((3, B))
        phi[1] = 0.0
        out = batch_distorted_transient_async(phi, 3, B)
        assert torch.count_nonzero(out[1]) == 0
        assert torch.isclose(out[0].sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(out[2].sum(), torch.tensor(1.0), atol=1e-5)


# =========================================================================== #
# 3b — pile-up physics                                                         #
# =========================================================================== #
class TestPileUpPhysics:
    def test_no_deadtime_preserves_shape(self):
        # dt=0 => no pile-up distortion => measured distribution ∝ arrival rate.
        phi = _phi((1, B))
        out = batch_distorted_transient_async(phi, 0, B)
        ratio = out / phi
        assert float(ratio.std() / ratio.mean()) < 1e-5

    def test_deadtime_biases_toward_early_bins(self):
        # Flat arrival rate + dead time => earlier bins win the first photon,
        # so the distorted transient is monotonically decreasing.
        flat = torch.full((1, B), 0.5)
        s = batch_distorted_transient_sync(flat, 0, B)[0]
        assert torch.all(s[1:] <= s[:-1] + 1e-7)
        assert s[0] > s[-1]

    def test_higher_flux_shifts_detection_earlier(self):
        lo = batch_distorted_transient_sync(torch.full((1, B), 0.2), 0, B)[0]
        hi = batch_distorted_transient_sync(torch.full((1, B), 2.0), 0, B)[0]
        assert _com(hi) < _com(lo)


# =========================================================================== #
# 3c — cross-checks between implementations                                    #
# =========================================================================== #
class TestImplementationCrossChecks:
    @pytest.mark.xfail(
        reason="batch_distorted_transient_async has NOT been migrated to the corrected "
        "single-pixel kernel: it still re-arms one bin late and uses the inclusive "
        "survival slice. calculate_distorted_transient is now MC-exact, so the two "
        "legitimately disagree. Out of scope until the multi-pixel pass.",
        strict=True,
    )
    def test_batch_async_matches_eigendecomposition(self):
        # The vectorized power-iteration must reproduce the original per-pixel
        # eigendecomposition of the same free-running Markov model.
        phi = _phi((B,), seed=2)
        batched = batch_distorted_transient_async(phi.unsqueeze(0), 3, B, n_iterations=500)[0]
        eigen = calculate_distorted_transient(phi, 3, B)
        assert torch.allclose(batched, eigen, atol=1e-5)

    def test_sync_single_hit_matches_coates(self):
        """Finding H1, now fixed: single-hit calc_sync reduces to Coates."""
        phi = _phi((B,), seed=2)
        coates = batch_distorted_transient_sync(phi.unsqueeze(0), B, B)[0]
        multihit = calculate_distorted_transient_sync(phi, B, B)
        assert torch.allclose(coates, multihit, atol=1e-3)


# =========================================================================== #
# 3d — Monte-Carlo ground truth vs closed-form forward models                  #
# =========================================================================== #
# simulate_pixel_ewh is the sampled reference: it now delegates to the
# timestamp simulator in detector.py (Poisson arrival *counts* per bin + a
# dead-time walk). The closed-form fast paths must match it.
MC_B = 10
MC_PULSES = 20000


def _mc(phi, free_running, dead_time_bins, n_pulses=MC_PULSES, seed=0):
    torch.manual_seed(seed)
    h = Histogrammer(HistConfig())
    hist = h.simulate_pixel_ewh(phi, n_pulses, phi.shape[-1], free_running, dead_time_bins, fast_sim=False)
    return hist / hist.sum()


class TestMonteCarloForwardModel:
    def test_gated_single_hit_matches_coates(self):
        # Headline check: the fast single-hit path == sampled ground truth.
        phi = _phi((MC_B,), seed=3)
        mc = _mc(phi, free_running=False, dead_time_bins=MC_B)
        coates = batch_distorted_transient_sync(phi.unsqueeze(0), MC_B, MC_B)[0]
        assert torch.allclose(mc, coates, atol=0.02)

    def test_gated_single_hit_matches_calc_sync(self):
        """Finding H1, now fixed: calc_sync single-hit matches sampled truth."""
        phi = _phi((MC_B,), seed=3)
        mc = _mc(phi, free_running=False, dead_time_bins=MC_B)
        calc_sync = calculate_distorted_transient_sync(phi, MC_B, MC_B)
        assert torch.allclose(mc, calc_sync, atol=0.02)

    def test_dt0_low_flux_matches_async(self):
        # At low flux, 1 - e^{-phi} ~ phi, so async(dt=0) ~ MC.
        phi = torch.full((MC_B,), 0.02)
        mc = _mc(phi, free_running=True, dead_time_bins=0, n_pulses=40000)
        async_ = batch_distorted_transient_async(phi.unsqueeze(0), 0, MC_B)[0]
        assert torch.allclose(mc, async_, atol=0.02)

    def test_dt0_ground_truth_is_arrival_rate(self):
        # With zero dead time every arriving photon is detected, so the sampled
        # truth follows phi itself -- not the per-bin detection probability
        # 1 - e^{-phi}, which is what a Bernoulli-per-bin sampler would give.
        phi = torch.rand(MC_B, generator=torch.Generator().manual_seed(5)) * 2.0 + 0.5
        mc = _mc(phi, free_running=True, dead_time_bins=0, n_pulses=40000)
        assert torch.allclose(mc, phi / phi.sum(), atol=0.02)

        bernoulli = 1.0 - torch.exp(-phi)
        bernoulli = bernoulli / bernoulli.sum()
        assert not torch.allclose(mc, bernoulli, atol=0.02)

    def test_dt0_high_flux_matches_async(self):
        phi = torch.rand(MC_B, generator=torch.Generator().manual_seed(5)) * 2.0 + 0.5
        mc = _mc(phi, free_running=True, dead_time_bins=0, n_pulses=40000)
        async_ = batch_distorted_transient_async(phi.unsqueeze(0), 0, MC_B)[0]
        assert torch.allclose(mc, async_, atol=0.02)


# =========================================================================== #
# 3e — simulate_ewh / simulate_ewh_diff dispatcher wiring                      #
# =========================================================================== #
class TestSimulateEwhDispatch:
    def test_fast_path_total_counts_equal_n_pulses(self):
        h = Histogrammer(HistConfig())
        rates = _phi((3, B), seed=7)
        n_pulses = 500
        for free_running, dt in [(False, 0), (True, 3)]:
            hists = h.simulate_ewh(rates, n_pulses, B, free_running=free_running, dead_time_bins=dt, fast_sim=True)
            for hist in hists:
                assert float(hist.sum()) == pytest.approx(n_pulses, rel=1e-4)

    def test_diff_returns_expected_value(self):
        h = Histogrammer(HistConfig())
        rates = _phi((3, B), seed=7)
        out = h.simulate_ewh_diff(rates, 500, B, free_running=True, dead_time_bins=0)
        assert torch.allclose(out, rates * 500)

    def test_diff_rejects_nonzero_dead_time(self):
        h = Histogrammer(HistConfig())
        rates = _phi((3, B), seed=7)
        with pytest.raises(AssertionError):
            h.simulate_ewh_diff(rates, 500, B, free_running=True, dead_time_bins=5)


# =========================================================================== #
# 3f — EDH is a reduction of the EWH, not a separate detection model           #
# =========================================================================== #
EDH_PULSES = 4000


def _edh(n_edh_bins=4, phi=None, free_running=True, dt=0, seed=0, n_pulses=EDH_PULSES):
    h = HistogrammerEDH(HistConfig())
    phi = _phi((B,), seed=seed) if phi is None else phi
    return h.simulate_pixel_edh(
        phi, n_pulses, n_edh_bins, free_running, dt,
        generator=torch.Generator().manual_seed(seed),
    )


class TestEdhFromEwh:
    @pytest.mark.parametrize("free_running,dt", [(True, 0), (True, 3), (False, 0), (False, 5)])
    def test_photon_hist_matches_ewh_at_same_seed(self, free_running, dt):
        """EDH and EWH share one detection model, so a shared seed must give
        bit-identical histograms -- the whole point of the migration."""
        phi = _phi((B,), seed=2)
        edh_h = HistogrammerEDH(HistConfig())
        ewh_h = Histogrammer(HistConfig())
        hist_edh, _ = edh_h.simulate_pixel_edh(
            phi, EDH_PULSES, 4, free_running, dt,
            generator=torch.Generator().manual_seed(9),
        )
        hist_ewh = ewh_h.simulate_pixel_ewh(
            phi, EDH_PULSES, B, free_running, dt,
            generator=torch.Generator().manual_seed(9),
        )
        assert torch.equal(hist_edh, hist_ewh)

    def test_photon_hist_is_at_full_tdc_resolution(self):
        """n_hist_bins counts equi-depth bins; it must not rebin the transient."""
        hist, edh = _edh(n_edh_bins=4)
        assert hist.shape == (B,)
        assert edh.shape == (5,)

    def test_boundaries_are_monotonic_and_span_the_cycle(self):
        _, edh = _edh(n_edh_bins=4)
        assert float(edh[0]) == 0.0
        assert float(edh[-1]) == B
        assert torch.all(edh[1:] >= edh[:-1]), edh

    def test_bins_carry_equal_photon_mass(self):
        """The defining property: each equi-depth bin holds 1/N of the photons,
        even though the transient is far from flat.

        Boundaries are fractional -- photon_hist2edh interpolates within the bin
        straddling each quantile -- so the mass has to be integrated against a
        linearly interpolated CDF rather than by rounding to whole bins.
        """
        n_edh = 4
        phi = torch.linspace(0.05, 0.6, B)  # steep ramp, no single dominant bin
        hist, edh = _edh(n_edh_bins=n_edh, phi=phi, seed=4, n_pulses=20000)

        cum = torch.cat([torch.zeros(1), torch.cumsum(hist, 0)])

        def cdf(x):
            k = min(int(x), B - 1)
            return float(cum[k] + (x - k) * hist[k]) if x < B else float(cum[B])

        mass = [cdf(float(edh[i + 1])) - cdf(float(edh[i])) for i in range(n_edh)]
        target = float(hist.sum()) / n_edh
        for m in mass:
            assert m == pytest.approx(target, rel=1e-3), (mass, target)

    def test_flat_transient_gives_evenly_spaced_boundaries(self):
        n_edh = 4
        hist, edh = _edh(n_edh_bins=n_edh, phi=torch.full((B,), 0.3), seed=6, n_pulses=20000)
        expected = torch.linspace(0, B, n_edh + 1)
        assert torch.allclose(edh, expected, atol=1.0), (edh, expected)

    def test_negative_arrival_rates_do_not_produce_nan(self):
        """Convolution/offset math can emit small negatives; the old EDH loop fed
        them straight to torch.poisson and propagated NaN into the boundaries."""
        phi = _phi((B,), seed=8)
        phi[2] = -0.05
        hist, edh = _edh(phi=phi, seed=8)
        assert not torch.isnan(hist).any()
        assert not torch.isnan(edh).any()
        assert float(hist[2]) == 0.0

    def test_dead_time_reduces_counts(self):
        phi = _phi((B,), seed=1) * 2.0
        no_dt, _ = _edh(phi=phi, dt=0, seed=1)
        with_dt, _ = _edh(phi=phi, dt=4, seed=1)
        assert float(with_dt.sum()) < float(no_dt.sum())

    def test_simulate_edh_returns_one_entry_per_pixel(self):
        h = HistogrammerEDH(HistConfig())
        rates = _phi((3, B), seed=7)
        hists, edhs = h.simulate_edh(rates, 1000, 4, free_running=True, dead_time_bins=0)
        assert len(hists) == 3 and len(edhs) == 3
        assert all(x.shape == (B,) for x in hists)
        assert all(e.shape == (5,) for e in edhs)
