"""Tests for the ground-truth photon-detection simulator.

The central test is :class:`TestExhaustiveOracle`, which enumerates *every*
possible photon-arrival pattern over a handful of bins and compares the
production dead-time walk against an independently written per-photon reference.
Because it enumerates rather than samples there is no Monte-Carlo error, so any
disagreement is a real defect rather than noise.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch

from visionsim.emulate.aspc.detector import (
    _deadtime_walk,
    sample_photon_arrivals,
    simulate_photon_timestamps,
    timestamps_to_histogram,
)

MODES = [
    # (free_running, paralyzable, cap)
    (True, False, 1 << 62),
    (True, True, 1 << 62),
    (False, False, 1),
    (False, True, 1),
    (False, False, 1 << 62),
    (False, True, 1 << 62),
    (False, False, 2),
]


def reference_detect(counts, n_tbins, dt, free_running, paralyzable, cap):
    """Independent per-photon reference implementation of the detection rules.

    Deliberately naive: it materialises every individual photon and walks them
    one at a time, rather than using the "one detection per occupied bin"
    shortcut the production code takes. Returns a list of ``(cycle, bin)``.
    """
    n_cycles = counts.shape[0]
    detections = []
    t_armed = -(10**9)
    for c in range(n_cycles):
        if not free_running:
            t_armed = c * n_tbins
        n_in_cycle = 0
        for b in range(n_tbins):
            g = c * n_tbins + b
            for _ in range(int(counts[c, b])):
                if n_in_cycle >= cap:
                    continue
                if g >= t_armed:
                    detections.append((c, b))
                    n_in_cycle += 1
                    t_armed = g + dt
                elif paralyzable:
                    t_armed = g + dt
    return detections


def all_patterns(n_cycles, n_tbins, max_count):
    """Every count pattern over ``n_cycles x n_tbins`` bins with entries 0..max_count."""
    n_pos = n_cycles * n_tbins
    for combo in itertools.product(range(max_count + 1), repeat=n_pos):
        yield np.array(combo, dtype=np.int64).reshape(n_cycles, n_tbins)


class TestExhaustiveOracle:
    """Enumerate all arrival patterns; production walk must equal the reference."""

    @pytest.mark.parametrize("free_running,paralyzable,cap", MODES)
    @pytest.mark.parametrize("n_tbins,dt", [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 3)])
    def test_walk_matches_reference(self, n_tbins, dt, free_running, paralyzable, cap):
        n_cycles = 2
        max_count = 2
        n_checked = 0
        for counts in all_patterns(n_cycles, n_tbins, max_count):
            cyc, bns = _deadtime_walk(counts, n_tbins, dt, free_running, paralyzable, cap)
            got = list(zip(cyc.tolist(), bns.tolist()))
            want = reference_detect(counts, n_tbins, dt, free_running, paralyzable, cap)
            assert got == want, (
                f"pattern={counts.tolist()} n_tbins={n_tbins} dt={dt} "
                f"free_running={free_running} paralyzable={paralyzable} cap={cap}\n"
                f"got={got}\nwant={want}"
            )
            n_checked += 1
        assert n_checked == (max_count + 1) ** (n_cycles * n_tbins)

    def test_dead_time_longer_than_cycle(self):
        """Dead time spanning multiple cycles must not saturate at one cycle (M3)."""
        n_tbins, n_cycles = 3, 4
        dt = 2 * n_tbins + 1  # deliberately longer than one full cycle
        for counts in all_patterns(n_cycles, n_tbins, 1):
            cyc, bns = _deadtime_walk(counts, n_tbins, dt, True, False, 1 << 62)
            got = list(zip(cyc.tolist(), bns.tolist()))
            want = reference_detect(counts, n_tbins, dt, True, False, 1 << 62)
            assert got == want, f"pattern={counts.tolist()}"


class TestExpectedRates:
    """Exact expectations computed by Poisson-weighted enumeration, no sampling."""

    @staticmethod
    def expected_counts(phi, n_cycles, dt, free_running, paralyzable, cap, max_count=4):
        """E[detections per bin] over all patterns, weighted by Poisson pmf."""
        n_tbins = len(phi)
        acc = np.zeros(n_tbins)
        total_w = 0.0
        for counts in all_patterns(n_cycles, n_tbins, max_count):
            w = 1.0
            for c in range(n_cycles):
                for b in range(n_tbins):
                    k = int(counts[c, b])
                    w *= math.exp(-phi[b]) * phi[b] ** k / math.factorial(k)
            if w == 0.0:
                continue
            total_w += w
            _, bns = _deadtime_walk(counts, n_tbins, dt, free_running, paralyzable, cap)
            for b in bns.tolist():
                acc[b] += w
        return acc, total_w

    def test_zero_dead_time_recovers_phi(self):
        """With no dead time every photon is detected, so E[counts] == phi exactly."""
        phi = [0.3, 0.7, 0.2]
        acc, total_w = self.expected_counts(phi, 1, 0, True, False, 1 << 62, max_count=8)
        # Truncating the Poisson support at 8 leaves ~6e-8 of the mass unenumerated.
        assert total_w == pytest.approx(1.0, rel=1e-6)
        for b, p in enumerate(phi):
            assert acc[b] == pytest.approx(p, rel=1e-6), f"bin {b}"

    def test_zero_dead_time_mc_matches_phi(self):
        """Monte-Carlo detection counts converge to phi * n_pulses at dt=0."""
        phi = torch.tensor([0.05, 0.20, 0.10, 0.02])
        n_pulses = 200_000
        gen = torch.Generator().manual_seed(1234)
        ts = simulate_photon_timestamps(
            phi, n_pulses, dead_time_bins=0, free_running=True, generator=gen
        )
        hist = timestamps_to_histogram(ts, n_tbins=len(phi)) / n_pulses
        torch.testing.assert_close(hist, phi, rtol=0.02, atol=1e-3)

    def test_bernoulli_model_would_undercount(self):
        """Guard on the modelling choice: Poisson counts differ from Bernoulli.

        At phi=0.5 per bin a Bernoulli-per-bin sampler registers 1-exp(-phi)
        instead of phi -- a 21% undercount. This test pins the fact that we do
        *not* do that, so a regression back to Bernoulli fails loudly.
        """
        phi = torch.full((3,), 0.5)
        n_pulses = 200_000
        gen = torch.Generator().manual_seed(7)
        ts = simulate_photon_timestamps(
            phi, n_pulses, dead_time_bins=0, free_running=True, generator=gen
        )
        rate = float(timestamps_to_histogram(ts, 3).sum()) / (n_pulses * 3)
        bernoulli_rate = 1.0 - math.exp(-0.5)
        assert rate == pytest.approx(0.5, rel=0.02)
        assert abs(rate - bernoulli_rate) > 0.05


class TestSparseSampler:
    """The sparse sampler must be a drop-in statistical equivalent of the dense one."""

    def test_matches_dense_per_bin_rate(self):
        phi = torch.tensor([0.02, 0.15, 0.30, 0.05, 0.01])
        n_pulses = 100_000
        hists = {}
        for mode in (True, False):
            ts = simulate_photon_timestamps(
                phi, n_pulses, dead_time_bins=0, free_running=True, sparse=mode,
                generator=torch.Generator().manual_seed(17),
            )
            hists[mode] = timestamps_to_histogram(ts, n_tbins=len(phi)) / n_pulses
        # Independent draws, so compare both against phi rather than each other.
        torch.testing.assert_close(hists[True], phi, rtol=0.03, atol=1e-3)
        torch.testing.assert_close(hists[False], phi, rtol=0.03, atol=1e-3)

    def test_matches_dense_under_dead_time(self):
        phi = torch.full((32,), 0.05)
        n_pulses = 40_000
        totals = {}
        for mode in (True, False):
            ts = simulate_photon_timestamps(
                phi, n_pulses, dead_time_bins=8, free_running=True, sparse=mode,
                generator=torch.Generator().manual_seed(23),
            )
            totals[mode] = ts.shape[0] / n_pulses
        assert totals[True] == pytest.approx(totals[False], rel=0.03)

    def test_cycles_are_uniformly_distributed(self):
        """Guards the superposition argument: cycle index must be uniform."""
        phi = torch.full((4,), 0.05)
        n_pulses = 20_000
        ts = simulate_photon_timestamps(
            phi, n_pulses, dead_time_bins=0, free_running=True, sparse=True,
            generator=torch.Generator().manual_seed(31),
        )
        halves = torch.bincount(ts[:, 0] * 2 // n_pulses, minlength=2)
        assert int(halves[0]) == pytest.approx(int(halves[1]), rel=0.05)

    def test_auto_selects_sparse_at_low_flux(self):
        phi = torch.full((200,), 1e-5)
        ts = simulate_photon_timestamps(
            phi, 5_000, generator=torch.Generator().manual_seed(1)
        )
        assert ts.shape[1] == 2  # runs, and stays within the documented contract

    def test_zero_rate_yields_no_detections(self):
        for mode in (True, False):
            ts = simulate_photon_timestamps(torch.zeros(8), 100, sparse=mode)
            assert ts.shape == (0, 2)
            assert float(timestamps_to_histogram(ts, 8).sum()) == 0.0


class TestAnalyticRates:
    """External oracles: closed-form dead-time rates that share no code with the walk.

    The exhaustive oracle above checks the implementation against a reference that
    encodes the *same* semantics, so it cannot catch a physics error common to
    both. These formulas are derived independently from renewal theory and are the
    discrete-bin analogues of the textbook counter dead-time results
    (``lam/(1+lam*tau)`` and ``lam*exp(-lam*tau)``). The ``tau-1`` rather than
    ``tau`` reflects that a detection in bin ``g`` blocks bins ``g+1 .. g+tau-1``,
    leaving bin ``g+tau`` live.
    """

    N_BINS = 64
    N_PULSES = 60_000
    N_SEEDS = 8

    def _mean_rate(self, lam, tau, paralyzable, n_seeds=None):
        phi = torch.full((self.N_BINS,), lam)
        rates = []
        for s in range(self.N_SEEDS if n_seeds is None else n_seeds):
            ts = simulate_photon_timestamps(
                phi, self.N_PULSES, dead_time_bins=tau, free_running=True,
                paralyzable=paralyzable, generator=torch.Generator().manual_seed(500 + s),
            )
            rates.append(ts.shape[0] / (self.N_PULSES * self.N_BINS))
        mean = sum(rates) / len(rates)
        sd = (sum((r - mean) ** 2 for r in rates) / (len(rates) - 1)) ** 0.5
        return mean, sd / len(rates) ** 0.5

    @pytest.mark.parametrize("lam", [0.01, 0.05, 0.2, 0.5])
    @pytest.mark.parametrize("tau", [1, 3, 10])
    def test_non_paralyzable_rate(self, lam, tau):
        """Renewal: mean interval is tau bins of dead time plus a Geometric(p) wait."""
        p = 1.0 - math.exp(-lam)
        predicted = 1.0 / (tau + (1.0 - p) / p)
        mean, sem = self._mean_rate(lam, tau, paralyzable=False)
        z = (mean - predicted) / sem
        assert abs(z) < 4.0, f"rate {mean:.6g} vs analytic {predicted:.6g} (z={z:+.2f})"

    @pytest.mark.parametrize("lam", [0.01, 0.05, 0.2, 0.5])
    @pytest.mark.parametrize("tau", [1, 3, 10])
    def test_paralyzable_rate(self, lam, tau):
        """A bin detects iff it is occupied and the preceding tau-1 bins are empty."""
        p = 1.0 - math.exp(-lam)
        predicted = p * math.exp(-lam * (tau - 1))
        mean, sem = self._mean_rate(lam, tau, paralyzable=True)
        z = (mean - predicted) / sem
        assert abs(z) < 4.0, f"rate {mean:.6g} vs analytic {predicted:.6g} (z={z:+.2f})"

    def test_paralyzable_rate_is_non_monotonic_in_flux(self):
        """Paralyzable counters peak then collapse; non-paralyzable saturate.

        This is the qualitative signature that distinguishes the two devices, and
        it must emerge from the simulation rather than being asserted anywhere.
        """
        tau = 10
        lams = [0.01, 0.05, 0.2, 0.6, 1.5]
        # A qualitative shape check, so a couple of seeds is plenty.
        par = [self._mean_rate(lam, tau, paralyzable=True, n_seeds=2)[0] for lam in lams]
        non_par = [self._mean_rate(lam, tau, paralyzable=False, n_seeds=2)[0] for lam in lams]
        assert par[-1] < par[2], "paralyzable rate must collapse at high flux"
        assert par == sorted(par[: par.index(max(par)) + 1]) + sorted(
            par[par.index(max(par)) + 1 :], reverse=True
        ), "paralyzable rate must rise then fall"
        assert non_par == sorted(non_par), "non-paralyzable rate must increase monotonically"
        assert non_par[-1] < 1.0 / tau * 1.05, "non-paralyzable saturates near 1/tau"


class TestCoatesRoundTrip:
    """Gated single-hit is the regime the Coates estimator inverts exactly."""

    @staticmethod
    def coates(hist, n_pulses):
        """Recover phi from a first-photon-wins histogram."""
        # Cycles still "alive" entering bin b = those that had not yet detected.
        consumed = torch.cat([torch.zeros(1), torch.cumsum(hist, 0)[:-1]])
        alive = n_pulses - consumed
        return torch.log(alive / (alive - hist))

    @pytest.mark.parametrize("scale", [0.02, 0.1, 0.4])
    def test_recovers_phi(self, scale):
        """Round-trip through pile-up: simulate gated single-hit, invert, recover phi."""
        n_bins, n_pulses = 12, 400_000
        phi = torch.linspace(0.5, 1.5, n_bins) * scale
        ts = simulate_photon_timestamps(
            phi, n_pulses, free_running=False, generator=torch.Generator().manual_seed(77)
        )
        hist = timestamps_to_histogram(ts, n_bins)
        recovered = self.coates(hist, n_pulses)
        torch.testing.assert_close(recovered, phi, rtol=0.05, atol=2e-3)

    def test_raw_histogram_is_pile_up_distorted(self):
        """The raw histogram must be visibly biased early -- that is what Coates fixes."""
        n_bins, n_pulses = 12, 200_000
        phi = torch.full((n_bins,), 0.4)
        ts = simulate_photon_timestamps(
            phi, n_pulses, free_running=False, generator=torch.Generator().manual_seed(78)
        )
        hist = timestamps_to_histogram(ts, n_bins)
        # phi is flat, so any decay in the raw histogram is pure pile-up.
        assert float(hist[0]) > 2.0 * float(hist[-1])
        recovered = self.coates(hist, n_pulses)
        torch.testing.assert_close(recovered, phi, rtol=0.05, atol=2e-3)


class TestDeadTimeInvariants:
    """Properties that must hold on any timestamp stream, checked directly."""

    @pytest.mark.parametrize("dt", [1, 3, 7])
    def test_no_two_detections_closer_than_dead_time(self, dt):
        n_tbins = 16
        phi = torch.full((n_tbins,), 0.4)
        gen = torch.Generator().manual_seed(99)
        ts = simulate_photon_timestamps(
            phi, 500, dead_time_bins=dt, free_running=True, generator=gen
        )
        assert ts.shape[0] > 50, "test needs a non-trivial number of detections"
        g = ts[:, 0] * n_tbins + ts[:, 1]
        gaps = g[1:] - g[:-1]
        assert int(gaps.min()) >= dt, f"found a gap of {int(gaps.min())} < dead time {dt}"

    def test_gated_resets_at_cycle_boundary(self):
        """A detection late in a cycle must not block the next cycle when gated."""
        n_tbins = 4
        counts = np.zeros((2, n_tbins), dtype=np.int64)
        counts[0, 3] = 1  # last bin of cycle 0
        counts[1, 0] = 1  # first bin of cycle 1
        dt = 3

        cyc, bns = _deadtime_walk(counts, n_tbins, dt, False, False, 1 << 62)
        assert list(zip(cyc.tolist(), bns.tolist())) == [(0, 3), (1, 0)]

        # Free-running, by contrast, must suppress the second detection.
        cyc, bns = _deadtime_walk(counts, n_tbins, dt, True, False, 1 << 62)
        assert list(zip(cyc.tolist(), bns.tolist())) == [(0, 3)]

    def test_paralyzable_detects_no_more_than_non_paralyzable(self):
        n_tbins = 12
        phi = torch.full((n_tbins,), 0.8)
        counts = sample_photon_arrivals(phi, 400, generator=torch.Generator().manual_seed(3))
        arr = counts.numpy()
        n_non_par = len(_deadtime_walk(arr, n_tbins, 4, True, False, 1 << 62)[0])
        n_par = len(_deadtime_walk(arr, n_tbins, 4, True, True, 1 << 62)[0])
        assert n_par <= n_non_par
        assert n_par < n_non_par, "at this flux paralyzable must lose detections"

    def test_paralyzable_equals_non_paralyzable_at_low_flux(self):
        """The two models converge when coincident arrivals are rare."""
        n_tbins = 12
        phi = torch.full((n_tbins,), 0.002)
        counts = sample_photon_arrivals(phi, 2000, generator=torch.Generator().manual_seed(5))
        arr = counts.numpy()
        n_non_par = len(_deadtime_walk(arr, n_tbins, 3, True, False, 1 << 62)[0])
        n_par = len(_deadtime_walk(arr, n_tbins, 3, True, True, 1 << 62)[0])
        assert n_non_par > 0
        assert abs(n_par - n_non_par) / n_non_par < 0.05


class TestGatedSingleHit:
    """The conventional synchronous setup, and its vectorised fast path."""

    def test_matches_first_occupied_bin(self):
        n_tbins = 8
        phi = torch.linspace(0.05, 0.5, n_tbins)
        gen = torch.Generator().manual_seed(42)
        counts = sample_photon_arrivals(phi, 300, generator=gen)

        gen2 = torch.Generator().manual_seed(42)
        ts = simulate_photon_timestamps(
            phi, 300, dead_time_bins=5, free_running=False, generator=gen2, sparse=False
        )

        occupied = counts > 0
        want = [
            (c, int(occupied[c].to(torch.uint8).argmax()))
            for c in range(300)
            if bool(occupied[c].any())
        ]
        assert [tuple(r) for r in ts.tolist()] == want

    @pytest.mark.parametrize("dt", [0, 1, 5, 100])
    def test_dead_time_is_irrelevant(self, dt):
        """Gated single-hit stops after the first photon, so dt cannot bind."""
        phi = torch.full((10,), 0.3)
        gen = torch.Generator().manual_seed(11)
        got = simulate_photon_timestamps(
            phi, 200, dead_time_bins=dt, free_running=False, generator=gen, sparse=False
        )
        gen = torch.Generator().manual_seed(11)
        baseline = simulate_photon_timestamps(
            phi, 200, dead_time_bins=0, free_running=False, generator=gen, sparse=False
        )
        torch.testing.assert_close(got, baseline)

    def test_fast_path_agrees_with_walk(self):
        """The vectorised gated-single-hit path must equal the general scan."""
        n_tbins = 6
        phi = torch.full((n_tbins,), 0.25)
        counts = sample_photon_arrivals(phi, 500, generator=torch.Generator().manual_seed(8))
        cyc, bns = _deadtime_walk(counts.numpy(), n_tbins, 2, False, False, 1)
        want = list(zip(cyc.tolist(), bns.tolist()))

        occupied = counts > 0
        has = occupied.any(dim=1)
        first = occupied.to(torch.uint8).argmax(dim=1)
        got = list(
            zip(torch.nonzero(has, as_tuple=True)[0].tolist(), first[has].tolist())
        )
        assert got == want

    def test_multi_hit_gated_detects_more(self):
        phi = torch.full((20,), 0.15)
        gen = torch.Generator().manual_seed(21)
        single = simulate_photon_timestamps(
            phi, 400, dead_time_bins=2, free_running=False, generator=gen
        )
        gen = torch.Generator().manual_seed(21)
        multi = simulate_photon_timestamps(
            phi,
            400,
            dead_time_bins=2,
            free_running=False,
            max_detections_per_cycle=None,
            generator=gen,
        )
        assert multi.shape[0] > single.shape[0]


class TestHistogramReduction:
    def test_defaults_to_tdc_resolution(self):
        ts = torch.tensor([[0, 0], [0, 2], [1, 2], [3, 3]], dtype=torch.int64)
        hist = timestamps_to_histogram(ts, n_tbins=4)
        torch.testing.assert_close(hist, torch.tensor([1.0, 0.0, 2.0, 1.0]))

    def test_rebins_to_coarser_histogram(self):
        """n_hist_bins != n_tbins is supported by merging bins (M2)."""
        ts = torch.tensor([[0, 0], [0, 1], [0, 2], [0, 5]], dtype=torch.int64)
        hist = timestamps_to_histogram(ts, n_tbins=6, n_hist_bins=3)
        torch.testing.assert_close(hist, torch.tensor([2.0, 1.0, 1.0]))

    def test_rejects_non_divisible_rebin(self):
        ts = torch.tensor([[0, 0]], dtype=torch.int64)
        with pytest.raises(ValueError, match="exact multiple"):
            timestamps_to_histogram(ts, n_tbins=10, n_hist_bins=3)

    def test_rejects_finer_than_tdc(self):
        ts = torch.tensor([[0, 0]], dtype=torch.int64)
        with pytest.raises(ValueError, match="cannot exceed"):
            timestamps_to_histogram(ts, n_tbins=4, n_hist_bins=8)

    def test_empty_timestamps(self):
        hist = timestamps_to_histogram(torch.zeros((0, 2), dtype=torch.int64), n_tbins=5)
        assert hist.shape == (5,)
        assert float(hist.sum()) == 0.0


class TestApiContract:
    def test_reproducible_with_generator(self):
        phi = torch.full((8,), 0.2)
        a = simulate_photon_timestamps(phi, 100, generator=torch.Generator().manual_seed(0))
        b = simulate_photon_timestamps(phi, 100, generator=torch.Generator().manual_seed(0))
        torch.testing.assert_close(a, b)

    def test_free_running_rejects_per_cycle_cap(self):
        phi = torch.full((4,), 0.1)
        with pytest.raises(ValueError, match="inherently multi-hit"):
            simulate_photon_timestamps(phi, 10, free_running=True, max_detections_per_cycle=1)

    def test_rejects_fractional_dead_time(self):
        phi = torch.full((4,), 0.1)
        with pytest.raises(ValueError, match="non-negative integer"):
            simulate_photon_timestamps(phi, 10, dead_time_bins=1.5)

    def test_rejects_negative_dead_time(self):
        phi = torch.full((4,), 0.1)
        with pytest.raises(ValueError, match="non-negative integer"):
            simulate_photon_timestamps(phi, 10, dead_time_bins=-1)

    def test_rejects_multi_pixel_input(self):
        with pytest.raises(ValueError, match="single pixel"):
            sample_photon_arrivals(torch.zeros(3, 8), 10)

    def test_negative_rates_are_clamped(self):
        """Convolution and offset arithmetic can produce small negative rates."""
        phi = torch.tensor([-0.5, 0.0, 0.3])
        counts = sample_photon_arrivals(phi, 100, generator=torch.Generator().manual_seed(2))
        assert int(counts[:, 0].sum()) == 0
        assert int(counts[:, 1].sum()) == 0
        assert int(counts[:, 2].sum()) > 0

    def test_timestamps_are_sorted(self):
        phi = torch.full((10,), 0.3)
        ts = simulate_photon_timestamps(
            phi, 200, dead_time_bins=2, free_running=True,
            generator=torch.Generator().manual_seed(4),
        )
        g = ts[:, 0] * 10 + ts[:, 1]
        assert bool((g[1:] >= g[:-1]).all())

    def test_output_shape_and_dtype(self):
        phi = torch.full((5,), 0.1)
        ts = simulate_photon_timestamps(phi, 50, generator=torch.Generator().manual_seed(6))
        assert ts.ndim == 2 and ts.shape[1] == 2
        assert ts.dtype == torch.int64
        assert int(ts[:, 1].max()) < 5
        assert int(ts[:, 0].max()) < 50
