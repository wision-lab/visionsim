"""Single-pixel forward models vs the Monte-Carlo ground truth.

Scope, deliberately narrow (see ASPC_KNOWN_ISSUES.md §7):

  Case 1  free-running, non-paralyzable, dead time zero or non-zero
          -> ``calculate_distorted_transient``
  Case 2  gated (synchronous), non-paralyzable, **one detection per cycle**,
          dead time zero or non-zero
          -> ``calculate_distorted_transient_sync`` in single-hit mode

Batch (``batch_distorted_transient_*``) and differentiable (``simulate_ewh_diff``)
variants are out of scope here; they get their own pass once multi-pixel Monte
Carlo exists.

**Tolerances are derived, not hand-picked** (finding E4). Each histogram bin is a
binomial proportion over the realised detection count, so the 1-sigma error on a
bin holding fraction ``q`` of ``n`` detections is ``sqrt(q(1-q)/n)``. We compare
the *maximum* absolute deviation over ``n_bins`` bins, so the relevant threshold
is a few sigma on the worst bin; ``SIGMA_MULT`` sets how many. A hand-picked
``atol=0.02`` hides a real 0.006 bias at these pulse counts, which is how finding
M1 stayed open for as long as it did.
"""

from __future__ import annotations

import pytest
import torch

from visionsim.emulate.aspc.detector import (
    simulate_photon_timestamps,
    timestamps_to_histogram,
)
from visionsim.emulate.aspc.histogrammers import (
    calculate_distorted_transient,
    calculate_distorted_transient_sync,
)

B = 16
N_PULSES = 200_000
SIGMA_MULT = 5.0  # max-over-16-bins of a Gaussian needs ~3.2 sigma; 5 leaves headroom


def tolerance(total_detections: int, n_bins: int = B, q: float | None = None) -> float:
    """Binomial error bar for the max-abs-deviation statistic."""
    q = 1.0 / n_bins if q is None else q
    return SIGMA_MULT * (q * (1.0 - q) / max(total_detections, 1)) ** 0.5


def mc_shape(phi, dead_time_bins, free_running, cap, n_pulses=N_PULSES, seed=0):
    """Sampled ground-truth histogram, normalised, plus its total count."""
    timestamps = simulate_photon_timestamps(
        phi, n_pulses,
        dead_time_bins=dead_time_bins,
        free_running=free_running,
        paralyzable=False,
        max_detections_per_cycle=cap,
        generator=torch.Generator().manual_seed(seed),
    )
    hist = timestamps_to_histogram(timestamps, phi.shape[-1], B)
    total = float(hist.sum())
    return hist / total, total


# --- transient shapes, spanning what the model has to survive -------------- #
def phi_flat():
    return torch.full((B,), 0.15)

def phi_random_low():
    return torch.rand(B, generator=torch.Generator().manual_seed(3)) * 0.35 + 0.02

def phi_random_high():
    return torch.rand(B, generator=torch.Generator().manual_seed(7)) * 1.2 + 0.05

def phi_ramp():
    return torch.linspace(0.05, 0.6, B)

def phi_spike():
    """Lidar-like: a strong return on a dim ambient floor."""
    p = torch.full((B,), 0.01)
    p[5] = 3.0
    return p

SHAPES = [
    pytest.param(phi_flat, id="flat"),
    pytest.param(phi_random_low, id="random-low"),
    pytest.param(phi_random_high, id="random-high"),
    pytest.param(phi_ramp, id="ramp"),
    pytest.param(phi_spike, id="spike"),
]
DEAD_TIMES = [0, 1, 2, 4, 8]


# =========================================================================== #
# Case 1 — free-running, non-paralyzable                                      #
# =========================================================================== #
class TestFreeRunning:
    @pytest.mark.parametrize("make_phi", SHAPES)
    @pytest.mark.parametrize("dt", DEAD_TIMES)
    def test_matches_monte_carlo(self, make_phi, dt):
        phi = make_phi()
        mc, total = mc_shape(phi, dt, free_running=True, cap=None)
        model = calculate_distorted_transient(phi, dt, B)
        err = float((mc - model).abs().max())
        tol = tolerance(total)
        assert err < tol, f"max|MC - model| = {err:.5f} exceeds {SIGMA_MULT}-sigma = {tol:.5f}"

    @pytest.mark.parametrize("dt", [B, B + 1, 2 * B, 3 * B + 5])
    def test_dead_time_longer_than_one_cycle(self, dt):
        """M3: the model must not wrap dt modulo the cycle, which would silently
        turn an exact multiple of the cycle length into zero dead time."""
        phi = phi_random_low()
        mc, total = mc_shape(phi, dt, free_running=True, cap=None, n_pulses=400_000)
        model = calculate_distorted_transient(phi, dt, B)
        assert float((mc - model).abs().max()) < tolerance(total)

    def test_zero_dead_time_is_proportional_to_phi(self):
        """dt=0 means every arriving photon is detected, so the shape is phi
        itself -- not the per-bin detection probability 1 - exp(-phi)."""
        phi = phi_random_high()
        model = calculate_distorted_transient(phi, 0, B)
        assert torch.allclose(model, phi / phi.sum(), atol=1e-6)

        occupancy = 1.0 - torch.exp(-phi)
        assert not torch.allclose(model, occupancy / occupancy.sum(), atol=1e-3)

    def test_dead_time_biases_detections_earlier(self):
        """Pile-up: blocking later bins shifts mass toward the start of the cycle."""
        phi = phi_ramp()
        idx = torch.arange(B, dtype=torch.float64)
        com = [float((idx * calculate_distorted_transient(phi, dt, B).double()).sum())
               for dt in (1, 4, 8)]
        assert com[0] > com[1] > com[2], com


# =========================================================================== #
# Case 2 — gated, single-hit (one detection per cycle)                        #
# =========================================================================== #
SINGLE_HIT_DT = B  # dead_time_bins >= n_hist_bins selects single-hit


class TestGatedSingleHit:
    @pytest.mark.parametrize("make_phi", SHAPES)
    def test_matches_monte_carlo(self, make_phi):
        phi = make_phi()
        mc, total = mc_shape(phi, 0, free_running=False, cap=1)
        model = calculate_distorted_transient_sync(phi, SINGLE_HIT_DT, B)
        err = float((mc - model).abs().max())
        tol = tolerance(total)
        assert err < tol, f"max|MC - model| = {err:.5f} exceeds {SIGMA_MULT}-sigma = {tol:.5f}"

    @pytest.mark.parametrize("dt", [0, 1, 3, 8, B, 2 * B])
    def test_result_is_independent_of_dead_time(self, dt):
        """With one detection per cycle and a re-arm at every cycle boundary the
        dead time can never bind, so the sampled truth must not move at all."""
        phi = phi_random_low()
        mc, total = mc_shape(phi, dt, free_running=False, cap=1)
        model = calculate_distorted_transient_sync(phi, SINGLE_HIT_DT, B)
        assert float((mc - model).abs().max()) < tolerance(total)

    def test_reduces_to_first_photon_wins(self):
        """Single-hit is exactly Coates: p_j * prod_{m<j} (1 - p_m)."""
        phi = phi_random_high()
        p = 1.0 - torch.exp(-phi.double())
        survival = torch.cat([torch.ones(1, dtype=torch.float64), torch.cumprod(1.0 - p, 0)[:-1]])
        coates = p * survival
        coates = coates / coates.sum()
        model = calculate_distorted_transient_sync(phi, SINGLE_HIT_DT, B).double()
        assert torch.allclose(model, coates, atol=1e-9)

    def test_total_detections_follow_the_analytic_rate(self):
        """The model is a *shape* only. Converting it to counts needs
        n_pulses * (1 - exp(-sum(phi))), not n_pulses -- most cycles produce no
        detection at all at low flux."""
        phi = phi_random_low()
        _, total = mc_shape(phi, 0, free_running=False, cap=1)
        expected = N_PULSES * float(1.0 - torch.exp(-phi.sum()))
        assert total == pytest.approx(expected, rel=0.01)
        assert total < N_PULSES


# =========================================================================== #
# Shared contract                                                             #
# =========================================================================== #
class TestModelContract:
    @pytest.mark.parametrize("model", [calculate_distorted_transient,
                                       calculate_distorted_transient_sync])
    @pytest.mark.parametrize("dt", [0, 3, B])
    def test_returns_a_normalised_distribution(self, model, dt):
        out = model(phi_random_low(), dt, B)
        assert out.shape == (B,)
        assert float(out.sum()) == pytest.approx(1.0, abs=1e-6)
        assert torch.all(out >= 0)

    @pytest.mark.parametrize("model", [calculate_distorted_transient,
                                       calculate_distorted_transient_sync])
    def test_zero_flux_does_not_nan(self, model):
        out = model(torch.zeros(B), 3, B)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize("model", [calculate_distorted_transient,
                                       calculate_distorted_transient_sync])
    def test_negative_rates_are_clamped(self, model):
        phi = phi_random_low()
        phi[2] = -0.05
        out = model(phi, 3, B)
        assert not torch.isnan(out).any()
        assert float(out[2]) == pytest.approx(0.0, abs=1e-9)


# =========================================================================== #
# Differentiability                                                           #
# =========================================================================== #
# Both models are pure torch, so autograd flows end to end -- including through
# torch.linalg.eig in the free-running model. Pinned here because it is easy to
# break silently: writing into a preallocated tensor inside the recursion, or
# dropping to numpy for one line, breaks backward() without touching any of the
# numerical tests above.
DIFF_CASES = [
    pytest.param(calculate_distorted_transient, 0, id="free-running-dt0"),
    pytest.param(calculate_distorted_transient, 3, id="free-running-dt3"),
    pytest.param(calculate_distorted_transient_sync, 0, id="gated-dt0"),
    pytest.param(calculate_distorted_transient_sync, 3, id="gated-multi-hit"),
    pytest.param(calculate_distorted_transient_sync, B, id="gated-single-hit"),
]


def _centre_of_mass(out):
    return (out * torch.arange(out.shape[-1], dtype=out.dtype)).sum()


class TestDifferentiable:
    @pytest.mark.parametrize("model,dt", DIFF_CASES)
    def test_gradient_flows(self, model, dt):
        phi = (phi_random_low().double() + 0.0).requires_grad_(True)
        out = model(phi, dt, B)
        assert out.requires_grad, "graph broken -- output detached from input"
        _centre_of_mass(out).backward()
        assert phi.grad is not None
        assert not torch.isnan(phi.grad).any()
        assert float(phi.grad.abs().sum()) > 0

    @pytest.mark.parametrize("model,dt", DIFF_CASES)
    def test_gradient_matches_finite_differences(self, model, dt):
        base = phi_random_low().double()
        phi = base.clone().requires_grad_(True)
        _centre_of_mass(model(phi, dt, B)).backward()

        def f(p):
            with torch.no_grad():
                return float(_centre_of_mass(model(p, dt, B)))

        eps = 1e-6
        for i in (0, 3, B - 1):
            up, dn = base.clone(), base.clone()
            up[i] += eps
            dn[i] -= eps
            fd = (f(up) - f(dn)) / (2 * eps)
            analytic = float(phi.grad[i])
            assert analytic == pytest.approx(fd, abs=1e-4, rel=2e-2), (
                f"bin {i}: analytic {analytic:.6f} vs finite-difference {fd:.6f}")

    @pytest.mark.parametrize("model,dt", DIFF_CASES)
    def test_no_numpy_round_trip(self, model, dt):
        """A float32 input must come back float32, on the same device."""
        phi = phi_random_low()
        out = model(phi, dt, B)
        assert out.dtype == phi.dtype
        assert out.device == phi.device
