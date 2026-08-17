"""Stage 4 — The φ pipeline: ``calculate_transients`` (flat-wall oracle).

Everything in Stages 1–3 tested what happens *after* φ (the per-bin photon arrival
rate). This file tests what *produces* φ. A correct forward model fed a wrong φ is
still wrong, so this precedes further histogrammer work.

φ is the *arrival* process — scene geometry × laser × ambient. Gating and dead time
govern which arrivals become *detections*, which is entirely downstream. Nothing in
this file therefore depends on gated vs free-running operation.

**Semantics pinned here.** A SPAD pixel sees the *average* radiance over its FOV.
Each render pixel subtends ``1/n_fov_pixels`` of that FOV, so contributions are
weighted by ``1/n_fov_pixels`` rather than summed raw. Consequences:

  * a flat wall of uniform irradiance ``I`` yields a spike of height ``I``,
    independent of how many render pixels the FOV covers;
  * render resolution is a *sampling* choice and no longer changes the collected
    photon count (finding F3);
  * vignetting still attenuates the total, because it attenuates real signal.

Scope: **single frame only** — multi-frame semantics (T2) remain undecided, and the
one test here just pins today's behaviour so it cannot change silently.
"""

import numpy as np
import pytest
import torch

from visionsim.emulate.aspc.histogrammers import HistConfig, Histogrammer
from visionsim.emulate.aspc.utils import tof2depth, ureg

B = 20  # transient bins
MAX_DEPTH = 10.0  # metres
H = W = 8


@pytest.fixture
def hist():
    return Histogrammer(HistConfig())


def _scene(depth, irradiance=1.0, h=H, w=W, n_frames=1):
    """Flat wall: every pixel at the same depth with the same irradiance."""
    irr = torch.full((n_frames, h, w), float(irradiance))
    dep = torch.full((n_frames, h, w), float(depth))
    off = torch.zeros(n_frames, h, w)
    return irr, dep, off


def _full_mask(h=H, w=W):
    return torch.ones(1, h, w)


def _expected_bin(depth, n_bins=B, max_depth=MAX_DEPTH):
    return int(np.floor(depth * n_bins / max_depth)) % n_bins


# =========================================================================== #
# 4a — depth -> bin mapping                                                    #
# =========================================================================== #
class TestDepthToBin:
    @pytest.mark.parametrize("depth", [0.5, 2.5, 4.9, 7.3, 9.9])
    def test_flat_wall_is_a_single_spike_at_the_right_bin(self, hist, depth):
        irr, dep, off = _scene(depth)
        t, _ = hist.calculate_transients(irr, dep, off, _full_mask(), B, MAX_DEPTH)
        nonzero = torch.nonzero(t[0]).flatten().tolist()
        assert nonzero == [_expected_bin(depth)], f"depth={depth} landed in bins {nonzero}"

    def test_spike_height_is_mean_irradiance_not_sum(self, hist):
        """The SPAD averages over its FOV, so the height is the irradiance itself."""
        irr, dep, off = _scene(3.0, irradiance=1.0)
        t, _ = hist.calculate_transients(irr, dep, off, _full_mask(), B, MAX_DEPTH)
        assert float(t[0].max()) == pytest.approx(1.0)

    def test_spike_height_is_linear_in_irradiance(self, hist):
        lo, _, _ = _scene(3.0, irradiance=1.0)
        hi, dep, off = _scene(3.0, irradiance=2.5)
        t_lo, _ = hist.calculate_transients(lo, dep, off, _full_mask(), B, MAX_DEPTH)
        t_hi, _ = hist.calculate_transients(hi, dep, off, _full_mask(), B, MAX_DEPTH)
        assert float(t_hi.sum()) == pytest.approx(2.5 * float(t_lo.sum()))

    def test_mean_irradiance_is_preserved(self, hist):
        """Binning neither creates nor destroys signal."""
        irr, dep, off = _scene(4.0, irradiance=0.7)
        t, _ = hist.calculate_transients(irr, dep, off, _full_mask(), B, MAX_DEPTH)
        assert float(t.sum()) == pytest.approx(float(irr[0].mean()))

    def test_bin_is_monotonic_in_depth(self, hist):
        bins = []
        for d in [1.0, 3.0, 5.0, 7.0, 9.0]:
            irr, dep, off = _scene(d)
            t, _ = hist.calculate_transients(irr, dep, off, _full_mask(), B, MAX_DEPTH)
            bins.append(int(t[0].argmax()))
        assert bins == sorted(bins) and len(set(bins)) == len(bins)


# =========================================================================== #
# 4b — multi-surface scenes                                                    #
# =========================================================================== #
class TestTwoPlanes:
    def test_two_depths_give_two_spikes_with_correct_split(self, hist):
        irr, dep, off = _scene(1.0)
        dep[0, :4, :] = 2.0  # top half near
        dep[0, 4:, :] = 6.0  # bottom half far
        t, _ = hist.calculate_transients(irr, dep, off, _full_mask(), B, MAX_DEPTH)
        near, far = _expected_bin(2.0), _expected_bin(6.0)
        assert torch.nonzero(t[0]).flatten().tolist() == sorted([near, far])
        assert float(t[0, near]) == pytest.approx(0.5)  # half the FOV
        assert float(t[0, far]) == pytest.approx(0.5)

    def test_tilted_plane_spreads_across_bins(self, hist):
        irr, dep, off = _scene(1.0)
        dep[0] = torch.linspace(0.5, 9.5, H).view(H, 1).expand(H, W)
        t, _ = hist.calculate_transients(irr, dep, off, _full_mask(), B, MAX_DEPTH)
        assert int(torch.count_nonzero(t[0])) == H  # one bin per distinct row depth
        assert float(t.sum()) == pytest.approx(1.0)


# =========================================================================== #
# 4c — FOV mask semantics                                                      #
# =========================================================================== #
class TestFovMaskSemantics:
    @pytest.mark.parametrize("shape", [(100, 100), (300, 400), (37, 53)])
    def test_unit_fov_covers_every_pixel(self, hist, shape):
        m = hist.get_pixel_fov_mask(torch.zeros(*shape), 0, 1, 0, 1, vignette=False)
        assert int(m.sum()) == shape[0] * shape[1]

    def test_default_full_scene_fov_is_exactly_full(self, hist):
        """The default 'full scene' entry must be 1.0, not 0.999: bounds are rounded to
        integer pixels, so 0.999 silently drops the last row/col at some resolutions."""
        full = HistConfig().pixel_fov_list[-1]
        assert full == [0, 1.0, 0, 1.0]
        for shape in [(300, 400), (1000, 1000)]:
            m = hist.get_pixel_fov_mask(torch.zeros(*shape), *full, vignette=False)
            assert int(m.sum()) == shape[0] * shape[1]

    @pytest.mark.parametrize("fov", [[0, 1, 0, 1], [0, 0.5, 0, 0.5], [0, 0.5, 0, 1], [0, 0.25, 0, 0.25]])
    def test_transient_total_is_independent_of_fov_size(self, hist, fov):
        """A SPAD averages over its FOV, so a smaller FOV sees the same flat wall at the
        same brightness. Before the F3 fix this scaled with the pixel count."""
        masks = hist.get_perpixel_fov_masks(torch.zeros(H, W), [fov], vignette=False)
        irr, dep, off = _scene(3.0, irradiance=1.0)
        t, _ = hist.calculate_transients(irr, dep, off, masks, B, MAX_DEPTH)
        assert float(t.sum()) == pytest.approx(1.0)

    def test_multiple_fovs_produce_independent_rows(self, hist):
        masks = hist.get_perpixel_fov_masks(
            torch.zeros(H, W), [[0, 0.5, 0, 1], [0.5, 1, 0, 1]], vignette=False
        )
        irr, dep, off = _scene(1.0)
        dep[0, : H // 2, :] = 2.0
        dep[0, H // 2 :, :] = 6.0
        t, _ = hist.calculate_transients(irr, dep, off, masks, B, MAX_DEPTH)
        assert t.shape == (2, B)
        assert int(t[0].argmax()) == _expected_bin(2.0)
        assert int(t[1].argmax()) == _expected_bin(6.0)

    def test_vignette_weights_are_applied(self, hist):
        """Vignetting attenuates real signal, so it reduces the total. Weights sum to
        1.75 over the 3 non-zero pixels -> 1.75/3."""
        mask = torch.tensor([[[1.0, 0.5], [0.25, 0.0]]])
        irr, dep, off = _scene(3.0, h=2, w=2)
        t, _ = hist.calculate_transients(irr, dep, off, mask, B, MAX_DEPTH)
        assert float(t.sum()) == pytest.approx(1.75 / 3.0)

    def test_no_vignette_gives_unattenuated_total(self, hist):
        mask = torch.ones(1, 2, 2)
        irr, dep, off = _scene(3.0, h=2, w=2)
        t, _ = hist.calculate_transients(irr, dep, off, mask, B, MAX_DEPTH)
        assert float(t.sum()) == pytest.approx(1.0)

    def test_empty_fov_yields_zero_row(self, hist):
        irr, dep, off = _scene(3.0)
        t, amb = hist.calculate_transients(irr, dep, off, torch.zeros(1, H, W), B, MAX_DEPTH)
        assert t.shape == (1, B)
        assert float(t.sum()) == pytest.approx(0.0)
        assert float(amb[0]) == pytest.approx(0.0)


# =========================================================================== #
# 4d — out-of-range returns alias (mode-independent)                           #
# =========================================================================== #
class TestOutOfRangeAliasing:
    def test_beyond_max_depth_aliases_rather_than_clamping(self, hist):
        """A return from beyond max_depth arrives during a later laser cycle, so it
        folds back to (2d/c) mod (1/f). It must NOT pile into the last bin."""
        irr, dep, off = _scene(1.0, h=2, w=2)
        dep[0] = torch.tensor([[5.0, 50.0], [500.0, 5000.0]])
        t, _ = hist.calculate_transients(irr, dep, off, torch.ones(1, 2, 2), B, MAX_DEPTH)
        assert float(t[0, -1]) == pytest.approx(0.0)
        # 5m -> bin 10; 50/500/5000m all alias to bin 0
        assert float(t[0, 10]) == pytest.approx(0.25)
        assert float(t[0, 0]) == pytest.approx(0.75)

    @pytest.mark.parametrize("depth,expected", [(12.0, 4), (22.0, 4), (10.0, 0), (20.0, 0)])
    def test_aliasing_wraps_modulo_max_depth(self, hist, depth, expected):
        irr, dep, off = _scene(depth)
        t, _ = hist.calculate_transients(irr, dep, off, _full_mask(), B, MAX_DEPTH)
        assert int(t[0].argmax()) == expected

    def test_in_range_surface_survives_out_of_range_neighbours(self, hist):
        irr, dep, off = _scene(1.0, h=2, w=2)
        dep[0] = torch.tensor([[5.0, 50.0], [500.0, 5000.0]])
        t, _ = hist.calculate_transients(irr, dep, off, torch.ones(1, 2, 2), B, MAX_DEPTH)
        assert float(t[0, _expected_bin(5.0)]) == pytest.approx(0.25)


# =========================================================================== #
# 4e — invalid depths are rejected, not epsilon-guarded                        #
# =========================================================================== #
class TestInvalidDepths:
    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_depth_contributes_nothing(self, hist, bad):
        irr, dep, off = _scene(1.0, h=2, w=2)
        dep[0] = torch.tensor([[3.0, bad], [bad, bad]])
        t, _ = hist.calculate_transients(irr, dep, off, torch.ones(1, 2, 2), B, MAX_DEPTH)
        assert torch.isfinite(t).all()
        assert torch.nonzero(t[0]).flatten().tolist() == [_expected_bin(3.0)]
        assert float(t.sum()) == pytest.approx(0.25)  # 1 of 4 pixels valid

    def test_all_invalid_gives_zero_row_without_nan(self, hist):
        irr, dep, off = _scene(0.0)  # depth 0 everywhere == "no surface"
        t, _ = hist.calculate_transients(irr, dep, off, _full_mask(), B, MAX_DEPTH)
        assert torch.isfinite(t).all()
        assert float(t.sum()) == pytest.approx(0.0)


# =========================================================================== #
# 4f — frame semantics (characterisation only; T2 undecided)                   #
# =========================================================================== #
class TestFrameSemantics:
    def test_multiframe_currently_stacks_rather_than_accumulates(self, hist):
        """CHARACTERISATION, not an endorsement. N frames x M masks yields N*M rows.
        Whether frames *should* stack (moving scene) or accumulate (single exposure)
        is an open decision — T2. This test exists so it cannot change by accident."""
        irr, dep, off = _scene(3.0, n_frames=3)
        t, amb = hist.calculate_transients(irr, dep, off, torch.ones(2, H, W), B, MAX_DEPTH)
        assert t.shape == (6, B)
        assert len(amb) == 6

    def test_single_frame_shape_is_one_row_per_fov(self, hist):
        irr, dep, off = _scene(3.0, n_frames=1)
        t, amb = hist.calculate_transients(irr, dep, off, torch.ones(3, H, W), B, MAX_DEPTH)
        assert t.shape == (3, B)
        assert len(amb) == 3


# =========================================================================== #
# 4g — ambient offsets                                                         #
# =========================================================================== #
class TestAmbientOffsets:
    def test_offset_is_averaged_over_fov_and_spread_over_bins(self, hist):
        irr, dep, _ = _scene(3.0)
        off = torch.full((1, H, W), 2.0)
        _, amb = hist.calculate_transients(irr, dep, off, _full_mask(), B, MAX_DEPTH)
        assert float(amb[0]) == pytest.approx(2.0 / B)

    def test_offset_is_independent_of_fov_size(self, hist):
        """Same F3 reasoning as the signal path: ambient must not scale with how many
        render pixels happen to fall inside the FOV."""
        irr, dep, _ = _scene(3.0)
        off = torch.full((1, H, W), 2.0)
        totals = []
        for fov in [[0, 1, 0, 1], [0, 0.5, 0, 0.5]]:
            masks = hist.get_perpixel_fov_masks(torch.zeros(H, W), [fov], vignette=False)
            _, amb = hist.calculate_transients(irr, dep, off, masks, B, MAX_DEPTH)
            totals.append(float(amb[0]))
        assert totals[1] == pytest.approx(totals[0])

    def test_zero_ambient_gives_zero_offset(self, hist):
        irr, dep, off = _scene(3.0)
        _, amb = hist.calculate_transients(irr, dep, off, _full_mask(), B, MAX_DEPTH)
        assert float(amb[0]) == pytest.approx(0.0)


# =========================================================================== #
# 4h — cross-layer invariants                                                  #
# =========================================================================== #
class TestBinWidthConvention:
    def test_kernel_bin_width_matches_transient_bin_width(self):
        """Pinning test for A4. `get_kernel` uses 2*tof2depth(tau) (round-trip) while
        camera.py passes bin_width = 2*tof2depth(1/f)/n_bins; the transient bins one-way
        depth over max_resolvable_depth. The two factors of 2 cancel *exactly*, so a
        tau-long pulse spans tau*f*n_bins bins by both routes. Correct but fragile — this
        fails if either convention is 'fixed' in isolation."""
        f = 10e6 * ureg.hertz
        n_bins = 1000
        for tau_ns in [1.0, 2.0, 5.0]:
            tau = tau_ns * ureg.nanosecond
            bin_width = 2 * tof2depth(1 / f) / n_bins
            kernel_bins = float((2 * tof2depth(tau) / bin_width).to(ureg.dimensionless).magnitude)
            transient_bins = float((tau * f).to(ureg.dimensionless).magnitude) * n_bins
            assert kernel_bins == pytest.approx(transient_bins, rel=1e-9)


class TestKernelNormalisation:
    @pytest.mark.parametrize("shape", ["gaussian", "square"])
    @pytest.mark.parametrize("tau_ns", [1.0, 2.0, 5.0])
    def test_pipeline_kernel_conserves_energy(self, shape, tau_ns):
        """The convolution must conserve photons, so the IRF has to sum to 1. The
        pipeline previously passed normalize=None, leaving the gaussian as a density in
        1/m (sum ~ 1/bin_width ~ 33) and the square as a plateau (sum ~ pulse_bins),
        inflating every arrival rate by a configuration-dependent factor."""
        from visionsim.emulate.aspc.sources import DynamicSource

        f = 10e6 * ureg.hertz
        bin_width = 2 * tof2depth(1 / f) / 1000
        ds = DynamicSource(pulse_width=tau_ns * ureg.nanosecond, pulse_shape=shape)
        _, kernel = ds.get_kernel(bin_width, "sum")
        assert float(np.sum(kernel)) == pytest.approx(1.0, rel=1e-6)


class TestConvolutionOrientation:
    def test_asymmetric_irf_matches_true_convolution(self, hist):
        """F.conv1d is cross-correlation; without a kernel flip an asymmetric IRF comes
        out time-reversed, biasing the recovered depth. Checked against numpy's
        convolution rather than hand-derived indices."""
        irf_np = np.array([0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        transient = torch.zeros(1, 2 * B)
        transient[0, 10] = 1.0

        out = hist.calculate_arrival_rates(torch.tensor(irf_np), transient, 0.0, 2 * B)
        expected = np.convolve(transient[0].numpy(), irf_np, mode="same")

        assert np.allclose(out[0].numpy(), expected, atol=1e-6)
        # Sanity: the mirrored (cross-correlation) result is genuinely different, so
        # this test would have failed before the flip.
        assert not np.allclose(expected, np.convolve(transient[0].numpy(), irf_np[::-1], mode="same"))

    def test_asymmetric_irf_preserves_photon_count(self, hist):
        irf_np = np.array([0.0, 0.2, 0.7, 0.1, 0.0], dtype=np.float32)
        transient = torch.zeros(1, 2 * B)
        transient[0, 10] = 1.0
        out = hist.calculate_arrival_rates(torch.tensor(irf_np), transient, 0.0, 2 * B)
        assert float(out.sum()) == pytest.approx(float(irf_np.sum()), rel=1e-5)

    def test_symmetric_irf_is_unaffected_by_the_flip(self, hist):
        irf = torch.tensor([0.25, 0.5, 0.25])
        transient = torch.zeros(1, 2 * B)
        transient[0, 10] = 1.0
        out = hist.calculate_arrival_rates(irf, transient, 0.0, 2 * B)
        assert int(out[0].argmax()) == 10
        assert float(out[0, 9]) == pytest.approx(float(out[0, 11]))


class TestResolutionDecoupling:
    def test_per_pixel_irradiance_is_resolution_independent(self):
        """Half of F3: radiance divides by num_pixels*omega = total FOV solid angle,
        which does not depend on the render grid."""
        from visionsim.emulate.aspc.sensors import SPADSensor
        from visionsim.emulate.aspc.sources import PulsedLaser
        from visionsim.emulate.aspc.utils import irradiance_photons

        laser = PulsedLaser(
            wavelength=550 * ureg.nanometer,
            frequency=10e6 * ureg.hertz,
            pulse_width=1 * ureg.nanosecond,
            avg_watts=1 * ureg.milliwatt,
            pulse_shape="gaussian",
        )
        vals = []
        for h, w in [(75, 100), (150, 200), (300, 400)]:
            s = SPADSensor(size=(h, w), fov=66 * ureg.degree)
            rad = laser.get_scene_radiance(torch.tensor(0.5), 3.0 * ureg.meter, s.w * s.h, s.omega)
            irr = (rad * np.pi / 4 * (1 / s.f_number) ** 2).to(irradiance_photons) * (
                s.pixel_pitch.to(ureg.meter)
            ) ** 2
            vals.append(float(irr.magnitude))
        assert vals[1] == pytest.approx(vals[0], rel=1e-9)
        assert vals[2] == pytest.approx(vals[0], rel=1e-9)

    @pytest.mark.parametrize("grid", [(8, 8), (16, 16), (64, 64), (75, 100)])
    def test_transient_is_independent_of_render_resolution(self, hist, grid):
        """The other half of F3: averaging over the FOV means a finer render grid
        samples the same wall more densely without collecting more light."""
        h, w = grid
        irr, dep, off = _scene(3.0, irradiance=1.0, h=h, w=w)
        t, _ = hist.calculate_transients(irr, dep, off, torch.ones(1, h, w), B, MAX_DEPTH)
        # rel=1e-4 absorbs float32 accumulation error over thousands of pixels.
        assert float(t.sum()) == pytest.approx(1.0, rel=1e-4)
        assert int(t[0].argmax()) == _expected_bin(3.0)
