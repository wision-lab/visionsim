"""Stage 4 — The φ pipeline: ``calculate_transients`` (flat-wall oracle)."""

import numpy as np
import pytest
import torch

from visionsim.emulate.aspc.histogrammers import HistConfig, Histogrammer
from visionsim.emulate.aspc.utils import tof2depth, ureg

B = 20
MAX_DEPTH = 10.0
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
        dep[0, :4, :] = 2.0
        dep[0, 4:, :] = 6.0
        t, _ = hist.calculate_transients(irr, dep, off, _full_mask(), B, MAX_DEPTH)
        near, far = _expected_bin(2.0), _expected_bin(6.0)
        assert torch.nonzero(t[0]).flatten().tolist() == sorted([near, far])
        assert float(t[0, near]) == pytest.approx(0.5)
        assert float(t[0, far]) == pytest.approx(0.5)

    def test_tilted_plane_spreads_across_bins(self, hist):
        irr, dep, off = _scene(1.0)
        dep[0] = torch.linspace(0.5, 9.5, H).view(H, 1).expand(H, W)
        t, _ = hist.calculate_transients(irr, dep, off, _full_mask(), B, MAX_DEPTH)
        assert int(torch.count_nonzero(t[0])) == H
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
        full = HistConfig().pixel_fov_list[-1]
        assert full == [0, 1.0, 0, 1.0]
        for shape in [(300, 400), (1000, 1000)]:
            m = hist.get_pixel_fov_mask(torch.zeros(*shape), *full, vignette=False)
            assert int(m.sum()) == shape[0] * shape[1]

    @pytest.mark.parametrize("fov", [[0, 1, 0, 1], [0, 0.5, 0, 0.5], [0, 0.5, 0, 1], [0, 0.25, 0, 0.25]])
    def test_transient_total_is_independent_of_fov_size(self, hist, fov):
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
# 4d — out-of-range returns alias                                              #
# =========================================================================== #
class TestOutOfRangeAliasing:
    def test_beyond_max_depth_aliases_rather_than_clamping(self, hist):
        irr, dep, off = _scene(1.0, h=2, w=2)
        dep[0] = torch.tensor([[5.0, 50.0], [500.0, 5000.0]])
        t, _ = hist.calculate_transients(irr, dep, off, torch.ones(1, 2, 2), B, MAX_DEPTH)
        assert float(t[0, -1]) == pytest.approx(0.0)
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
# 4e — invalid depths are rejected                                            #
# =========================================================================== #
class TestInvalidDepths:
    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_depth_contributes_nothing(self, hist, bad):
        irr, dep, off = _scene(1.0, h=2, w=2)
        dep[0] = torch.tensor([[3.0, bad], [bad, bad]])
        t, _ = hist.calculate_transients(irr, dep, off, torch.ones(1, 2, 2), B, MAX_DEPTH)
        assert torch.isfinite(t).all()
        assert torch.nonzero(t[0]).flatten().tolist() == [_expected_bin(3.0)]
        assert float(t.sum()) == pytest.approx(0.25)

    def test_all_invalid_gives_zero_row_without_nan(self, hist):
        irr, dep, off = _scene(0.0)
        t, _ = hist.calculate_transients(irr, dep, off, _full_mask(), B, MAX_DEPTH)
        assert torch.isfinite(t).all()
        assert float(t.sum()) == pytest.approx(0.0)


# =========================================================================== #
# 4f — frame semantics                                                         #
# =========================================================================== #
class TestFrameSemantics:
    def test_multiframe_currently_stacks_rather_than_accumulates(self, hist):
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
        from visionsim.emulate.aspc.emitter import DynamicSource

        f = 10e6 * ureg.hertz
        bin_width = 2 * tof2depth(1 / f) / 1000
        ds = DynamicSource(pulse_width=tau_ns * ureg.nanosecond, pulse_shape=shape)
        _, kernel = ds.get_kernel(bin_width, "sum")
        assert float(np.sum(kernel)) == pytest.approx(1.0, rel=1e-6)


class TestConvolutionOrientation:
    def test_asymmetric_irf_matches_true_convolution(self, hist):
        irf_np = np.array([0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        transient = torch.zeros(1, 2 * B)
        transient[0, 10] = 1.0

        out = hist.calculate_arrival_rates(torch.tensor(irf_np), transient, 0.0, 2 * B)
        expected = np.convolve(transient[0].numpy(), irf_np, mode="same")

        assert np.allclose(out[0].numpy(), expected, atol=1e-6)
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
        from visionsim.emulate.aspc.emitter import PulsedLaser
        from visionsim.emulate.aspc.sensors import SPADSensor
        from visionsim.emulate.aspc.utils import irradiance_photons

        laser = PulsedLaser(
            wavelength=550 * ureg.nanometer,
            pulse_repetition=10e6 * ureg.hertz,
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
        h, w = grid
        irr, dep, off = _scene(3.0, irradiance=1.0, h=h, w=w)
        t, _ = hist.calculate_transients(irr, dep, off, torch.ones(1, h, w), B, MAX_DEPTH)
        assert float(t.sum()) == pytest.approx(1.0, rel=1e-4)
        assert int(t[0].argmax()) == _expected_bin(3.0)