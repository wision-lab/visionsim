"""Stage 2a/2b — Active light source: pulse kernels and scene radiance.

2a: ``DynamicSource.get_kernel`` — the laser's temporal pulse (IRF). The whole
    arrival-rate convolution assumes a sum-normalized (energy-conserving) kernel.
2b: ``PulsedLaser.get_scene_radiance`` — the active-signal reference path
    (correct units, inverse-square falloff, linearity). Stage 2c will contrast
    this with the ambient ``Sun`` path (finding A1).
"""

import numpy as np
import pytest

from visionsim.emulate.aspc.sources import DynamicSource, PulsedLaser
from visionsim.emulate.aspc.utils import radiance_photons, tof2depth, ureg, watts2photons

REL = 1e-9


def _laser(**kw):
    defaults = dict(
        wavelength=550 * ureg.nanometer,
        frequency=10e6 * ureg.hertz,
        pulse_width=1 * ureg.nanosecond,
        avg_watts=1 * ureg.milliwatt,
        pulse_shape="gaussian",
    )
    defaults.update(kw)
    return PulsedLaser(**defaults)


# =========================================================================== #
# Stage 2a — pulse kernels                                                     #
# =========================================================================== #
class TestPulseKernelNormalization:
    @pytest.mark.parametrize("shape", ["gaussian", "square"])
    def test_sum_normalized_integrates_to_one(self, shape):
        ds = DynamicSource(pulse_width=1 * ureg.nanosecond, pulse_shape=shape)
        _, kernel = ds.get_kernel(bin_width=1 * ureg.centimeter, normalize="sum")
        assert float(np.sum(kernel)) == pytest.approx(1.0, rel=1e-6)

    def test_custom_callable_sum_normalized_to_one(self):
        ds = DynamicSource(
            pulse_width=1 * ureg.nanosecond,
            pulse_shape="custom",
            pulse_shape_custom=lambda z: np.exp(-(z**2)),
        )
        _, kernel = ds.get_kernel(bin_width=1 * ureg.centimeter, normalize="sum")
        assert float(np.sum(kernel)) == pytest.approx(1.0, rel=1e-6)

    @pytest.mark.parametrize("shape", ["gaussian", "square"])
    def test_max_normalized_peaks_at_one(self, shape):
        ds = DynamicSource(pulse_width=1 * ureg.nanosecond, pulse_shape=shape)
        _, kernel = ds.get_kernel(bin_width=1 * ureg.centimeter, normalize="max")
        assert float(np.max(kernel)) == pytest.approx(1.0, rel=1e-6)


class TestPulseKernelShape:
    @pytest.mark.parametrize("shape", ["gaussian", "square"])
    def test_kernel_is_non_negative(self, shape):
        ds = DynamicSource(pulse_width=1 * ureg.nanosecond, pulse_shape=shape)
        _, kernel = ds.get_kernel(bin_width=1 * ureg.centimeter, normalize="sum")
        assert float(np.min(kernel)) >= 0.0

    @pytest.mark.parametrize("shape", ["gaussian", "square"])
    def test_x_and_kernel_lengths_match(self, shape):
        ds = DynamicSource(pulse_width=1 * ureg.nanosecond, pulse_shape=shape)
        x, kernel = ds.get_kernel(bin_width=1 * ureg.centimeter, normalize="sum")
        assert len(x) == len(kernel)

    @pytest.mark.parametrize("shape", ["gaussian", "square"])
    def test_wider_pulse_gives_wider_kernel(self, shape):
        bw = 1 * ureg.centimeter
        _, narrow = DynamicSource(pulse_width=1 * ureg.nanosecond, pulse_shape=shape).get_kernel(bw, "sum")
        _, wide = DynamicSource(pulse_width=4 * ureg.nanosecond, pulse_shape=shape).get_kernel(bw, "sum")
        assert len(wide) > len(narrow)

    def test_square_kernel_has_flat_plateau(self):
        ds = DynamicSource(pulse_width=2 * ureg.nanosecond, pulse_shape="square")
        _, kernel = ds.get_kernel(bin_width=1 * ureg.centimeter, normalize="sum")
        nonzero = kernel[kernel > 0]
        assert nonzero.size > 0
        # all non-zero entries are equal (a box), so spread is ~0
        assert float(np.ptp(nonzero)) == pytest.approx(0.0, abs=1e-12)


# =========================================================================== #
# Stage 2b — active-source scene radiance (reference path)                     #
# =========================================================================== #
class TestPulsedLaserRadiance:
    def test_returns_photon_radiance_units(self):
        r = _laser().get_scene_radiance(0.5, 5.0 * ureg.meter, 1000, 1e-6 * ureg.steradian)
        # convertible to count / sr / m**2 without raising, and positive
        assert r.to(radiance_photons).magnitude > 0
        assert r.dimensionality == radiance_photons.dimensionality

    def test_inverse_square_falloff(self):
        laser = _laser()
        near = laser.get_scene_radiance(0.5, 5.0 * ureg.meter, 1000, 1e-6 * ureg.steradian).magnitude
        far = laser.get_scene_radiance(0.5, 10.0 * ureg.meter, 1000, 1e-6 * ureg.steradian).magnitude
        assert near / far == pytest.approx(4.0, rel=1e-4)

    def test_linear_in_albedo(self):
        laser = _laser()
        lo = laser.get_scene_radiance(0.5, 5.0 * ureg.meter, 1000, 1e-6 * ureg.steradian).magnitude
        hi = laser.get_scene_radiance(1.0, 5.0 * ureg.meter, 1000, 1e-6 * ureg.steradian).magnitude
        assert hi / lo == pytest.approx(2.0, rel=REL)

    def test_inverse_in_num_pixels(self):
        laser = _laser()
        few = laser.get_scene_radiance(0.5, 5.0 * ureg.meter, 1000, 1e-6 * ureg.steradian).magnitude
        many = laser.get_scene_radiance(0.5, 5.0 * ureg.meter, 2000, 1e-6 * ureg.steradian).magnitude
        assert few / many == pytest.approx(2.0, rel=REL)


class TestPulsedLaserDerivedQuantities:
    def test_max_resolvable_depth(self):
        f = 10e6 * ureg.hertz
        laser = _laser(frequency=f)
        expected = tof2depth(1 / f).to(ureg.meter).magnitude
        assert laser.max_resolvable_depth.to(ureg.meter).magnitude == pytest.approx(expected, rel=REL)

    def test_num_photons_per_cycle(self):
        f, lam, watts = 10e6 * ureg.hertz, 550 * ureg.nanometer, 1 * ureg.milliwatt
        laser = _laser(frequency=f, wavelength=lam, avg_watts=watts)
        expected = watts2photons(watts, 1 / f, lam).to(ureg.count).magnitude
        assert laser.num_photons_per_cycle.to(ureg.count).magnitude == pytest.approx(expected, rel=REL)

    def test_higher_frequency_shrinks_max_depth(self):
        slow = _laser(frequency=5e6 * ureg.hertz).max_resolvable_depth.to(ureg.meter).magnitude
        fast = _laser(frequency=20e6 * ureg.hertz).max_resolvable_depth.to(ureg.meter).magnitude
        assert fast < slow
