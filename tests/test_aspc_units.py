"""Stage 1 — Pure physics / unit-conversion functions in ``visionsim.emulate.aspc.utils``.

These are the lowest-level building blocks (no torch tensors, no config, no data
loader). Each test has a closed-form oracle so a failure localizes to one formula.

Known-bug findings from the audit are encoded as ``xfail`` so they are documented
and tracked without breaking the suite. When a bug is fixed the corresponding test
flips to ``xpass`` and pytest flags it.
"""

import numpy as np
import pytest
from scipy.constants import c as C_SCIPY
from scipy.constants import h as H_SCIPY

from visionsim.emulate.aspc import utils as aspc_utils
from visionsim.emulate.aspc.utils import (
    focal_length_from_fov,
    fov_from_focal_length,
    pyramid_solid_angle,
    tof2depth,
    ureg,
)
from visionsim.emulate.aspc.utils import watts2photons

REL = 1e-9


# --------------------------------------------------------------------------- #
# tof2depth: depth = c * t / 2  (one-way distance from round-trip time)        #
# --------------------------------------------------------------------------- #
class TestTof2Depth:
    def test_returns_length(self):
        d = tof2depth(1e-6 * ureg.second)
        assert d.check("[length]")

    def test_matches_formula_with_module_constant(self):
        # Validates the *formula* against whatever speed of light the module uses.
        t = 1e-6 * ureg.second
        expected = aspc_utils.c * t.to(ureg.second).magnitude / 2.0  # meters
        assert tof2depth(t).to(ureg.meter).magnitude == pytest.approx(expected, rel=REL)

    def test_is_linear_in_time(self):
        d1 = tof2depth(1e-6 * ureg.second).to(ureg.meter).magnitude
        d2 = tof2depth(2e-6 * ureg.second).to(ureg.meter).magnitude
        assert d2 == pytest.approx(2.0 * d1, rel=REL)

    def test_zero_time_is_zero_depth(self):
        assert tof2depth(0.0 * ureg.second).to(ureg.meter).magnitude == pytest.approx(0.0)

    def test_uses_physical_speed_of_light(self):
        # Finding A2 (fixed): utils.c now imports scipy.constants.c.
        t = 1e-6 * ureg.second
        expected = C_SCIPY * 1e-6 / 2.0  # physically correct one-way depth
        assert tof2depth(t).to(ureg.meter).magnitude == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------- #
# watts2photons: n = (watts * t) / (h * c / lambda)                            #
# --------------------------------------------------------------------------- #
class TestWatts2Photons:
    def test_returns_count(self):
        n = watts2photons(1.0 * ureg.watt, 1.0 * ureg.second, 550 * ureg.nanometer)
        assert n.check(ureg.count.dimensionality)

    def test_matches_formula(self):
        watts, t, lam = 1.0 * ureg.watt, 1.0 * ureg.second, 550 * ureg.nanometer
        photon_energy = H_SCIPY * aspc_utils.c / lam.to(ureg.meter).magnitude
        expected = (1.0 * 1.0) / photon_energy
        got = watts2photons(watts, t, lam).to(ureg.count).magnitude
        assert got == pytest.approx(expected, rel=REL)

    def test_linear_in_power_and_time(self):
        base = watts2photons(1.0 * ureg.watt, 1.0 * ureg.second, 550 * ureg.nanometer).magnitude
        assert watts2photons(2.0 * ureg.watt, 1.0 * ureg.second, 550 * ureg.nanometer).magnitude == pytest.approx(
            2.0 * base, rel=REL
        )
        assert watts2photons(1.0 * ureg.watt, 3.0 * ureg.second, 550 * ureg.nanometer).magnitude == pytest.approx(
            3.0 * base, rel=REL
        )

    def test_shorter_wavelength_fewer_photons(self):
        # Higher-energy (shorter wavelength) photons => fewer of them for same energy.
        blue = watts2photons(1.0 * ureg.watt, 1.0 * ureg.second, 400 * ureg.nanometer).magnitude
        red = watts2photons(1.0 * ureg.watt, 1.0 * ureg.second, 700 * ureg.nanometer).magnitude
        assert blue < red

    def test_zero_power_is_zero_photons(self):
        assert watts2photons(0.0 * ureg.watt, 1.0 * ureg.second, 550 * ureg.nanometer).magnitude == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# pyramid_solid_angle: Omega = 4 * arcsin(sin(a/2) * sin(b/2))                 #
# --------------------------------------------------------------------------- #
class TestPyramidSolidAngle:
    def test_returns_steradian(self):
        omega = pyramid_solid_angle(66 * ureg.degree, 44 * ureg.degree)
        assert omega.check(ureg.steradian.dimensionality)

    def test_matches_closed_form(self):
        a, b = 66 * ureg.degree, 44 * ureg.degree
        ar, br = np.deg2rad(66), np.deg2rad(44)
        expected = 4.0 * np.arcsin(np.sin(ar / 2) * np.sin(br / 2))
        assert pyramid_solid_angle(a, b).to(ureg.steradian).magnitude == pytest.approx(expected, rel=REL)

    def test_symmetric_in_arguments(self):
        x = pyramid_solid_angle(66 * ureg.degree, 44 * ureg.degree).magnitude
        y = pyramid_solid_angle(44 * ureg.degree, 66 * ureg.degree).magnitude
        assert x == pytest.approx(y, rel=REL)

    def test_small_angle_limit_approaches_product(self):
        # For small apex angles, Omega ~ a * b (in radians).
        a = b = 1.0 * ureg.degree
        ar = np.deg2rad(1.0)
        got = pyramid_solid_angle(a, b).to(ureg.steradian).magnitude
        assert got == pytest.approx(ar * ar, rel=1e-3)

    def test_monotonic_in_angle(self):
        small = pyramid_solid_angle(10 * ureg.degree, 10 * ureg.degree).magnitude
        large = pyramid_solid_angle(40 * ureg.degree, 40 * ureg.degree).magnitude
        assert large > small


# --------------------------------------------------------------------------- #
# fov <-> focal length are inverses                                           #
# --------------------------------------------------------------------------- #
class TestFovFocalLength:
    def test_fov_returns_angle(self):
        fov = fov_from_focal_length(50 * ureg.millimeter, 36 * ureg.millimeter)
        assert fov.check("[]") or fov.check(ureg.radian.dimensionality)

    def test_focal_returns_length(self):
        fl = focal_length_from_fov(40 * ureg.degree, 36 * ureg.millimeter)
        assert fl.check("[length]")

    def test_focal_matches_closed_form(self):
        fov, d = 40 * ureg.degree, 36 * ureg.millimeter
        expected_m = d.to(ureg.meter).magnitude / (2.0 * np.tan(np.deg2rad(40) / 2.0))
        assert focal_length_from_fov(fov, d).to(ureg.meter).magnitude == pytest.approx(expected_m, rel=REL)

    @pytest.mark.parametrize("fov_deg", [20.0, 44.0, 66.0, 90.0])
    def test_round_trip_fov_to_focal_to_fov(self, fov_deg):
        d = 36 * ureg.millimeter
        fl = focal_length_from_fov(fov_deg * ureg.degree, d)
        back = fov_from_focal_length(fl, d).to(ureg.degree).magnitude
        assert back == pytest.approx(fov_deg, rel=1e-9)

    def test_longer_focal_narrower_fov(self):
        d = 36 * ureg.millimeter
        wide = fov_from_focal_length(20 * ureg.millimeter, d).to(ureg.degree).magnitude
        tele = fov_from_focal_length(200 * ureg.millimeter, d).to(ureg.degree).magnitude
        assert tele < wide
