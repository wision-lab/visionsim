"""Stage 2c — Ambient source & black body: ``Sun`` and ``BlackBodySource``.

Contrasts the ambient path against the active reference path from Stage 2b.
Audit findings are encoded as strict ``xfail`` so they are tracked without
reddening the suite; a fix flips them to ``xpass``.

Findings exercised here:
  * A1 — ``Sun.get_scene_radiance`` carries a spurious ``omega`` factor. pint
    treats steradian as dimensionless so the units look right, but the ambient
    radiance wrongly scales with the camera's per-pixel solid angle. Physically,
    the sun's surface irradiance is intrinsic, so scene radiance must NOT depend
    on ``omega`` (whereas the active path legitimately does).
  * A6 — ``BlackBodySource.total_radiance`` uses ``scipy.constants.sigma`` as a
    bare float, so it returns ``kelvin**4`` instead of ``W/m**2/sr``.
  * B1 — ``Sun.params`` / ``repr(Sun(...))`` reference attributes that are never
    set (``stability_factor``) or convert bad units (``intensity.to(watt)``).
"""

import numpy as np
import pytest

from visionsim.emulate.aspc.sources import BlackBodySource, LightConditions, Sun
from visionsim.emulate.aspc.utils import radiance, radiance_photons, ureg

REL = 1e-9


def _sun(**kw):
    defaults = dict(
        temperature=5778 * ureg.kelvin,
        light_conditions=LightConditions.BRIGHT_SUNLIGHT,
        lambda_pass=550 * ureg.nanometer,
        delta_lambda=10 * ureg.nanometer,
    )
    defaults.update(kw)
    return Sun(**defaults)


# =========================================================================== #
# BlackBodySource physics                                                      #
# =========================================================================== #
class TestBlackBodyPhysics:
    def test_lambda_max_wien_closed_form(self):
        bb = BlackBodySource(temperature=5778 * ureg.kelvin)
        expected_m = 2.8978e-3 / 5778.0  # Wien constant / T
        assert bb.lambda_max().to(ureg.meter).magnitude == pytest.approx(expected_m, rel=1e-4)

    def test_hotter_body_peaks_at_shorter_wavelength(self):
        cool = BlackBodySource(temperature=3000 * ureg.kelvin).lambda_max().to(ureg.nanometer).magnitude
        hot = BlackBodySource(temperature=6000 * ureg.kelvin).lambda_max().to(ureg.nanometer).magnitude
        assert hot < cool

    def test_total_radiance_scales_as_t_to_the_fourth(self):
        bb1 = BlackBodySource(temperature=3000 * ureg.kelvin)
        bb2 = BlackBodySource(temperature=6000 * ureg.kelvin)
        ratio = (bb2.total_radiance() / bb1.total_radiance()).to(ureg.dimensionless).magnitude
        assert ratio == pytest.approx(16.0, rel=1e-6)

    def test_spectrum_peaks_near_lambda_max(self):
        # NOTE: wavelengths must be in METERS here — see A6b below; the pipeline
        # (Sun.__init__) also feeds meters, which is the only correct usage.
        bb = BlackBodySource(temperature=5778 * ureg.kelvin)
        lam = np.linspace(200e-9, 1500e-9, 400) * ureg.meter
        vals = bb.radiance_per_wavelength(lam)
        peak_nm = lam[int(np.argmax(vals.magnitude))].to(ureg.nanometer).magnitude
        assert peak_nm == pytest.approx(bb.lambda_max().to(ureg.nanometer).magnitude, rel=0.05)

    def test_radiance_per_wavelength_positive(self):
        bb = BlackBodySource(temperature=5778 * ureg.kelvin)
        assert float(bb.radiance_per_wavelength(500e-9 * ureg.meter).magnitude) > 0

    def test_radiance_per_wavelength_unit_invariant(self):
        # Finding A6b (fixed): wavelength is now converted to meters internally.
        bb = BlackBodySource(temperature=5778 * ureg.kelvin)
        in_nm = bb.radiance_per_wavelength(500 * ureg.nanometer).to_base_units().magnitude
        in_m = bb.radiance_per_wavelength(500e-9 * ureg.meter).to_base_units().magnitude
        assert float(in_nm) == pytest.approx(float(in_m), rel=1e-6)

    def test_total_radiance_has_radiance_units(self):
        # Finding A6 (fixed): sigma is now a Quantity in W/(m^2 K^4).
        bb = BlackBodySource(temperature=5778 * ureg.kelvin)
        assert bb.total_radiance().dimensionality == radiance.dimensionality


# =========================================================================== #
# Sun construction / scalar attributes (these pass today)                      #
# =========================================================================== #
class TestSunAttributes:
    def test_lux_from_light_condition_enum(self):
        assert _sun().lux.to(ureg.lux).magnitude == pytest.approx(
            LightConditions.BRIGHT_SUNLIGHT.value.to(ureg.lux).magnitude
        )

    def test_c_eff_is_dimensionless_fraction(self):
        c_eff = _sun().c_eff
        assert c_eff.check(ureg.dimensionless)
        assert 0.0 < c_eff.magnitude < 1.0


# =========================================================================== #
# Sun.get_scene_radiance — ambient reference behaviour + A1                    #
# =========================================================================== #
class TestSunSceneRadiance:
    def test_returns_photon_radiance_units(self):
        r = _sun().get_scene_radiance(1e-6 * ureg.steradian, 0.5, 10e6 * ureg.hertz)
        assert r.dimensionality == radiance_photons.dimensionality
        assert r.to(radiance_photons).magnitude > 0

    def test_linear_in_albedo(self):
        sun = _sun()
        lo = sun.get_scene_radiance(1e-6 * ureg.steradian, 0.5, 10e6 * ureg.hertz).magnitude
        hi = sun.get_scene_radiance(1e-6 * ureg.steradian, 1.0, 10e6 * ureg.hertz).magnitude
        assert hi / lo == pytest.approx(2.0, rel=REL)

    @pytest.mark.xfail(
        reason="Finding A1: ambient scene radiance wrongly scales with the "
        "camera's per-pixel omega (spurious factor). It should be independent "
        "of omega, since the sun's surface irradiance is intrinsic.",
        strict=True,
    )
    def test_radiance_independent_of_sensor_omega(self):
        sun = _sun()
        r1 = sun.get_scene_radiance(1e-6 * ureg.steradian, 0.5, 10e6 * ureg.hertz).magnitude
        r2 = sun.get_scene_radiance(2e-6 * ureg.steradian, 0.5, 10e6 * ureg.hertz).magnitude
        assert r1 == pytest.approx(r2, rel=1e-6)


# =========================================================================== #
# B1 — params / repr should not raise                                          #
# =========================================================================== #
class TestSunParamsRepr:
    def test_params_does_not_raise(self):
        # Finding B1 (fixed): __init__ now assigns stability_factor/light_conditions.
        params = _sun().params
        assert len(params) == 6

    def test_repr_does_not_raise(self):
        # Finding B1 (fixed): __repr__ now formats intensity as W/m**2.
        assert repr(_sun()).startswith("Sun(")
