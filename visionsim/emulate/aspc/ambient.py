import math
from enum import Enum
from typing import Tuple, Union

import numpy as np
from pint import Quantity
from scipy.constants import c, h, k, sigma
from scipy.integrate import trapezoid

from visionsim.emulate.aspc.utils import (
    radiance_photons,
    ureg,
    validate_units,
    watts2photons,
)


class LightConditions(Enum):
    BRIGHTEST_SUNLIGHT = 120000
    BRIGHT_SUNLIGHT = 111000
    AVERAGE_SUNLIGHT = 109870
    BRIGHT_SHADE = 20000
    OVERCAST = 2500
    SUNSET = 400
    SUNRISE = 400
    STORM_OVERCAST = 200
    OVERCAST_SUNSET = 40
    OVERCAST_SUNRISE = 40
    FULL_MOON = 0.25
    QUARTER_MOON = 0.01
    STARLIGHT_WITH_AIRGLOW = 0.002
    STARLIGHT_WITHOUT_AIRGLOW = 0.0002

    @property
    def value(self) -> Quantity:
        return super().value * ureg.lux


def get_light_conditions_from_string(condition_str: str) -> LightConditions:
    return getattr(LightConditions, condition_str)


class BlackBodySource:
    """Planck blackbody radiation source."""

    @validate_units()
    def __init__(self, temperature: Quantity = 5778 * ureg.kelvin):
        self.temperature = temperature

    def lambda_max(self) -> Quantity:
        b = 2.897771955e-3 * ureg.meter * ureg.kelvin
        return b / self.temperature

    def radiance_per_wavelength(self, wavelength: Quantity) -> Quantity:
        lam = wavelength.to(ureg.meter)
        numerator = 2 * h * c**2 / (lam**5)
        exp_arg = (h * c / (lam * k * self.temperature)).magnitude
        denominator = (np.exp(exp_arg) if hasattr(exp_arg, "__len__") else math.exp(exp_arg)) - 1
        return numerator / denominator

    def total_radiance(self) -> Quantity:
        sigma_q = sigma * ureg.watt / ureg.meter**2 / ureg.kelvin**4
        return (sigma_q * self.temperature**4) / (np.pi * ureg.steradian)


class Sun(BlackBodySource):
    """Ambient blackbody source (Sun/Moon/Sky) for background noise simulation."""

    @validate_units()
    def __init__(
        self,
        *,
        intensity=3.828e26 * ureg.watt / ureg.meter**2,
        stability_factor=0.01 * ureg.dimensionless,
        temperature=5778 * ureg.kelvin,
        lambda_pass=500e-9 * ureg.meter,
        delta_lambda=10e-9 * ureg.meter,
        light_conditions: Union[LightConditions, str] = LightConditions.BRIGHT_SUNLIGHT,
        emissivity: float = 1.0,
    ):
        super().__init__(temperature=temperature)
        self.intensity = intensity
        self.stability_factor = stability_factor
        self.lambda_pass = lambda_pass
        self.delta_lambda = delta_lambda
        self.emissivity = emissivity

        if isinstance(light_conditions, str):
            self.light_conditions = get_light_conditions_from_string(light_conditions)
        else:
            self.light_conditions = light_conditions

        self.c_eff = self._calculate_filter_efficiency()

    @property
    def lux(self) -> Quantity:
        return self.light_conditions.value if isinstance(self.light_conditions, LightConditions) else self.light_conditions

    def _calculate_filter_efficiency(self) -> Quantity:
        filter_lam = np.array([
            (self.lambda_pass - self.delta_lambda).to(ureg.meter).magnitude,
            self.lambda_pass.to(ureg.meter).magnitude,
            (self.lambda_pass + self.delta_lambda).to(ureg.meter).magnitude,
        ]) * ureg.meters

        radiance_vals = self.radiance_per_wavelength(filter_lam)
        passes_through = trapezoid(radiance_vals.magnitude, x=filter_lam.magnitude) * radiance_vals.units
        eff = passes_through / self.total_radiance()
        return eff.magnitude * ureg.dimensionless

    @ureg.check(None, None, ureg.hertz)
    def get_scene_radiance(self, rho_hat, pulse_repetition: Quantity):
        watts_eff_per_area = 0.0079 * self.lux.to(ureg.lux).magnitude * self.c_eff * ureg.watt
        photons_eff = (
            watts2photons(watts_eff_per_area, 1 / pulse_repetition, self.lambda_pass) / ureg.meter**2
        )
        radiance = (photons_eff.to(ureg.count / ureg.meter**2) * rho_hat / np.pi) / ureg.steradian
        return radiance.to(radiance_photons)

    @property
    def params(self) -> Tuple:
        return (
            self.intensity,
            self.stability_factor,
            self.temperature,
            self.lambda_pass,
            self.delta_lambda,
            self.light_conditions,
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Sun) and self.params == other.params

    def __hash__(self) -> int:
        return hash(self.params)

    def __repr__(self) -> str:
        cond_name = self.light_conditions.name if isinstance(self.light_conditions, LightConditions) else str(self.light_conditions)
        return (
            f"Sun(temperature={self.temperature:.0f~P}, "
            f"passband={self.lambda_pass.to(ureg.nanometer):.1f~P}, "
            f"light_conditions={cond_name})"
        )