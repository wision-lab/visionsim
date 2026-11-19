from __future__ import annotations

from typing import Any

import numpy as np
import torch
from pint import Quantity

from visionsim.emulate.aspc.utils import (
    radiance_photons,
    tof2depth,
    ureg,
    validate_units,
    watts2photons,
)


class DynamicSource:
    """Base class for pulsed light sources."""

    @validate_units()
    def __init__(
        self,
        pulse_width: Quantity = 1.0 * ureg.nanosecond,
        pulse_shape: str = "gaussian",
        pulse_shape_custom: Any = None,
    ):
        self.pulse_width = pulse_width
        self.pulse_shape = pulse_shape
        self.pulse_shape_custom = pulse_shape_custom

    def get_kernel(self, bin_width: Quantity, normalize: str = "sum") -> tuple[np.ndarray, np.ndarray]:
        if self.pulse_width.check(ureg.meter.dimensionality):
            pw_m = self.pulse_width.to(ureg.meter).magnitude
        else:
            pw_m = tof2depth(self.pulse_width).to(ureg.meter).magnitude

        if bin_width.check(ureg.meter.dimensionality):
            bw_m = bin_width.to(ureg.meter).magnitude
        else:
            bw_m = tof2depth(bin_width).to(ureg.meter).magnitude

        num_bins = max(1, round(pw_m / bw_m))
        if self.pulse_shape == "gaussian":
            half_num = max(3 * num_bins, 5)
            x = np.arange(-half_num, half_num + 1) * bw_m
            sigma = pw_m / 2.35482
            kernel = np.exp(-0.5 * (x / sigma) ** 2)
        elif self.pulse_shape == "square":
            half_num = num_bins
            x = np.arange(-half_num, half_num + 1) * bw_m
            kernel = np.where(np.abs(x) <= pw_m / 2, 1.0, 0.0)
        elif self.pulse_shape == "custom" and callable(self.pulse_shape_custom):
            half_num = max(3 * num_bins, 5)
            x = np.arange(-half_num, half_num + 1) * bw_m
            kernel = self.pulse_shape_custom(x / pw_m)
        else:
            x = np.array([0.0])
            kernel = np.array([1.0])

        if normalize == "sum":
            s: float = float(np.sum(kernel))
            if s > 0:
                kernel = kernel / s
        elif normalize == "max":
            m: float = float(np.max(kernel))
            if m > 0:
                kernel = kernel / m

        return x, kernel


class PulsedLaser(DynamicSource):
    """Active illumination pulsed laser source."""

    @validate_units()
    def __init__(
        self,
        wavelength: Quantity = 940 * ureg.nanometer,
        pulse_repetition: Quantity = 10 * ureg.megahertz,
        pulse_width: Quantity = 6 * ureg.nanosecond,
        avg_watts: Quantity = 1e-7 * ureg.watt,
        pulse_shape: str = "gaussian",
        pulse_shape_custom: Any = None,
    ):
        super().__init__(
            pulse_width=pulse_width,
            pulse_shape=pulse_shape,
            pulse_shape_custom=pulse_shape_custom,
        )
        self.wavelength = wavelength
        self.pulse_repetition = pulse_repetition
        self.avg_watts = avg_watts

    @property
    def max_resolvable_depth(self) -> Quantity:
        return tof2depth(1 / self.pulse_repetition)

    @property
    def num_photons_per_cycle(self) -> Quantity:
        return watts2photons(self.avg_watts, 1 / self.pulse_repetition, self.wavelength)

    def get_scene_radiance(
        self,
        albedo: torch.Tensor | float,
        depth: torch.Tensor | Quantity | float,
        num_pixels: int,
        omega: Quantity,
    ) -> Quantity:
        n_photons = self.num_photons_per_cycle.magnitude
        depth_m = depth.magnitude if hasattr(depth, "magnitude") else depth
        albedo_val = albedo.magnitude if hasattr(albedo, "magnitude") else albedo

        # Total emitter solid angle = num_pixels * omega
        omega_val = omega.magnitude if hasattr(omega, "magnitude") else omega
        total_omega = omega_val * num_pixels

        # Radiance is total photons divided by total illuminated solid angle and area
        rad_val = (n_photons * albedo_val) / (np.pi * (depth_m**2) * total_omega)
        return rad_val * radiance_photons
