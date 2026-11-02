"""Shared thermal model values and physical constants."""

import math

AMBIENT_TEMPERATURE_K = 295.0
STEFAN_BOLTZMANN_SI = 5.670374419e-8  # W/(m² K⁴)
STEFAN_BOLTZMANN_MM = STEFAN_BOLTZMANN_SI / 1e6
CYCLES_LOUT_TO_IRRADIANCE = math.pi
