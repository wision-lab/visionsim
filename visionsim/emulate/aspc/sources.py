import numpy as np
import math
import random
from typing import Tuple
from units import validate_units
from utils import ureg, Quantity, tof2depth, radiance_photons, watts2photons
from scipy.constants import Wien, c, h, k, sigma
import matplotlib.pyplot as plt
from enum import Enum
import time

# See: https://en.wikipedia.org/wiki/Daylight#Intensity_in_different_conditions
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
    def value(self):
        return super().value * ureg.lux

class LightSource:
    """
    Base class for all light sources
    """
    
    def __init__(self):
        self.is_source = True  # Indicates this is a light source

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        # Use array_equal as it deals with tensors/ndarrays/scalars
        return all(np.array_equal(i, j) for i, j in zip(self.params, other.params))

    def __hash__(self):
        return hash(self.params)

    @property
    def params(self):
        raise NotImplementedError("LightSource must be subclassed.")

"""
Constant and Dynamic Light Sources - temporal behavior
Coherent and Black Body Light Sources - spectral behavior

Pulsed Laser / Active Light source - combines dynamic and coherent properties
Ambient Light Source - combines constant and black body properties
Flickering Lamp - combines dynamic and black body properties
"""

class ConstantSource(LightSource):
    """
    Light sources with constant output; fluctuations based on stability factor
    """
    @validate_units()
    def __init__(self, intensity=ureg.watt/ureg.meter**2, stability_factor=0.01 * ureg.dimensionless):
        super().__init__()
        self.intensity = intensity                  # Base intensity in watts / meter^2
        self.stability_factor = stability_factor    # close to 0 for high stability (can vary due to temperature, voltage, etc.)
    
    def get_intensity(self) -> Quantity:
        fluctuation = 1 + random.uniform(-self.stability_factor, self.stability_factor)
        return self.intensity * fluctuation

    @property
    def params(self):
        return self.intensity, self.stability_factor

    def __repr__(self):
        return f"ConstantSource(intensity={self.intensity.to(ureg.watt/ureg.meter**2)}, stability_factor={self.stability_factor.to(ureg.dimensionless)})"


class DynamicSource(LightSource):
    """
    Light sources with time-varying output
    """
    @validate_units()
    def __init__(
        self, 
        modulation_frequency = 1.0 * ureg.hertz, 
        modulation_amplitude = 0.5 * ureg.dimensionless, 
        phase = 0.0 * ureg.radian,
        pulse_width = 1 * ureg.nanosecond,
        pulse_shape="gaussian"
        ):
        super().__init__()
        self.modulation_frequency = modulation_frequency    # Hz
        self.modulation_amplitude = modulation_amplitude    # Amplitude of variation (0-1)
        self.phase = phase                                  # Phase shift in radians
        self.pulse_width = pulse_width                      # Pulse width in seconds
        self.pulse_shape = pulse_shape                      # 'gaussian', 'square', callable, or array
   
    @validate_units()
    def get_kernel(self, bin_width=1 * ureg.centimeter, normalize="sum"):
        """Given histogram parameters, return a kernel representing the pulse
        Note: The kernel should have integral of 1.0 to conserve energy.
            This corresponds to the "sum" normalization.

        Args:
            bin_width: Width of each bin (in meters)
            normalize: Either "max", "sum" or None. Default: "sum"

        Returns:
            bins, kernel values
        """
        # Multiply by 2 as we aren't interested in the "two-way" distance of tof.
        pulse_width = 2.0 * tof2depth(self.pulse_width)
        pulse_bins = pulse_width / bin_width
        pulse_bins = np.ceil(pulse_bins.to(ureg.dimensionless).magnitude)

        if isinstance(self.pulse_shape, str):
            if self.pulse_shape == "gaussian":
                sigma = pulse_width / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                x = np.arange(-2 * pulse_bins, 2 * pulse_bins + 1) * bin_width
                kernel = np.exp(-(x**2) / (2.0 * sigma**2)) / (sigma.to(ureg.meter).magnitude * np.sqrt(2.0 * np.pi))
            elif self.pulse_shape == "square":
                x = np.arange(-pulse_bins, 2 * pulse_bins) * bin_width
                kernel = np.repeat([0, 1, 0], pulse_bins)
            else:
                raise ValueError(f"Unsupported pulse shape string: {self.pulse_shape}")
        elif callable(self.pulse_shape):
            # User has provided a function, e.g., lambda x: np.sinc(x / T)
            x = np.arange(-2 * pulse_bins, 2 * pulse_bins + 1) * bin_width
            kernel = self.pulse_shape(x)
        elif isinstance(self.pulse_shape, np.ndarray):
            # User has given the kernel directly
            kernel = self.pulse_shape
        else:
            raise TypeError("pulse_shape must be a string ('gaussian', 'square'), a callable, or a numpy array.")
    
        if normalize:
            if normalize.lower() in ("max", "sum"):
                kernel = kernel / getattr(kernel, normalize.lower())()
            else:
                raise ValueError(
                    "Unrecognized value for `normalize`. Expected either None, 'max' or "
                    f"'integral' but got {normalize}."
                )

        kernel = kernel.magnitude if isinstance(kernel, Quantity) else kernel
        return x, kernel
    
    def plot_kernel(self, bin_width, normalize):
        x, k = self.get_kernel(bin_width, normalize)
        plt.plot(x, k)
        plt.title(f"Pulse kernel")
        plt.xlabel("Depth (m)")
        plt.ylabel("Amplitude")
        plt.show()

    @property
    def params(self):
        return self.modulation_frequency, self.modulation_amplitude, self.phase, self.pulse_width, self.pulse_shape

    def __repr__(self):
        return f"DynamicSource(modulation_frequency={self.modulation_frequency.to(ureg.hertz)}, modulation_amplitude={self.modulation_amplitude.to(ureg.dimensionless)}, phase={self.phase.to(ureg.radian)}, pulse_width={self.pulse_width.to(ureg.nanosecond)}, pulse_shape={self.pulse_shape})"


class CoherentSource(LightSource):
    """A coherent source (most likely a laser) with it's
    peak power (in watts) and a single wavelength (in meters)
    """
    @validate_units()
    def __init__(
        self, 
        *,
        wavelength=550 * ureg.nanometer,
        avg_watts=ureg.watt,
        coherence_length = 1e-3 * ureg.meter, 
        beam_divergence = 0.001 * ureg.radian,
        polarization_angle = 0.0 * ureg.radian
        ):
        super().__init__()
        self.wavelength = wavelength
        self.avg_watts = avg_watts
        self.coherence_length = coherence_length  # Spatial coherence in meters
        self.beam_divergence = beam_divergence  # Beam divergence angle in radians
        self.polarization_angle = polarization_angle  # Linear polarization angle
        self.is_collimated = True
    
    @property
    def params(self):
        return self.wavelength, self.avg_watts, self.coherence_length, self.beam_divergence, self.polarization_angle

    def __repr__(self):
        return f"CoherentSource(wavelength={self.wavelength.to(ureg.nanometer)}, avg_watts={self.avg_watts})"


class BlackBodySource(LightSource):
    """Black body source (stars, incandescent bulbs, etc.) 
    with temperature in Kelvins and illuminance in lux (lumens per square meter)
    Epsilon is for gray-body approximations, its the emissivity of the body,
    which is a function of lambda (the wavelength).
    See: https://en.wikipedia.org/wiki/Emissivity"""

    @validate_units()
    def __init__(
        self, 
        *, 
        temperature=5778 * ureg.kelvin,
        illuminance=7000 * ureg.lux,
        emissivity: float = 1.0
        ):
        super().__init__()
        self.temperature = temperature      # Temperature in Kelvin
        self.emissivity = emissivity        # Emissivity (0-1, 1 = perfect black body)
        self._lux = illuminance

    @property
    def lux(self):
        if isinstance(self._lux, Enum):
            return self._lux.value
        return self._lux
    
    def show_power_spectrum(self):
        lam = np.linspace(1, 2000, 1000) * ureg.nanometer
        plt.plot(lam, self.radiance_per_wavelength(lam))
        plt.axvline(x=self.lambda_max(), color="r", linestyle="--")
    
    def radiance_per_wavelength(self, wavelength) -> float:
        """Calculate radiance at a specific wavelength for a black-body source.
        Plank's Law see: https://en.wikipedia.org/wiki/Planck%27s_law
        B(λ,T) = (2hc²/λ⁵) * 1/(e^(hc/λkT) - 1)"""        
        numerator = 2 * h * c**2 / (wavelength**5)
        denominator = math.exp(h * c / (wavelength * k * self.temperature)) - 1
        return numerator / denominator
    
    def lambda_max(self) -> float:
        # Wien's Law, see: https://en.wikipedia.org/wiki/Wien%27s_displacement_law
        return Wien / self.temperature
    
    def total_radiance(self) -> float:
        """Calculate the total radiance emitted by a black-body source over all wavelengths.
        This is the integral over all wavelengths of plank's law, but there's a closed
        form to calculate it, namely the Stefan Boltzmann Law.

        See: https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law
            https://hyperphysics.phy-astr.gsu.edu/hbase/thermo/stefan2.html

        Args:
            T: Source temperature in kelvin

        Returns:
            Total radiance of source (watts per square metre per steradian)
        """
        return sigma / np.pi * self.temperature**4

    @property
    def params(self):
        return self.temperature, self.emissivity, self.lux

    def __repr__(self):
        return f"BlackBodySource(temperature={self.temperature}, emissivity={self.emissivity}, illuminance={self._lux})"

# Concrete combinations with specific behaviors

class PulsedLaser(DynamicSource, CoherentSource):
    """
    Pulsed laser combining dynamic (time-varying) and coherent properties
    
    This should be modeled as an ACTIVE source:
    - Directional emission
    - High intensity, focused beam
    - Time-structured output (pulses)
    - Requires active power input
    """
    @validate_units()
    def __init__(
        self,  
        wavelength=ureg.nanometer, 
        frequency=10e6 * ureg.hertz, 
        pulse_width=5e-9 * ureg.second, 
        avg_watts=1.0 * ureg.watt,
        pulse_shape: str = "gaussian"
        ):
        
        # Initialize both parent classes
        DynamicSource.__init__(self, pulse_width=pulse_width, pulse_shape=pulse_shape)
        CoherentSource.__init__(self, wavelength=wavelength, avg_watts=avg_watts)
        
        self.frequency = frequency  
        self.max_resolvable_depth = tof2depth(1 / frequency)
        self.gaussian = (pulse_shape.lower() == "gaussian")
        self.peak_watts = (self.avg_watts / (pulse_width * frequency)).to_reduced_units().to_compact()
        self.peak_watts = 2 * np.sqrt(np.log(2) / np.pi) * self.peak_watts if self.gaussian else self.peak_watts
        self.num_photons_per_cycle = watts2photons(self.avg_watts, 1 / self.frequency, self.wavelength)

    @ureg.check(None, None, ureg.meter, None, ureg.steradian, ureg.meter)
    def get_scene_radiance(self, rho_hat, depth_map, num_pixels, omega, epsilon=1e-12 * ureg.meters):
        # Instead of returning the radiance in units of W/m^2, this factors in the pulse width
        # and light source frequency and thus returns the radiance per cycle in units of #photons/m^2
        # Note: this assumes a lambertian BRDF as we have rho/pi.
        num_photons_per_solid_angle = self.num_photons_per_cycle / (num_pixels * omega)
        radiance = rho_hat / np.pi * num_photons_per_solid_angle / (depth_map + epsilon) ** 2
        return radiance.to(radiance_photons)
    
    def sigma(self, as_depth=False):
        if not self.gaussian:
            raise NotImplementedError("Sigma attribute is only for gaussian pulses")
        pulse_width = 2.0 * tof2depth(self.pulse_width) if as_depth else self.pulse_width
        return pulse_width / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    
    @property
    def params(self):
        return self.wavelength, self.frequency, self.pulse_width, self.avg_watts, self.pulse_shape
    
    def __repr__(self):
        return f"PulsedLaser(wavelength={self.wavelength.to(ureg.nanometer)}, frequency={self.frequency.to(ureg.hertz)}, pulse_width={self.pulse_width.to(ureg.nanosecond)}, avg_watts={self.avg_watts}, pulse_shape={self.pulse_shape})"

class Sun(ConstantSource, BlackBodySource):
    """
    Solar radiation combining constant and black body properties
    
    This should be modeled as an AMBIENT source:
    - Provides background lighting
    - Considered constant over short time scales
    - Natural, diffuse illumination
    """
    @validate_units()
    def __init__(
        self, 
        *,
        intensity=3.828e26 * ureg.watt/ureg.meter**2,       # Solar luminosity in watts
        stability_factor=0.01 * ureg.dimensionless,
        temperature=5778 * ureg.kelvin,         # Solar surface temperature
        lambda_pass=500e-9 * ureg.meter,       # Peak visible wavelength
        delta_lambda=10e-9 * ureg.meter,       # Bandwidth
        light_conditions=LightConditions.BRIGHT_SUNLIGHT
        ):
        
        # Initialize both parent classes
        ConstantSource.__init__(self, intensity=intensity, stability_factor=stability_factor)
        BlackBodySource.__init__(self, temperature=temperature)
        
        self.lambda_pass = lambda_pass
        self.delta_lambda = delta_lambda
        # Get percentage of power that passes through filter
        filter_lam = (
            np.array(
                [
                    i.to(ureg.meter).magnitude
                    for i in (lambda_pass - delta_lambda, lambda_pass, lambda_pass + delta_lambda)
                ]
            )
            * ureg.meters
        )
        passes_through = np.trapz(self.radiance_per_wavelength(filter_lam), x=filter_lam)
        self.c_eff = passes_through / self.total_radiance()
        self.c_eff.ito(ureg.dimensionless)

    @ureg.check(None, ureg.steradian, None, ureg.hertz)
    def get_scene_radiance(self, omega, rho_hat, frequency):
        # Get scene radiance due to ambient source
        # For sunlight the conversion from lux to watts is 0.0079 [Source?]
        # Note: this assumes a lambertian BRDF as we have rho/pi.
        watts_eff_per_area = 0.0079 * self.lux.to(ureg.lux).magnitude * self.c_eff * ureg.watt
        photons_eff_per_area_per_cycle = (
            watts2photons(watts_eff_per_area, 1 / frequency, self.lambda_pass) / ureg.meter**2
        )
        radiance = photons_eff_per_area_per_cycle.to(ureg.count / ureg.meter**2) * (omega * rho_hat) / np.pi
        return radiance.to(radiance_photons)

    @property
    def params(self):
        return self.intensity, self.stability_factor, self.temperature, self.lambda_pass, self.delta_lambda, self.light_conditions
    
    def __repr__(self):
        return f"Sun(intensity={self.intensity.to(ureg.watt)}, stability_factor={self.stability_factor.to(ureg.dimensionless)}, temperature={self.temperature.to(ureg.kelvin)}, lambda_pass={self.lambda_pass.to(ureg.nanometer)}, delta_lambda={self.delta_lambda.to(ureg.nanometer)}, light_conditions={self.light_conditions})"

class FlickeringLamp(DynamicSource, BlackBodySource):
    """
    Flickering lamp combining dynamic (time-varying) and black body properties
    
    This should be modeled as an AMBIENT source:
    - Provides background lighting with temporal variations
    - Natural flickering behavior (like incandescent bulbs, fluorescent lights)
    - Black body spectral characteristics
    """
    @validate_units()
    def __init__(
        self,
        *,
        temperature=2700 * ureg.kelvin,           # Typical incandescent bulb temperature
        illuminance=800 * ureg.lux,               # Typical room lighting
        modulation_frequency=120 * ureg.hertz,    # AC power frequency (US: 60Hz, EU: 50Hz)
        modulation_amplitude=0.1 * ureg.dimensionless,  # 10% flicker amplitude
        phase=0.0 * ureg.radian,
        pulse_width=1e-3 * ureg.second,          # 1ms pulse width for flicker
        pulse_shape="gaussian",
        emissivity=0.9                            # Typical bulb emissivity
        ):
        
        # Initialize both parent classes
        DynamicSource.__init__(self, modulation_frequency=modulation_frequency, modulation_amplitude=modulation_amplitude, phase=phase, pulse_width=pulse_width, pulse_shape=pulse_shape)
        BlackBodySource.__init__(self, temperature=temperature, illuminance=illuminance, emissivity=emissivity)
        
        # Calculate effective intensity based on black body properties
        self.effective_intensity = self._calculate_effective_intensity()
    
    def _calculate_effective_intensity(self) -> Quantity:
        """Calculate the effective intensity based on black body temperature and illuminance"""
        # Convert illuminance to watts per square meter using typical conversion
        # For incandescent bulbs, approximately 0.0079 W/m² per lux
        watts_per_area = 0.0079 * self.lux.to(ureg.lux).magnitude * ureg.watt / ureg.meter**2
        return watts_per_area
    
    def get_intensity(self) -> Quantity:
        """Get the current intensity with flickering modulation"""
        base_intensity = self.effective_intensity
        # Apply dynamic modulation from DynamicSource
        modulation = 1 + self.modulation_amplitude * np.sin(
            2 * np.pi * self.modulation_frequency * time.time() + self.phase
        )
        return base_intensity * modulation
    
    @ureg.check(None, ureg.steradian, None, ureg.hertz)
    def get_scene_radiance(self, omega, rho_hat, frequency):
        """Get scene radiance due to flickering ambient source"""
        # Get current modulated intensity
        current_intensity = self.get_intensity()
        
        # Convert to photons per cycle
        photons_per_area_per_cycle = watts2photons(
            current_intensity, 
            1 / frequency, 
            self.lambda_max()  # Use peak wavelength from black body
        ) / ureg.meter**2
        
        # Calculate radiance (assumes lambertian BRDF)
        radiance = photons_per_area_per_cycle.to(ureg.count / ureg.meter**2) * (omega * rho_hat) / np.pi
        return radiance.to(radiance_photons)
    
    @property
    def params(self):
        return self.temperature, self.emissivity, self.lux, self.modulation_frequency, self.modulation_amplitude, self.phase, self.pulse_width, self.pulse_shape
    
    def __repr__(self):
        return f"FlickeringLamp(temperature={self.temperature.to(ureg.kelvin)}, illuminance={self.lux.to(ureg.lux)}, modulation_frequency={self.modulation_frequency.to(ureg.hertz)}, modulation_amplitude={self.modulation_amplitude.to(ureg.dimensionless)})"

class CombinedSource:
    def __init__(self, *sources):
        # Filter out None sources
        self.sources = [s for s in sources if s is not None]

    def get_scene_radiance(self, *args, **kwargs):
        result = 0
        for source in self.sources:
            result += source.get_scene_radiance(*args, **kwargs)
        return result