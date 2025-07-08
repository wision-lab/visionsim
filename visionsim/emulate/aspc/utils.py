from pint import UnitRegistry, set_application_registry
from scipy.constants import Wien, c, h, k, sigma

ureg = UnitRegistry()
ureg.setup_matplotlib(True)
Q_ = ureg.Quantity
set_application_registry(ureg)

radiance = ureg.watts / ureg.steradian / ureg.meter**2
irradiance = ureg.watts / ureg.meter**2

radiance_photons = ureg.count / ureg.steradian / ureg.meter**2
irradiance_photons = ureg.count / ureg.meter**2

@ureg.wraps(ureg.meter, ureg.second)
def tof2depth(t):
    return c * t / 2

@ureg.wraps(ureg.count, (ureg.watt, ureg.second, ureg.meter))
def watts2photons(watts, t, wavelength):
    energy = watts * t
    photon_energy = h * c / wavelength
    return energy / photon_energy