from pint import UnitRegistry, set_application_registry
from scipy.constants import Wien, c, h, k, sigma
import numpy as np

ureg = UnitRegistry()
ureg.setup_matplotlib(True)
Q_ = ureg.Quantity
set_application_registry(ureg)

radiance = ureg.watts / ureg.steradian / ureg.meter**2
irradiance = ureg.watts / ureg.meter**2

radiance_photons = ureg.count / ureg.steradian / ureg.meter**2
irradiance_photons = ureg.count / ureg.meter**2

#############################################
#              Physical laws                #
#############################################

@ureg.wraps(ureg.meter, ureg.second)
def tof2depth(t):
    return c * t / 2

@ureg.wraps(ureg.count, (ureg.watt, ureg.second, ureg.meter))
def watts2photons(watts, t, wavelength):
    energy = watts * t
    photon_energy = h * c / wavelength
    return energy / photon_energy

#############################################
#          Optics Modeling Utils            #
#############################################


@ureg.wraps(ureg.radian, (ureg.meter, ureg.meter))
def fov_from_focal_length(fl, d):
    """Convert from the focal length to angular FOV
    See: https://en.wikipedia.org/wiki/Angle_of_view

    Args:
        fl: Focal length, either per pixel or total
        d: Image sensor dimension. If focal length is
        f_x/f_y then d should be the pixel pitch. If the
        focal length is for the whole imaging system (i.e
        it refers to the diagonal) then d should be the
        image sensor's diagonal length.

    Returns:
        Field of view in radians
    """
    return 2 * np.arctan2(d, 2 * fl)

@ureg.wraps(ureg.meter, (ureg.radian, ureg.meter))
def focal_length_from_fov(fov, d):
    """
    Convert from the angular FOV to the focal length,
    See `fov_from_focal_length` for details.
    """
    # Here FOV is twice the apex angle
    return d / (2 * np.tan(fov / 2))

@ureg.wraps(ureg.steradian, (ureg.radian, ureg.radian))
def pyramid_solid_angle(a, b):
    """
    Solid angle subtended by a four-sided right rectangular pyramid with apex angles a and b
    See: https://en.wikipedia.org/wiki/Solid_angle#Pyramid
    """
    return 4 * np.arcsin(np.sin(a / 2) * np.sin(b / 2))