from collections import OrderedDict, defaultdict
from pint import Quantity
import textwrap
from units import validate_units
from utils import focal_length_from_fov, fov_from_focal_length, ureg, pyramid_solid_angle, radiance_photons, irradiance_photons, resize_like
from typing import Union, Tuple
from scipy.constants import c, h, k, sigma
import numpy as np

from sources import PulsedLaser, Sun, FlickeringLamp, CombinedSource
from histogrammer import Histogrammer

# Define Frame class if not imported
class Frame:
    def __init__(self, sensor, rgb, depth):
        self.sensor = sensor
        self.rgb = rgb
        self.depth = depth

class SensorBase:
    """Base class for all sensors"""

    @validate_units()
    def __init__(
        self,
        *,
        size: Union[int, Tuple] = (5400, 3600),
        aspect: float = None,
        f: Union[float, Tuple] = None,
        fov=(66 * ureg.degree, 44 * ureg.degree),
        aperture: float = None,
        f_number: float = 1.4,
        pixel_pitch=6.6667 * ureg.micrometer,
        fov_list=None,
    ):
        """Class that represents a sensor with perspective camera functionality. It builds the intrinsic
        camera matrix using the following parameters.

        Parameters that aren't supplied will be computed.
        One of (f, fov) and one of (aperture, f_number) are required.

        Note: Skew is currently not supported.

        Args:
            size: Sensor dimensions. If not specified, then assume 1x1.
            aspect: Aspect ration (w/h) of sensor, computed if not given.
            f: Focal length of Lenses per pixel (in meters)
            fov: Field of view of camera (in radians)
            aperture: This is the "clear aperture", the the diameter of the entrance pupil
                of the camera. This is not the lens diameter.
            f_number: F-Number of the camera, this is equal to f/aperture.
        """
        self._param_names = []

        self.size = size
        self.aspect = aspect
        self.f = f
        self.fov = fov
        self.aperture = aperture
        self.f_number = f_number
        self.pixel_pitch = pixel_pitch
        
        if self.size is None:
            self.size = 1
        if isinstance(self.size, int):
            if self.aspect:
                self.size = (int(self.size / self.aspect), self.size)
            else:
                self.size = (self.size, self.size)
        self.h, self.w = self.size
        self.aspect = aspect or self.w / self.h
        self.num_pixels = self.h * self.w
        self.diagonal = np.sqrt(self.h**2 + self.w**2)
        self.pixel_pitch = pixel_pitch
        self.fov_list = fov_list
        
        if not (f is not None) ^ (fov is not None):
            raise ValueError("Only one of focal length or FOV is required.")
        if f is None:
            if isinstance(fov, (int, float, Quantity)):
                # Assume FOV is diagonal FOV
                self.fov_x, self.fov_y = fov * self.w / self.diagonal, fov * self.h / self.diagonal
            else:
                self.fov_x, self.fov_y = fov
            self.f_x = focal_length_from_fov(self.fov_x, self.w * self.pixel_pitch)
            self.f_y = focal_length_from_fov(self.fov_y, self.h * self.pixel_pitch)
        if fov is None:
            if isinstance(f, (int, float)):
                self.f_x, self.f_y = f * self.h / self.diagonal, f * self.w / self.diagonal
            else:
                self.f_x, self.f_y = f
            self.fov_x = fov_from_focal_length(self.f_x, self.w)
            self.fov_y = fov_from_focal_length(self.f_y, self.h)
        self.f_x, self.f_y = self.f_x.to(ureg.millimeter), self.f_y.to(ureg.millimeter)
        self.fov_x, self.fov_y = self.fov_x.to(ureg.degree), self.fov_y.to(ureg.degree)
        self.f_diag = np.sqrt(self.f_x**2 + self.f_y**2).to(ureg.millimeter)
        self.fov_diag = np.sqrt(self.fov_x**2 + self.fov_y**2).to(ureg.degree)

        if not (aperture is not None) ^ (f_number is not None):
            raise ValueError("Only one of aperture or f_number is required.")
        if aperture is None:
            self.f_number = f_number
            self.aperture = self.f_diag / self.f_number
        elif f_number is None:
            self.aperture = aperture
            self.f_number = self.f_diag / self.aperture
        self.aperture.ito(ureg.centimeter)

        # Build up the camera intrinsics
        self.c_x, self.c_y = self.w / 2, self.h / 2

        # TODO: Add support for near and far plane see:
        #   https://www.khronos.org/registry/glTF/specs/2.0/glTF-2.0.html#projection-matrices
        self.intrinsics = np.array(
            [
                [self.f_x.to(ureg.meters).magnitude, 0, self.c_x, 0],
                [0, self.f_y.to(ureg.meters).magnitude, self.c_y, 0],
                [0, 0, 1, 0],
            ]
        )

        # Solid angle per pixel
        self.omega = pyramid_solid_angle(self.fov_x, self.fov_y)
        self.omega = self.omega / (self.w * self.h)

        # Set _param_names
        self._param_names = [
            "w",
            "h",
            "diagonal",
            "f_x",
            "f_y",
            "f_diag",
            "fov_x",
            "fov_y",
            "fov_diag",
            "omega",
            "f_number",
            "aperture",
            "aspect",
            "diagonal",
        ]

    @property
    def params(self):
        return OrderedDict((i, getattr(self, i)) for i in self._param_names)

    @property
    def sensor_area(self):
        return self.w * self.h * self.pixel_pitch**2

    def __hash__(self):
        return hash(tuple(self.params.items()))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        a = tuple(hash(i) if not hasattr(i, "pseudo_hash") else i.pseudo_hash for i in self.params.values())
        b = tuple(hash(i) if not hasattr(i, "pseudo_hash") else i.pseudo_hash for i in other.params.values())
        return hash(a) == hash(b)

    def __repr__(self):
        def formatter(k, v):
            if isinstance(v, Quantity):
                return f"{k}={v.to_compact():.2f~P}"
            elif isinstance(v, (int, float)):
                return f"{k}={v:.2f}"
            return f"{k}={v}"

        params = ",\n".join(formatter(k, v) for k, v in self.params.items())
        params = textwrap.indent(params, "\t")
        return f"{self.__class__.__name__}(\n{params}\n)"

    def map_camera2image(self, camera_points):
        """Map a point in the camera's coordinate frame to the image's
        coordinate frame.

        The camera's frame is right handed, centered on the optical axis with
        Z pointing out of the camera. The image's frame is centered at the upper
        left corner of the sensor with X going to the right, and Y down.

        Args:
            camera_points: an (3 or 4, N) dimensional array of points in 3D (or homogeneous 3D)

        Returns:
            image_points: an (2, N) array of projected points
        """
        if camera_points.shape[0] not in (3, 4):
            raise ValueError(
                f"Expected an array of 3D points with first dimension 3 or "
                f"4 (homogeneous), instead got {camera_points.shape[0]}."
            )
        if camera_points.shape[0] == 3:
            camera_points = np.pad(camera_points, ((0, 1),), mode="constant", constant_values=1)
        image_points = np.einsum("j..., ij->i...", camera_points, self.intrinsics)
        return image_points[:-1, ...] / image_points[-1, ...]

    @ureg.wraps(irradiance_photons, (None, radiance_photons))
    def get_irradiance(self, surface_radiance):
        r"""Let $E$ be the image irradiance and $L$ the surface radiance, then we have:
        $$E = L \frac{\pi}{4} \left(\frac{d}{f}\right)^2 cos^4(\alpha)$$
        With:
        - $d$: diameter of lenses
        - $f$: effective focal length
            - $\left(\frac{d}{f}\right)$: 1/f-number
        - $cos^4(\alpha)$: brightness falloff term, $\alpha$ assumed to be small enough"""
        return surface_radiance * np.pi / 4 * (1 / self.f_number) ** 2


class SPADSensor(SensorBase):

    @validate_units()
    def __init__(
        self,
        **sensor_base_kwargs,
    ):
        super().__init__(**sensor_base_kwargs)
        self._sensor_base_kwargs = sensor_base_kwargs
        self.fov_list = sensor_base_kwargs.get('fov_list', None)

    @classmethod
    def from_params(cls, params):
        params.update(**params.pop("sensor_base_kwargs", {}))
        return cls(**params, reset_on_init=False)