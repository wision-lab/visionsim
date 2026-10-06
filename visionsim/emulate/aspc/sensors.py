import textwrap
from collections import OrderedDict
from typing import Tuple

import numpy as np
from pint import Quantity

from visionsim.emulate.aspc.utils import (
    irradiance_photons,
    pyramid_solid_angle,
    radiance_photons,
    ureg,
)


def _ensure_quantity(val, default_unit):
    """Ensure value is wrapped in a Pint Quantity with default_unit if unitless."""
    if isinstance(val, Quantity):
        return val
    return val * default_unit


class SensorBase:
    """Base class for camera sensors with unit-aware perspective projections."""

    def __init__(
        self,
        *,
        size: Tuple[int, int] = (1080, 1920),
        pixel_pitch=10 * ureg.micrometer,
        fov=(90.5 * ureg.degree, 90.5 * ureg.degree),
        f_number=1.4,
    ):
        self.h, self.w = size
        self.num_pixels = self.h * self.w
        self.pixel_pitch = _ensure_quantity(pixel_pitch, ureg.micrometer)
        self.f_number = _ensure_quantity(f_number, ureg.dimensionless)
        self.fov = fov
        # 1. Physical sensor dimensions & aspect ratio
        self.sensor_w = (self.w * self.pixel_pitch).to(ureg.millimeter)
        self.sensor_h = (self.h * self.pixel_pitch).to(ureg.millimeter)
        self.diagonal = np.sqrt(self.sensor_w**2 + self.sensor_h**2).to(ureg.millimeter)
        self.aspect = float(self.w / self.h)

        # 2. Extract or unpack FOV across all input formats
        fov_x_raw, fov_y_raw = None, None

        if isinstance(fov, Quantity):
            # Quantity array (YAML !Quantity [.5 degree, .5 degree]) or scalar Quantity
            if getattr(fov, "ndim", 0) > 0 and len(fov) >= 2:
                fov_x_raw, fov_y_raw = fov[0], fov[1]
            elif isinstance(getattr(fov, "magnitude", None), (list, tuple, np.ndarray)) and len(fov.magnitude) >= 2:
                fov_x_raw, fov_y_raw = fov[0], fov[1]
            else:
                fov_x_raw = fov
        elif isinstance(fov, (list, tuple)):
            # Python list/tuple of Quantities or numbers
            fov_x_raw = fov[0]
            if len(fov) > 1:
                fov_y_raw = fov[1]
        else:
            # Raw scalar
            fov_x_raw = fov

        self.fov_x = _ensure_quantity(fov_x_raw, ureg.degree).to(ureg.degree)

        if fov_y_raw is not None:
            self.fov_y = _ensure_quantity(fov_y_raw, ureg.degree).to(ureg.degree)
        else:
            # Derive fov_y using sensor aspect ratio if only scalar FOV provided
            tan_half_y = np.tan(self.fov_x.to(ureg.rad).magnitude / 2.0) / self.aspect
            self.fov_y = (2.0 * np.arctan(tan_half_y) * ureg.rad).to(ureg.degree)

        # 3. Calculate diagonal FOV
        tan_x2 = np.tan(self.fov_x.to(ureg.rad).magnitude / 2.0)
        tan_y2 = np.tan(self.fov_y.to(ureg.rad).magnitude / 2.0)
        self.fov_diag = (2.0 * np.arctan(np.sqrt(tan_x2**2 + tan_y2**2)) * ureg.rad).to(ureg.degree)

        # 4. Compute physical focal lengths
        self.f_x = (self.sensor_w / (2.0 * tan_x2)).to(ureg.millimeter)
        self.f_y = (self.sensor_h / (2.0 * tan_y2)).to(ureg.millimeter)
        self.f_diag = ((self.f_x + self.f_y) / 2.0).to(ureg.millimeter)

        # 5. Focal lengths in PIXELS for intrinsics K matrix
        self.f_x_px = float(self.w / (2.0 * tan_x2))
        self.f_y_px = float(self.h / (2.0 * tan_y2))

        # 6. Compute aperture diameter
        self.aperture = (self.f_diag / self.f_number).to(ureg.millimeter)

        # 7. Intrinsics matrix (K)
        self.c_x = self.w / 2.0
        self.c_y = self.h / 2.0
        self.intrinsics = np.array([
            [self.f_x_px, 0.0, self.c_x, 0.0],
            [0.0, self.f_y_px, self.c_y, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=float)

        # 8. Solid angle per pixel
        total_solid_angle = pyramid_solid_angle(self.fov_x, self.fov_y)
        self.omega = total_solid_angle / self.num_pixels

        # Parameter names list
        self._param_names = [
            "w",
            "h",
            "aspect",
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
        """Map a point in the camera's coordinate frame to the image frame."""
        if camera_points.shape[0] not in (3, 4):
            raise ValueError(
                f"Expected an array of 3D points with first dimension 3 or "
                f"4 (homogeneous), instead got {camera_points.shape[0]}."
            )
        if camera_points.shape[0] == 3:
            camera_points = np.pad(
                camera_points,
                ((0, 1), (0, 0)) if camera_points.ndim == 2 else ((0, 1),),
                mode="constant",
                constant_values=1,
            )
        image_points = np.einsum("j..., ij->i...", camera_points, self.intrinsics)
        return image_points[:-1, ...] / image_points[-1, ...]

    @ureg.wraps(irradiance_photons, (None, radiance_photons))
    def get_irradiance(self, surface_radiance):
        return surface_radiance * np.pi / 4 * (1 / self.f_number) ** 2


class SPADSensor(SensorBase):
    def __init__(self, **sensor_base_kwargs):
        super().__init__(**sensor_base_kwargs)
        self._sensor_base_kwargs = sensor_base_kwargs