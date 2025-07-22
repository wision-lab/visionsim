from collections import OrderedDict
from pint import Quantity
import textwrap
from units import validate_units
from utils import focal_length_from_fov, fov_from_focal_length, ureg, pyramid_solid_angle, radiance_photons, irradiance_photons
from typing import Union, Tuple
from scipy.constants import c, h, k, sigma
import numpy as np
from functools import partial

from .sources import PulsedLaser, Sun, FlickeringLamp, CombinedSource

def make_fov_mask(
    image_shape,
    orig_fov,
    new_fov=None,
    vignette=False,
    vignette_shape='circle',
    user_mask=None,
    vignette_strength=1.0,
    softness=2.0,
):
    """
    Create a FOV mask for an image, with optional vignetting and user mask support.
    FOV mask and vignette are exclusive: if new_fov is set and different from orig_fov, only FOV mask is applied;
    if vignette is True, only vignette is applied.
    Args:
        image_shape: (height, width) of the image
        orig_fov: (fov_x, fov_y) original FOV in degrees or radians
        new_fov: (fov_x, fov_y) new FOV to mask to; if None, use original (no mask)
        vignette: bool, whether to apply vignetting
        vignette_shape: 'circle', 'ellipse', or 'rect'
        user_mask: np.ndarray or None, user-supplied mask (same shape as image)
        vignette_strength: float, 0 (none) to 1 (full)
        softness: float, controls the smoothness of the vignette gradient (higher = softer)
    Returns:
        mask: np.ndarray, float32, shape=image_shape, 0=masked, 1=fully visible
    """
    h, w = image_shape
    mask = np.ones((h, w), dtype=np.float32)

    if new_fov is not None and tuple(new_fov) != tuple(orig_fov):
        # FOV mask only
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        orig_fov_x, orig_fov_y = orig_fov
        new_fov_x, new_fov_y = new_fov
        norm_x = (x - cx) / (w / 2) * np.tan(orig_fov_x / 2)
        norm_y = (y - cy) / (h / 2) * np.tan(orig_fov_y / 2)
        fov_limit_x = np.tan(new_fov_x / 2) / np.tan(orig_fov_x / 2)
        fov_limit_y = np.tan(new_fov_y / 2) / np.tan(orig_fov_y / 2)
        mask_ellipse = ((norm_x / fov_limit_x) ** 2 + (norm_y / fov_limit_y) ** 2) <= 1
        mask = mask * mask_ellipse.astype(np.float32)
    elif vignette:
        # Vignette only
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        norm_x = (x - cx) / (w / 2)
        norm_y = (y - cy) / (h / 2)
        if vignette_shape == 'circle':
            r = np.sqrt(norm_x**2 + norm_y**2)
        elif vignette_shape == 'ellipse':
            r = np.sqrt(norm_x**2 + norm_y**2)
        elif vignette_shape == 'rect':
            r = np.maximum(np.abs(norm_x), np.abs(norm_y))
        else:
            raise ValueError(f"Unknown vignette_shape: {vignette_shape}")
        vign = (1 - vignette_strength * r) ** softness
        vign = np.clip(vign, 0, 1)
        mask = mask * vign
    # else: mask remains all ones

    # User mask
    if user_mask is not None:
        mask = mask * user_mask.astype(np.float32)

    return mask

class SensorBase:
    """Base class for all sensors"""

    def __init__(self):
        self._param_names = []

    def get_irradiance(self, surface_radiance):
        raise NotImplementedError()

    @property
    def params(self):
        return OrderedDict((i, getattr(self, i)) for i in self._param_names)

    @property
    def sensor_area(self):
        raise NotImplementedError()

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

class PerspectiveCamera(SensorBase):
    """Model a perspective camera. Default values here should
    model a full-frame (35x24mm sensor) camera with ~19.4Mpix"""

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
    ):
        """Class that represents a perspective camera. It builds the intrinsic
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
        super().__init__()

        if size is None:
            size = 1
        if isinstance(size, int):
            if aspect:
                size = (int(size / aspect), size)
            else:
                size = (size, size)
        self.h, self.w = size
        self.aspect = aspect or self.w / self.h
        self.num_pixels = self.h * self.w
        self.diagonal = np.sqrt(self.h**2 + self.w**2)
        self.pixel_pitch = pixel_pitch

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
    def sensor_area(self):
        return self.w * self.h * self.pixel_pitch**2

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


class SPADCamera(PerspectiveCamera):
    # TODO: Add rolling shutter capabilities

    @validate_units()
    def __init__(
        self,
        *,
        histogrammer: Histogrammer = object(),
        active_source: PulsedLaser = object(),
        constant_ambient_source: Union[Sun, None] = None,
        dynamic_ambient_source: Union[FlickeringLamp, None] = None,      
        offsets: Union[Quantity, np.ndarray] = None,
        reset_on_init=True,
        **perspective_camera_kwargs,
    ):
        super().__init__(**perspective_camera_kwargs)
        self._validate_components(histogrammer, active_source, constant_ambient_source, dynamic_ambient_source)
        self._perspective_camera_kwargs = perspective_camera_kwargs
        self._param_names.extend(["histogrammer", "active_source", "ambient_source"])
        self.histogrammer = histogrammer
        self.active_source = active_source   
        self.ambient_source = SPADCamera.combine_sources(constant_ambient_source, dynamic_ambient_source)  
        self.offsets = offsets

        if reset_on_init:
            self.histogrammer.reshape((self.h, self.w))
            self.reset()

        self.flux_levels = partial(self.histogrammer.flux_levels, self)
        self.detection_probability = partial(self.histogrammer.detection_probability, self)
        self.sbr = partial(self.histogrammer.sbr, self)
        self.sbd = partial(self.histogrammer.sbd, self)
        self.to_transients = partial(self.histogrammer.to_transients, self)

    @staticmethod
    def _validate_components(histogrammer, active_source, ambient_source, dynamic_ambient_source=None):
        if not isinstance(histogrammer, Histogrammer):
            raise ValueError("A histogrammer is required")
        if not isinstance(active_source, PulsedLaser):
            raise ValueError("Active light source is required")

        if active_source.max_resolvable_depth < histogrammer.max_depth:
            raise ValueError(
                f"Depth range is not unambiguous! Max depth is {histogrammer.max_depth.to_compact():~P} "
                f"but the laser repetition rate of {active_source.frequency:~P} implies a max "
                f"unambiguous depth of only {active_source.max_resolvable_depth.to(ureg.meter):~P}."
            )

        # Check both ambient_source and dynamic_ambient_source for bandpass blocking
        for src in (ambient_source, dynamic_ambient_source):
            if src is not None:
                if (
                    src.lambda_pass - src.delta_lambda > active_source.wavelength
                    or active_source.wavelength > src.lambda_pass + src.delta_lambda
                ):
                    bandpass = src.lambda_pass.plus_minus(src.delta_lambda).to(ureg.nanometer)
                    raise ValueError(
                        f"Active light source with wavelength {active_source.wavelength.to(ureg.nanometer):.2f~P} "
                        f"will be completely blocked by bandpass-filter centered around {bandpass.to(ureg.nanometer):.2f~P}"
                    )

    @staticmethod
    def combine_sources(*sources):
        # Remove None sources
        sources = [s for s in sources if s is not None]
        if not sources:
            return None
        if len(sources) == 1:
            return sources[0]
        return CombinedSource(*sources)

    @ureg.check(None, None, ureg.meter)
    def add_measurement(self, rhos, point_depths):
        # Calculate active component of transient
        # Note: We only store "impulses" which accumulate the dirac pulses into a sparse array.
        active_radiance = self.active_source.get_scene_radiance(rhos, point_depths, self.num_pixels, self.omega)
        weights = self.get_irradiance(active_radiance) * self.pixel_pitch**2

        # TODO: Implement `safe` getitem in Histogrammer to allow for histogramer += when depth values are invalid
        #   This will require support for Ellipses ...
        idxs, mask = self.histogrammer.bin_index(point_depths, mask=True)
        idxs, weights = idxs[~mask], weights[~mask].to(ureg.dimensionless).magnitude
        self.histogrammer.impulses[~mask.flatten(), idxs.flatten()] += weights.flatten()

        # Calculate DC offset due to ambient light
        # Note: The ambient radiance is received over the whole pulse, it needs to be
        #   divided by n_bins when constructing the final transient.
        if self.ambient_source is not None:
            ambient_radiance = self.ambient_source.get_scene_radiance(self.omega, rhos, self.active_source.frequency)
            ambient_irradiance = self.get_irradiance(ambient_radiance) * self.pixel_pitch**2
            self.offsets += ambient_irradiance

    def crossover_depth(self, smoothing_factor=1.0):
        """Get depth at which the signal flux becomes equal to background flux

        This uses the impulse response so is unaffected by pulse width, or noise.
        It is meant as a best-case scenario max depth not a realistic max-depth.

        Note:
            1) These calculations bypass the irradiance calculation step as a
            simplification step. We just solve for for depth for the condition
            ambient_radiance == active_radiance.
            2) This ignores the max depth permitted by both the rep-rate and
            the histogrammer itself.

        Notice: This depth does *not* depend on the scene patch's albedo as it gets
            cancelled out. This should make intuitive sense as both the signal and
            background flux depend linearly on the patch's albedo. It also does not
            depend on the pixel's area as that affects both as well.
            It is affected by the laser power (and indirectly the frequency since we
            use ~average-power*freq) and the FoV of the sensor.
        """
        ambient_radiance = self.ambient_source.get_scene_radiance(self.omega, 1.0, self.active_source.frequency)
        ambient_radiance /= self.histogrammer.n_bins

        num_photons_per_solid_angle = self.active_source.num_photons_per_cycle / (self.num_pixels * self.omega)
        depth_squared = (1.0 / np.pi * num_photons_per_solid_angle * smoothing_factor) / ambient_radiance
        return np.sqrt(depth_squared)

    def reset(self):
        self.offsets = np.zeros((self.h, self.w), dtype=np.float32) * ureg.count
        self.histogrammer.reset()

    @staticmethod
    def process_rgbd_stream(sensors, stream, cycles=1000, max_frames=-1):
        """Integrate RGBD stream over n-`cycles` and fill-in histogrammer data.

        It's much more efficient to only iterate over the stream once so here we allow
        multiple sensors (and `cycles`) to be used at once.
        """
        sensors = list(sensors)
        cycles = [cycles] * len(sensors) if isinstance(cycles, int) else cycles
        num_frames = defaultdict(int)

        for sensor in sensors:
            sensor.reset()

        for i, ((rgb_idx, rgb_md, rgb_frame), (depth_idx, depth_md, depth_frame)) in enumerate(stream):
            depth_frame[depth_frame == 0] = np.nan
            rhos = rgb_frame[..., 1].astype(np.float32) / 255.0

            # Make sure albedo map is the same size as depth map
            if depth_frame.shape != rhos.shape:
                rhos = resize_like(rhos, depth_frame)

            for sensor in sensors:
                if sensor is not None:
                    sensor.add_measurement(rhos, depth_frame * ureg.meter)

            for idx, (sensor, n) in enumerate(zip(sensors, cycles)):
                if sensor is None:
                    continue

                if i % n == n - 1:
                    yield num_frames[idx], idx, Frame(sensor=sensor, rgb=rgb_frame, depth=depth_frame)
                    num_frames[idx] += 1
                    sensor.reset()

                    # Remove sensor once it's emitted `max_frames`
                    if 0 < max_frames <= num_frames[idx]:
                        sensors[idx] = None

    @classmethod
    def from_params(cls, params):
        params.update(**params.pop("perspective_camera_kwargs", {}))
        return cls(**params, reset_on_init=False)

    def __reduce__(self):
        # This is called when pickling the object.
        # We return a constructor and any arguments it needs.
        return self.from_params, (
            dict(
                histogrammer=self.histogrammer,
                active_source=self.active_source,
                ambient_source=self.ambient_source,
                offsets=self.offsets,
                perspective_camera_kwargs=self._perspective_camera_kwargs,
            ),
        )
