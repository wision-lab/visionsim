from __future__ import annotations

import functools
import inspect
import math
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Callable


import numpy as np
import pint
import torch
from pint import Quantity, UnitRegistry, set_application_registry
from ruamel.yaml import YAML
from scipy.constants import c  # Top-level export for unit tests
import torch.nn.functional as F

ureg = UnitRegistry()
ureg.setup_matplotlib(True)
Q_ = ureg.Quantity
set_application_registry(ureg)

radiance = ureg.watts / ureg.steradian / ureg.meter**2
irradiance = ureg.watts / ureg.meter**2
radiance_photons = ureg.count / ureg.steradian / ureg.meter**2
irradiance_photons = ureg.count / ureg.meter**2


def validate_units(enum_aware=True, ignore_kwargs=None):
    ignore_kwargs = ignore_kwargs or []

    def wrapper(func: Callable) -> Callable:
        @functools.wraps(func)
        def inner(*args, **kwargs):
            signature = inspect.signature(func)
            params = signature.parameters.items()

            params = filter(lambda it: it[0] in kwargs and it[0] not in ignore_kwargs, params)
            params = filter(lambda it: it[1].kind != inspect.Parameter.VAR_POSITIONAL, params)
            params = filter(lambda it: it[1].default != inspect.Parameter.empty, params)

            for param_name, param_value in params:
                value = kwargs.get(param_name)

                if enum_aware and isinstance(value, Enum):
                    if isinstance(param_value.default, Enum):
                        continue
                    value = value.value

                if isinstance(param_value.default, Quantity):
                    if not isinstance(value, Quantity):
                        raise pint.PintTypeError(
                            f"Expected a quantity for keyword-argument '{param_name}' with "
                            f"dimensionality {param_value.default.dimensionality}, got {type(value)}."
                        )
                    if not value.check(param_value.default.dimensionality):
                        raise pint.DimensionalityError(value, param_value.default)

            return func(*args, **kwargs)

        return inner

    return wrapper


@ureg.wraps(ureg.meter, ureg.second)
def tof2depth(t):
    return c * t / 2


@ureg.wraps(ureg.count, (ureg.watt, ureg.second, ureg.meter))
def watts2photons(watts, t, wavelength):
    from scipy.constants import h

    energy = watts * t
    photon_energy = h * c / wavelength
    return energy / photon_energy


@ureg.wraps(ureg.radian, (ureg.meter, ureg.meter))
def fov_from_focal_length(fl, d):
    return 2 * np.arctan2(d, 2 * fl)


@ureg.wraps(ureg.meter, (ureg.radian, ureg.meter))
def focal_length_from_fov(fov, d):
    return d / (2 * np.tan(fov / 2))


@ureg.wraps(ureg.steradian, (ureg.radian, ureg.radian))
def pyramid_solid_angle(a, b):
    return 4 * np.arcsin(np.sin(a / 2) * np.sin(b / 2))


def get_irradiance_with_fov(irradiance, sensor_fov, pixel_fov, omega, w, h):
    fov_x, fov_y = sensor_fov[0], sensor_fov[1]
    w_ratio, h_ratio = pixel_fov[3] - pixel_fov[2], pixel_fov[1] - pixel_fov[0]
    w_new, h_new = w * w_ratio, h * h_ratio
    fov_x, fov_y = fov_x * w_ratio, fov_y * h_ratio
    omega_new = pyramid_solid_angle(fov_x, fov_y) / (w_new * h_new)
    scale = (omega_new / omega).to(ureg.dimensionless).magnitude
    return irradiance * scale


def preproc_frames(
    depth_frame: np.ndarray | torch.Tensor,
    albedo_frame: np.ndarray | torch.Tensor,
    ambient_frame: np.ndarray | torch.Tensor,
    config: dict,
    device: str | torch.device = "cpu",
    requires_grad: bool = False,
) -> tuple[Quantity, Quantity, Quantity]:
    """Preprocesses a single frame triplet (depth, albedo, ambient) into PyTorch Tensors

    wrapped in Pint quantities.
    """
    Nr, Nc = config["sensor"]["size"]

    # 1. Validate max depth against laser pulse repetition rate
    pulse_rep = config["active_source"]["pulsed_laser"]["pulse_repetition"]
    if not hasattr(pulse_rep, "to"):
        pulse_rep = pulse_rep * ureg.hertz

    tmax = (1.0 / pulse_rep).to(ureg.second)
    max_resolvable_depth = (tmax * (c * ureg.meter / ureg.second) / 2).to(ureg.meter)

    scale_depth = config["histogrammer"]["max_depth"]
    if not hasattr(scale_depth, "to"):
        scale_depth = scale_depth * ureg.meter
    else:
        scale_depth = scale_depth.to(ureg.meter)

    if scale_depth > max_resolvable_depth:
        raise ValueError(
            f"histogrammer.max_depth ({scale_depth}) exceeds max resolvable depth ({max_resolvable_depth})."
        )
    max_depth = scale_depth.magnitude

    def _to_numpy(frame):
        # Dataset.__getitem__ returns (data, transform); image array is at index 0
        if isinstance(frame, (tuple, list)) and len(frame) == 2:
            frame = frame[0]

        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()

        return np.asarray(frame)

    # 2. Process Depth Frame
    raw_depth = _to_numpy(depth_frame)
    if raw_depth.ndim == 3:
        raw_depth = raw_depth[:, :, 0]

    if np.issubdtype(raw_depth.dtype, np.integer):
        depth_img = raw_depth.astype(np.float32) / 255.0 * float(max_depth)
    else:
        depth_img = raw_depth.astype(np.float32)

    depth_img = np.where(np.isfinite(depth_img) & (depth_img > 0), depth_img, 0.0)

    # 3. Process Albedo Frame (Diffuse Color Pass)
    raw_albedo = _to_numpy(albedo_frame)
    if raw_albedo.ndim == 3:
        raw_albedo = raw_albedo[:, :, 0]

    if np.issubdtype(raw_albedo.dtype, np.integer):
        albedo_img = raw_albedo.astype(np.float32) / 255.0
    else:
        albedo_img = raw_albedo.astype(np.float32)

    # 4. Process Ambient Frame (Full RGB Render Pass)
    raw_ambient = _to_numpy(ambient_frame)
    if raw_ambient.ndim == 3:
        raw_ambient = raw_ambient[:, :, 0]

    if np.issubdtype(raw_ambient.dtype, np.integer):
        ambient_img = raw_ambient.astype(np.float32) / 255.0
    else:
        ambient_img = raw_ambient.astype(np.float32)

    # 5. Convert to PyTorch Tensors and Resize with F.interpolate
    # Shape requirements: F.interpolate expects (Batch, Channel, Height, Width) -> (1, 1, H, W)
    depth_tensor = torch.tensor(depth_img, device=device, dtype=torch.float64, requires_grad=requires_grad).unsqueeze(0).unsqueeze(0)
    albedo_tensor = torch.tensor(albedo_img, device=device, dtype=torch.float64, requires_grad=requires_grad).unsqueeze(0).unsqueeze(0)
    ambient_tensor = torch.tensor(ambient_img, device=device, dtype=torch.float64, requires_grad=requires_grad).unsqueeze(0).unsqueeze(0)

    # Apply interpolation: 'nearest' for depth, 'bilinear' for smooth color/ambient maps
    depth_tensor = F.interpolate(depth_tensor, size=(Nr, Nc), mode="nearest").squeeze(0).squeeze(0)
    albedo_tensor = F.interpolate(albedo_tensor, size=(Nr, Nc), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    ambient_tensor = F.interpolate(ambient_tensor, size=(Nr, Nc), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)

    depth_qty = ureg.Quantity(depth_tensor, "meter")
    albedo_qty = ureg.Quantity(albedo_tensor, "dimensionless")
    ambient_qty = ureg.Quantity(ambient_tensor, "dimensionless")

    return depth_qty, albedo_qty, ambient_qty


def _validate_config(config: dict) -> None:
    errors = []

    expected_dimensionality = {
        ("histogrammer", "min_depth"): ureg.meter.dimensionality,
        ("histogrammer", "max_depth"): ureg.meter.dimensionality,
        ("histogrammer", "bin_width"): ureg.meter.dimensionality,
        ("histogrammer", "dead_time"): ureg.second.dimensionality,
        ("active_source", "pulsed_laser", "wavelength"): ureg.meter.dimensionality,
        ("active_source", "pulsed_laser", "pulse_repetition"): ureg.hertz.dimensionality,
        ("active_source", "pulsed_laser", "pulse_width"): ureg.second.dimensionality,
        ("active_source", "pulsed_laser", "avg_watts"): ureg.watt.dimensionality,
        ("ambient_source", "sun", "temperature"): ureg.kelvin.dimensionality,
        ("ambient_source", "sun", "lambda_pass"): ureg.meter.dimensionality,
        ("ambient_source", "sun", "delta_lambda"): ureg.meter.dimensionality,
        ("ambient_source", "sun", "intensity"): (ureg.watt / ureg.meter**2).dimensionality,
        ("sensor", "pixel_pitch"): ureg.meter.dimensionality,
        ("sensor", "fov"): ureg.degree.dimensionality,
    }

    for key_path, expected_dim in expected_dimensionality.items():
        found, val = _get_nested_val(config, key_path)
        if not found or val is None:
            continue

        param_name = ".".join(key_path)

        if isinstance(val, (tuple, list)):
            for idx, item in enumerate(val):
                if not isinstance(item, Quantity):
                    errors.append(f"'{param_name}[{idx}]' must be a Pint Quantity, got {type(item).__name__}")
                elif not item.check(expected_dim):
                    errors.append(f"'{param_name}[{idx}]' has incorrect dimensionality.")
        elif isinstance(val, Quantity):
            if not val.check(expected_dim):
                errors.append(f"'{param_name}' has incorrect unit dimensionality.")
        else:
            errors.append(f"'{param_name}' must be a Pint Quantity, got {type(val).__name__}")

    if errors:
        raise ValueError("Unit Validation Errors:\n - " + "\n - ".join(errors))

    hist = config.get("histogrammer", {})
    active = config.get("active_source", {}).get("pulsed_laser", {})

    min_depth = hist.get("min_depth")
    max_depth = hist.get("max_depth")
    n_bins = hist.get("n_bins")
    bin_width = hist.get("bin_width")
    pulse_rep = active.get("pulse_repetition")

    if min_depth is not None and min_depth < 0 * ureg.meters:
        errors.append(f"min_depth must be >= 0m, got {min_depth}")

    if max_depth is not None and min_depth is not None and max_depth <= min_depth:
        errors.append(f"max_depth ({max_depth}) must be > min_depth ({min_depth})")

    if pulse_rep is not None and pulse_rep > 0 * ureg.megahertz:
        max_resolvable_depth = tof2depth(1 / pulse_rep)
        if max_depth is not None and max_depth > max_resolvable_depth:
            errors.append(f"Max depth ({max_depth}) exceeds max resolvable depth ({max_resolvable_depth}).")

        if n_bins is not None and bin_width is not None:
            implied_bw = max_resolvable_depth / n_bins
            ratio = (bin_width / implied_bw).to(ureg.dimensionless).magnitude
            if not np.isclose(ratio, 1.0, rtol=5e-2):
                errors.append(
                    f"Configured bin_width ({bin_width}) contradicts implied bin width ({implied_bw:.6f~P}) "
                    f"from pulse_repetition ({pulse_rep}) and n_bins ({n_bins})."
                )

    if errors:
        raise ValueError("Invalid configuration:\n - " + "\n - ".join(errors))


def yaml_constructor(cls_or_func, *extra_args):
    def constructor(loader, node):
        if hasattr(loader, "construct_scalar") and isinstance(node, loader.model.ScalarNode):
            val = loader.construct_scalar(node)
            return cls_or_func(val, *extra_args)
        elif hasattr(node, "value") and isinstance(node.value, str):
            return cls_or_func(node.value, *extra_args)
        elif isinstance(node.value, list):
            seq = [loader.construct_object(child) for child in node.value]
            return cls_or_func(seq, *extra_args)
        elif isinstance(node.value, dict):
            mapping = {loader.construct_object(k): loader.construct_object(v) for k, v in node.value.items()}
            return cls_or_func(*extra_args, **mapping)
        return cls_or_func(node.value, *extra_args)

    return constructor


def _load_config(config_path: Path | str | dict | None) -> dict:
    if config_path is None:
        return {}

    if isinstance(config_path, dict):
        return deepcopy(config_path)

    yaml = YAML(typ="safe")
    safe_builtins = {
        "__builtins__": {"list": list, "dict": dict, "tuple": tuple},
        "np": np,
        "math": math,
    }

    def file_tag_constructor(loader, node):
        val = loader.construct_scalar(node)
        file_p = Path(val)
        if config_path and not file_p.is_absolute():
            file_p = Path(config_path).parent / file_p
        return str(file_p)

    yaml.Constructor.add_constructor(tag="!Quantity", constructor=yaml_constructor(ureg.Quantity))
    yaml.Constructor.add_constructor(tag="!expr", constructor=yaml_constructor(eval, safe_builtins))
    yaml.Constructor.add_constructor(tag="!file", constructor=file_tag_constructor)

    with open(config_path, "r") as f:
        return yaml.load(f) or {}


def _to_quantity(val: Any, unit_str: str) -> Any:
    if val is None:
        return None

    if isinstance(val, ureg.Quantity):
        if hasattr(val.magnitude, "__len__") and not isinstance(val.magnitude, (str, bytes)):
            return [ureg.Quantity(x, val.units) for x in val.magnitude]
        return val

    if isinstance(val, (tuple, list)):
        return [x if isinstance(x, ureg.Quantity) else ureg.Quantity(x, unit_str) for x in val]

    if isinstance(val, (int, float)):
        return ureg.Quantity(val, unit_str)

    if isinstance(val, str):
        try:
            q = ureg.Quantity(val)
            if q.dimensionless:
                return ureg.Quantity(q.magnitude, unit_str)
            return q
        except Exception:
            return ureg.Quantity(float(val), unit_str)

    return val


def _get_nested_val(d: dict, path: tuple[str, ...]) -> tuple[bool, Any]:
    curr = d
    for step in path:
        if isinstance(curr, dict) and step in curr:
            curr = curr[step]
        else:
            return False, None
    return True, curr


def _build_config(
    DEFAULT_CONFIG: dict,
    user_args: dict,
    config_file: Path | str | dict | None = None,
) -> dict:
    yml_config = _load_config(config_file)

    cli_mapping = {
        "min_depth": ("histogrammer", "min_depth"),
        "max_depth": ("histogrammer", "max_depth"),
        "n_bins": ("histogrammer", "n_bins"),
        "bin_width": ("histogrammer", "bin_width"),
        "pixel_fov_list": ("histogrammer", "pixel_fov_list"),
        "vignette": ("histogrammer", "vignette"),
        "n_pulses": ("histogrammer", "n_pulses"),
        "dead_time": ("histogrammer", "dead_time"),
        "free_running": ("histogrammer", "free_running"),
        "fast_sim": ("histogrammer", "fast_sim"),
        "active_enabled": ("active_source", "pulsed_laser", "enabled"),
        "active_wavelength": ("active_source", "pulsed_laser", "wavelength"),
        "active_pulse_repetition": ("active_source", "pulsed_laser", "pulse_repetition"),
        "active_pulse_width": ("active_source", "pulsed_laser", "pulse_width"),
        "active_avg_watts": ("active_source", "pulsed_laser", "avg_watts"),
        "active_pulse_shape": ("active_source", "pulsed_laser", "pulse_shape"),
        "ambient_enabled": ("ambient_source", "sun", "enabled"),
        "ambient_temperature": ("ambient_source", "sun", "temperature"),
        "ambient_light_conditions": ("ambient_source", "sun", "light_conditions"),
        "ambient_lambda_pass": ("ambient_source", "sun", "lambda_pass"),
        "ambient_delta_lambda": ("ambient_source", "sun", "delta_lambda"),
        "ambient_intensity": ("ambient_source", "sun", "intensity"),
        "sensor_size": ("sensor", "size"),
        "sensor_pixel_pitch": ("sensor", "pixel_pitch"),
        "sensor_f_number": ("sensor", "f_number"),
        "sensor_fov": ("sensor", "fov"),
    }

    cli_param_units = {
        "min_depth": "meters",
        "max_depth": "meters",
        "bin_width": "meters",
        "dead_time": "nanoseconds",
        "active_wavelength": "nanometers",
        "active_pulse_repetition": "megahertz",
        "active_pulse_width": "nanoseconds",
        "active_avg_watts": "watts",
        "ambient_temperature": "kelvin",
        "ambient_lambda_pass": "nanometers",
        "ambient_delta_lambda": "nanometers",
        "ambient_intensity": "watts / meter**2",
        "sensor_pixel_pitch": "micrometer",
        "sensor_fov": "degree",
    }

    config: dict = {}

    for param_name, key_path in cli_mapping.items():
        val = user_args.get(param_name)

        if val is None:
            found, yml_val = _get_nested_val(yml_config, key_path)
            if found and yml_val is not None:
                val = yml_val
            else:
                _, val = _get_nested_val(DEFAULT_CONFIG, key_path)

        if param_name in cli_param_units:
            val = _to_quantity(val, cli_param_units[param_name])

        d = config
        for step in key_path[:-1]:
            d = d.setdefault(step, {})
        d[key_path[-1]] = val

    for key in ("input_dir", "output_dir", "force", "pattern"):
        if key in user_args and user_args[key] is not None:
            config[key] = user_args[key]

    return config


def prepare_config(user_args: dict, config_file: Path | str | None = None) -> dict:
    DEFAULT_CONFIG = {
        "histogrammer": {
            "min_depth": 0 * ureg.meters,
            "max_depth": 10 * ureg.meters,
            "n_bins": 672,
            "bin_width": tof2depth(1 / (10 * ureg.megahertz)) / 672,
            "pixel_fov_list": [[0, 1.0, 0, 1.0]],
            "vignette": False,
            "n_pulses": 10000,
            "dead_time": 10 * ureg.nanoseconds,
            "free_running": True,
            "fast_sim": False,
        },
        "active_source": {
            "pulsed_laser": {
                "enabled": True,
                "wavelength": 940 * ureg.nanometers,
                "pulse_repetition": 10 * ureg.megahertz,
                "pulse_width": 6 * ureg.nanoseconds,
                "avg_watts": 0.000000007 * ureg.watts,
                "pulse_shape": "gaussian",
            }
        },
        "ambient_source": {
            "sun": {
                "enabled": True,
                "temperature": 5778 * ureg.kelvin,
                "light_conditions": "BRIGHT_SUNLIGHT",
                "lambda_pass": 550 * ureg.nanometers,
                "delta_lambda": 10 * ureg.nanometers,
                "intensity": 3.828e26 * ureg("watts / meter**2"),
            }
        },
        "sensor": {
            "size": (1080, 1920),
            "pixel_pitch": 10 * ureg.micrometer,
            "f_number": 1.4 * ureg.dimensionless,
            "fov": [90.5 * ureg.degree, 59.14 * ureg.degree],
        },
    }
    config = _build_config(DEFAULT_CONFIG, user_args, config_file=config_file)
    _validate_config(config)
    return config
