from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np


def spad(
    input_dir: Path,
    output_dir: Path,
    pattern: str | None = None,
    factor: float = 1.0,
    seed: int = 2147483647,
    max_size: int = 1000,
    force: bool = False,
) -> None:
    """Perform bernoulli sampling on linearized RGB frames to yield binary frames

    Args:
        input_dir: directory in which to look for frames
        output_dir: directory in which to save binary frames
        pattern: used to find source image files to convert to binary frames,
            not needed when ``input_dir`` points to a valid dataset.
        factor: multiplicative factor controlling dynamic range of output
        seed: random seed to use while sampling, ensures reproducibility
        max_size: maximum number of frames per output array before rolling over to new file
        force: if true, overwrite output file(s) if present, else throw error
    """
    from numpy.lib.format import open_memmap

    from visionsim.dataset import Dataset, Metadata
    from visionsim.emulate.spc import emulate_spc
    from visionsim.utils.color import srgb_to_linearrgb
    from visionsim.utils.progress import ElapsedProgress

    if input_dir.resolve() == output_dir.resolve():
        raise RuntimeError("Input and output directory cannot be the same!")
    if output_dir.exists() and not force:
        raise FileExistsError("Output directory already exists.")
    else:
        shutil.rmtree(output_dir, ignore_errors=True)

    if pattern:
        dataset = Dataset.from_pattern(input_dir, pattern)
    else:
        dataset = Dataset.from_path(input_dir)

    rng = np.random.default_rng(int(seed))
    output_dir.mkdir(exist_ok=True, parents=True)
    transforms: list[dict[str, Any]] = []

    with ElapsedProgress() as progress:
        task = progress.add_task("Writing SPAD frames", total=len(dataset))

        for i, (data, transform) in enumerate(dataset):
            remainder = len(dataset) - (i // max_size) * max_size

            if transform["file_path"].suffix.lower() not in (".exr", ".hdr"):
                # Image has been tonemapped so undo mapping
                data = srgb_to_linearrgb((data / 255.0).astype(float))
            else:
                data = data.astype(float) / 255.0

            # Default to bitpacking width
            binary_img = emulate_spc(data, factor=factor, rng=rng) * 255
            binary_img = binary_img.astype(np.uint8) >= 128
            binary_img = np.packbits(binary_img, axis=1)

            offset = i % max_size
            file_path = output_dir / f"{i // max_size:04}.npy"
            transform["file_path"] = file_path.name
            transform["bitpack_dim"] = 2
            transform["offset"] = offset
            h, w, c = data.shape

            if not file_path.exists():
                data = open_memmap(
                    file_path,
                    mode="w+",
                    dtype=np.uint8,
                    shape=(
                        min(max_size, remainder),
                        transform.get("h", h),
                        math.ceil(transform.get("w", w) / 8),
                        transform.get("c", c),
                    ),
                )
                data[offset] = binary_img
            else:
                open_memmap(file_path)[offset] = binary_img

            transforms.append(transform)
            progress.update(task, advance=1)

    if not pattern:
        Metadata.from_dense_transforms(transforms).save(output_dir / "transforms.json")


def events(
    input_dir: Path,
    output_dir: Path,
    fps: float | None = None,
    pattern: str | None = None,
    pos_thres: float = 0.2,
    neg_thres: float = 0.2,
    sigma_thres: float = 0.03,
    cutoff_hz: float = 200.0,
    leak_rate_hz: float = 1.0,
    shot_noise_rate_hz: float = 10.0,
    refractory_period_s: float = 0.0,
    photoreceptor_noise: bool = False,
    leak_jitter_fraction: float = 0.1,
    noise_rate_cov_decades: float = 0.1,
    seed: int = 2147483647,
    cs_lambda_pixels: float | None = None,
    cs_tau_p_ms: float | None = None,
    scidvs: bool = False,
    blur_sigma: float = 0.0,
    preview_step: int | None = None,
    only_preview: bool = False,
    force: bool = False,
) -> None:
    """Emulate an event camera using v2e and high speed input frames

    Args:
        input_dir: directory in which to look for frames
        output_dir: directory in which to save events
        fps: frame rate of input sequence, will try to infer from dataset if possible
        pattern: used to find source image files to convert to events, not needed when ``input_dir`` points to a valid dataset
        pos_thres: nominal threshold of triggering positive event in log intensity
        neg_thres: nominal threshold of triggering negative event in log intensity
        sigma_thres: std deviation of threshold in log intensity
        cutoff_hz: 3dB cutoff frequency in Hz of DVS photoreceptor
        leak_rate_hz: leak event rate per pixel in Hz, from junction leakage in reset switch
        shot_noise_rate_hz: shot noise rate in Hz
        refractory_period_s: minimum inter-event interval per pixel in seconds
        photoreceptor_noise: model shot noise as correlated Gaussian noise injected at the photoreceptor
        leak_jitter_fraction: fractional random variation applied to the per-pixel leak rate
        noise_rate_cov_decades: standard deviation (in decades) of the log-normal per-pixel noise-rate distribution
        seed: random seed to use while sampling, ensures reproducibility
        cs_lambda_pixels: space constant of the centre-surround surround in pixels
        cs_tau_p_ms: time constant of the surround low-pass filter in ms
        scidvs: simulate the high-gain adaptive photoreceptor of the SCIDVS pixel
        blur_sigma: standard deviation of the Gaussian blur applied to input frames in pixels, 0 disables blurring
        preview_step: accumulate events over this many frames before saving a visualization preview. If the
            number of input frames is not a multiple of the preview step, the last few frames will be dropped.
            If None, preview is disabled.
        only_preview: if true, only generate the preview and do not generate the full event file, if ``preview_step`` is not set, assume 1.
        force: if true, overwrite output file(s) if present, else throw error
    """
    import json

    import imageio.v3 as iio
    from scipy.ndimage import gaussian_filter

    from visionsim.dataset import Dataset
    from visionsim.emulate.dvs import EventEmulator
    from visionsim.simulate.blender import INDEX_PADDING, ITEMS_PER_SUBFOLDER
    from visionsim.utils.progress import ElapsedProgress

    if input_dir.resolve() == output_dir.resolve():
        raise RuntimeError("Input and output directory cannot be the same!")
    if output_dir.exists() and not force:
        raise FileExistsError("Output directory already exists.")
    else:
        shutil.rmtree(output_dir, ignore_errors=True)

    (output_dir / "frames").mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.txt"

    if pattern:
        dataset = Dataset.from_pattern(input_dir, pattern)
    else:
        dataset = Dataset.from_path(input_dir)

    if fps is None and dataset.cameras:
        framerates = {cam.fps for cam in dataset.cameras}
        if len(framerates) > 1:
            raise ValueError("Multiple cameras with different frame rates found.")
        fps = framerates.pop()
    if fps is None:
        raise ValueError("FPS not provided and could not be inferred from dataset, please specify.")

    if only_preview and preview_step is None:
        preview_step = 1

    emulator_kwargs = {
        "pos_thres": pos_thres,
        "neg_thres": neg_thres,
        "sigma_thres": sigma_thres,
        "cutoff_hz": cutoff_hz,
        "leak_rate_hz": leak_rate_hz,
        "shot_noise_rate_hz": shot_noise_rate_hz,
        "refractory_period_s": refractory_period_s,
        "photoreceptor_noise": photoreceptor_noise,
        "leak_jitter_fraction": leak_jitter_fraction,
        "noise_rate_cov_decades": noise_rate_cov_decades,
        "seed": seed,
        "cs_lambda_pixels": cs_lambda_pixels,
        "cs_tau_p_ms": cs_tau_p_ms,
        "scidvs": scidvs,
    }
    emulator = EventEmulator(**emulator_kwargs)  # type: ignore

    with open(output_dir / "params.json", "w") as f:
        json.dump(emulator_kwargs | {"fps": fps, "blur_sigma": blur_sigma}, f, indent=2)

    with open(events_path, "a+") as out, ElapsedProgress() as progress:
        task = progress.add_task("Writing DVS data...", total=len(dataset))
        viz = None

        for idx, (frame, _) in enumerate(dataset):  # type: ignore
            # Manually grayscale as we've already converted to floating point pixel values
            # Values from http://en.wikipedia.org/wiki/Grayscale
            r, g, b, *_ = np.transpose(frame, (2, 0, 1))
            luma = 0.0722 * b + 0.7152 * g + 0.2126 * r
            if blur_sigma > 0:
                luma = gaussian_filter(luma, sigma=blur_sigma)
            events = emulator.generate_events(luma, idx / int(fps))

            if events is not None:
                events[:, 0] *= 1e6
                rate = len(events) * int(fps) / 1e3

                if not only_preview:
                    np.savetxt(out, events.astype(int), fmt="%d", delimiter=",")

                if preview_step is not None and preview_step > 0:
                    if viz is None:
                        viz = np.ones_like(frame) * 255

                    _, px, py, _ = events[events[:, -1] == 1].T.astype(int)
                    _, nx, ny, _ = events[events[:, -1] == -1].T.astype(int)
                    viz[ny, nx, :3] = [255, 0, 0]
                    viz[py, px, :3] = [0, 0, 255]

                    if (idx + 1) % preview_step == 0:
                        folder_index = f"{idx // ITEMS_PER_SUBFOLDER:04}"
                        frame_index = f"{idx % ITEMS_PER_SUBFOLDER:0{INDEX_PADDING}}.png"
                        outpath = output_dir / "preview" / folder_index / frame_index
                        outpath.parent.mkdir(parents=True, exist_ok=True)
                        iio.imwrite(outpath, viz)
                        viz = None
            else:
                rate = 0

            progress.update(task, description=f"Writing DVS data ({rate:.1f} KEV/s)", advance=1)

    if only_preview:
        events_path.unlink()


def rgb(
    input_dir: Path,
    output_dir: Path,
    chunk_size: int = 10,
    factor: float = 1.0,
    readout_std: float = 20.0,
    fwc: int | None = None,
    duplicate: float = 1.0,
    pattern: str | None = None,
    force: bool = False,
) -> None:
    """Simulate real camera, adding read/poisson noise and tonemapping

    Args:
        input_dir: directory in which to look for frames
        output_dir: directory in which to save binary frames
        chunk_size: number of consecutive frames to average together
        factor: multiply image's linear intensity by this weight
        readout_std: standard deviation of gaussian read noise
        fwc: full well capacity of sensor in arbitrary units (relative to factor & chunk_size)
        duplicate: when chunk size is too small, this model is ill-suited and creates unrealistic noise.
            This parameter artificially increases the chunk size by using each input image ``duplicate`` number of times
        pattern: used to find source image files to convert to rgb frames,
            not needed when ``input_dir`` points to a valid dataset.
        force: if true, overwrite output file(s) if present
    """
    import imageio.v3 as iio
    import more_itertools as mitertools

    from visionsim.dataset import Dataset, Metadata
    from visionsim.emulate.rgb import emulate_rgb_from_sequence
    from visionsim.interpolate.pose import pose_interp
    from visionsim.simulate.blender import INDEX_PADDING, ITEMS_PER_SUBFOLDER
    from visionsim.utils.color import srgb_to_linearrgb
    from visionsim.utils.progress import ElapsedProgress

    if input_dir.resolve() == output_dir.resolve():
        raise RuntimeError("Input and output directory cannot be the same!")
    if output_dir.exists() and not force:
        raise FileExistsError("Output directory already exists.")
    else:
        shutil.rmtree(output_dir, ignore_errors=True)

    if pattern:
        dataset = Dataset.from_pattern(input_dir, pattern)
    else:
        dataset = Dataset.from_path(input_dir)

        if dataset.cameras is None or len(dataset.cameras) != 1:
            raise NotImplementedError("Cannot emulate an RGB camera from multiple cameras.")
    transforms = []

    with ElapsedProgress() as progress:
        task = progress.add_task("Writing RGB frames", total=len(dataset))
        for i, batch in enumerate(mitertools.ichunked(dataset, chunk_size)):
            folder_index = f"{i // ITEMS_PER_SUBFOLDER:04}"
            frame_index = f"{i % ITEMS_PER_SUBFOLDER:0{INDEX_PADDING}}.png"
            outpath = output_dir / folder_index / frame_index

            # Batch is an iterable of (data, transforms) that we need to reduce
            imgs_iter, transforms_iter = mitertools.unzip(batch)
            imgs = np.array([(i.astype(float) / 255.0).astype(float) for i in imgs_iter])

            # Assume images have been tonemapped and undo mapping
            imgs = srgb_to_linearrgb(imgs)

            rgb_img = emulate_rgb_from_sequence(
                imgs * duplicate,
                readout_std=readout_std,
                fwc=fwc or (chunk_size * duplicate),
                factor=factor,
            )

            if not pattern:
                # We checked that there's only a single camera, just re-use any transforms dict
                (transform, *_), transforms_iter = mitertools.spy(transforms_iter)
                poses = np.array([t["transform_matrix"] for t in transforms_iter])
                transform["transform_matrix"] = pose_interp(poses, k=np.clip(len(poses) - 1, 2, 3))(0.5)
                transform["file_path"] = outpath.relative_to(output_dir)
                transforms.append(transform)

            # TODO: Alpha and grayscale?
            # if rgb_img.shape[-1] == 1:
            #     rgb_img = np.repeat(rgb_img, 3, axis=-1)

            outpath.parent.mkdir(exist_ok=True, parents=True)
            iio.imwrite(outpath, (rgb_img * 255).astype(np.uint8))
            progress.update(task, advance=chunk_size)

    if not pattern:
        Metadata.from_dense_transforms(transforms).save(output_dir / "transforms.json")


def imu(
    input_dir: Path,
    output_file: Path | None = None,
    seed: int = 2147483647,
    gravity: str = "(0.0, 0.0, -9.8)",
    dt: float = 0.00125,
    init_bias_acc: str = "(0.0, 0.0, 0.0)",
    init_bias_gyro: str = "(0.0, 0.0, 0.0)",
    std_bias_acc: float = 5.5e-5,
    std_bias_gyro: float = 2e-5,
    std_acc: float = 8e-3,
    std_gyro: float = 1.2e-3,
    force: bool = False,
) -> None:
    """Simulate data from a co-located IMU using the poses in a ``transforms.json`` or ``transforms.db`` file.

    Args:
        input_dir: directory in which to look for transforms,
        output_file: file in which to save simulated IMU data. Prints to stdout if omitted.
        seed: RNG seed value for reproducibility.
        gravity: gravity vector in world coordinate frame. Given in m/s^2.
        dt: time between consecutive transforms.json poses (assumed regularly spaced). Given in seconds.
        init_bias_acc: initial bias/drift in accelerometer reading. Given in m/s^2.
        init_bias_gyro: initial bias/drift in gyroscope reading. Given in rad/s.
        std_bias_acc: stdev for random-walk component of error (drift) in accelerometer. Given in m/(s^3 sqrt(Hz))
        std_bias_gyro: stdev for random-walk component of error (drift) in gyroscope. Given in rad/(s^2 sqrt(Hz))
        std_acc: stdev for white-noise component of error in accelerometer. Given in m/(s^2 sqrt(Hz))
        std_gyro: stdev for white-noise component of error in gyroscope. Given in rad/(s sqrt(Hz))
        force: if true, overwrite output file(s) if present
    """

    import ast
    import sys

    from visionsim.dataset import Metadata
    from visionsim.emulate.imu import emulate_imu

    if output_file and output_file.exists() and not force:
        raise FileExistsError("Output file already exists.")

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(seed))
    gravity_ = np.array(ast.literal_eval(gravity))
    init_bias_acc_ = np.array(ast.literal_eval(init_bias_acc))
    init_bias_gyro_ = np.array(ast.literal_eval(init_bias_gyro))
    poses = Metadata.from_path(input_dir).poses

    data_gen = emulate_imu(
        poses,
        dt=dt,
        std_acc=std_acc,
        std_gyro=std_gyro,
        std_bias_acc=std_bias_acc,
        std_bias_gyro=std_bias_gyro,
        init_bias_acc=init_bias_acc_,
        init_bias_gyro=init_bias_gyro_,
        gravity=gravity_,
        rng=rng,
    )

    with open(output_file, "w") if output_file else sys.stdout as out:
        out.write("t,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,bias_ax,bias_ay,bias_az,bias_gx,bias_gy,bias_gz\n")
        for d in data_gen:
            out.write(
                "{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    d["t"], *d["acc_reading"], *d["gyro_reading"], *d["acc_bias"], *d["gyro_bias"]
                )
            )







def aspc(
    input_dir: Path,
    output_dir: Path,
    config_file: Path | None = None,
    min_depth: float | None = None,
    max_depth: float | None = None,
    n_bins: int | None = None,
    bin_width: float | None = None,
    pixel_fov_list: Path | None = None,
    vignette: bool | None = None,
    n_pulses: int | None = None,
    dead_time: float | None = None,
    free_running: bool | None = None,
    fast_sim: bool | None = None,
    active_enabled: bool | None = None,
    active_wavelength: float | None = None,
    active_pulse_repetition: float | None = None,
    active_pulse_width: float | None = None,
    active_avg_watts: float | None = None,
    active_pulse_shape: Literal["gaussian", "square"] | Path | None = None,
    ambient_enabled: bool | None = None,
    ambient_temperature: float | None = None,
    ambient_light_conditions: (
        Literal[
            "BRIGHTEST_SUNLIGHT",
            "BRIGHT_SUNLIGHT",
            "AVERAGE_SUNLIGHT",
            "BRIGHT_SHADE",
            "OVERCAST",
            "SUNSET",
            "SUNRISE",
            "STORM_OVERCAST",
            "OVERCAST_SUNSET",
            "OVERCAST_SUNRISE",
            "FULL_MOON",
            "QUARTER_MOON",
            "STARLIGHT_WITH_AIRGLOW",
            "STARLIGHT_WITHOUT_AIRGLOW",
        ]
        | None
    ) = None,
    ambient_lambda_pass: float | None = None,
    ambient_delta_lambda: float | None = None,
    ambient_intensity: float | None = None,
    sensor_size: tuple | None = None,
    sensor_pixel_pitch: float | None = None,
    sensor_f_number: float | None = None,
    sensor_fov: tuple | None = None,
    force: bool | None = None,
    pattern: str | None = None,
) -> None:
    """Simulate Single-Photon (SP) LiDAR data from rendered frames and depth maps.

    Args:
        input_dir: directory containing the input rendered frames and depth maps.
        output_dir: directory in which to save the simulated SP LiDAR data.
        config_file: path to an optional YAML configuration file for the simulation parameters.
        min_depth: minimum resolvable depth for the histogram. Given in meters.
        max_depth: maximum resolvable depth for the histogram. Given in meters.
        n_bins: total number of temporal bins in the histogram.
        bin_width: spatial width of each individual histogram bin. Given in meters.
        pixel_fov_list: path to specific field of view configurations for individual pixels or macro-pixels.
        vignette: vignetting effect applied to the sensor's optical system.
        n_pulses: number of laser pulses integrated per measurement/histogram.
        dead_time: recovery time required by the SPAD after detecting a photon before it can detect another. Given in nanoseconds.
        free_running: if true, indicates the SPAD operates in a free-running mode rather than a gated mode.
        fast_sim: if true, enables a computationally faster, approximate simulation mode.
        active_enabled: if true, toggles the active pulsed laser source on.
        active_wavelength: center wavelength of the emitted laser pulse. Given in nanometers.
        active_pulse_repetition: repetition rate of the laser pulses. Given in megahertz.
        active_pulse_width: temporal duration (often FWHM) of a single laser pulse. Given in nanoseconds.
        active_avg_watts: average power output of the pulsed laser. Given in watts.
        active_pulse_shape: the temporal profile of the laser pulse (e.g., "gaussian", "square") or path to a custom shape.
        ambient_enabled: if true, toggles the ambient light source (e.g., the sun) on.
        ambient_temperature: blackbody color temperature of the ambient light source. Given in kelvin.
        ambient_light_conditions: descriptor for the environment's lighting (e.g., "BRIGHT_SUNLIGHT", "OVERCAST").
        ambient_lambda_pass: center wavelength of the receiver's optical bandpass filter. Given in nanometers.
        ambient_delta_lambda: bandwidth of the receiver's optical bandpass filter. Given in nanometers.
        ambient_intensity: irradiance of the ambient light source. Given in watts / meter**2.
        sensor_size: pixel dimensions of the SPAD array (e.g., [height, width]).
        sensor_pixel_pitch: physical distance between the centers of adjacent pixels. Given in micrometer.
        sensor_f_number: the F-number (focal ratio) determining the optical aperture of the sensor.
        sensor_fov: the angular field of view of the camera sensor. Given in degree.
        force: if true, overwrite output file(s) or directory if present.
        pattern: glob pattern to filter which files to process in the input directory.
    """
    user_args = locals().copy()

    from visionsim.dataset import Dataset
    from visionsim.emulate.aspc import ASPCEmulator
    from visionsim.emulate.aspc.utils import ureg

    # Built-in default configuration initialized with proper Pint Quantity objects
    DEFAULT_CONFIG = {
        "histogrammer": {
            "min_depth": 0 * ureg.meters,
            "max_depth": 10 * ureg.meters,
            "n_bins": 672,
            "bin_width": 0.044 * ureg.meters,
            "pixel_fov_list": [[0, 0.4, 0.3, 0.6], [0.2, 0.6, 0.6, 0.9]],
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

    # 1. Map CLI parameter names to their nested YAML config hierarchy
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

    # 2. Map CLI parameters to their default Pint unit strings
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

    def _to_quantity(val, unit_str):
        if val is None:
            return None

        if isinstance(val, ureg.Quantity):
            # If a single Quantity wraps an array/tuple, convert it to a list of Quantities
            if hasattr(val.magnitude, "__len__"):
                return [ureg.Quantity(x, val.units) for x in val.magnitude]
            return val

        if isinstance(val, (tuple, list)):
            return [
                x if isinstance(x, ureg.Quantity) else ureg.Quantity(x, unit_str)
                for x in val
            ]

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

    # 3. Build overrides dict ONLY for parameters explicitly passed by caller
    config_overrides = {}

    for param_name, key_path in cli_mapping.items():
        val = user_args.get(param_name)
        if val is not None:
            if param_name in cli_param_units:
                val = _to_quantity(val, cli_param_units[param_name])

            d = config_overrides
            for step in key_path[:-1]:
                d = d.setdefault(step, {})
            d[key_path[-1]] = val
    
    # 4. Output directory setup
    if input_dir.resolve() == output_dir.resolve():
        raise RuntimeError("Input and output directory cannot be the same!")

    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Output directory '{output_dir}' already exists. Use force=True to overwrite."
            )
        shutil.rmtree(output_dir, ignore_errors=True)

    (output_dir / "frames").mkdir(parents=True, exist_ok=True)

    # 5. Load Dataset
    if pattern:
        depths = Dataset.from_pattern(input_dir / "depths" / pattern)
        albedo = Dataset.from_pattern(input_dir / "frames" / pattern)
    else:
        depths = Dataset.from_path(input_dir / "depths")
        albedo = Dataset.from_path(input_dir / "frames")


    
    # 6. Fallback: use config_file if given, otherwise use DEFAULT_CONFIG
    base_config = config_file if config_file is not None else DEFAULT_CONFIG

    emulator = ASPCEmulator(
        base_cfg=base_config,
        config_overrides=config_overrides,
    )

    total_frames = len(depths)

    # 7. Frame processing loop
    import torch

    from visionsim.utils.progress import ElapsedProgress
    def _prepare_frame_quantity(frame, default_unit):
        """Converts PyTorch Tensors or arrays to 2D NumPy arrays with Pint units."""
        # 1. Convert PyTorch tensor or generic sequence to NumPy

        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()

        # 2. Ensure (H, W) 2D format
        if frame.ndim != 2:
            raise ValueError(f"Expected frame shape (H, W) 2D array, got shape {frame.shape}")
        # Conver to torch tensor
        frame = torch.from_numpy(frame.copy()).to(device="cpu",dtype=torch.float64)

        # 3. Attach Pint units if not already attached
        if not hasattr(frame, "units"):
            frame = ureg.Quantity(frame, default_unit)

        return frame
    with ElapsedProgress() as progress:
        task = progress.add_task(
            "[cyan]Generating ASPC histograms...", total=total_frames
        )

        for frame_idx, (depth_frame, albedo_frame) in enumerate(
            zip(depths, albedo)
        ):
            depth_frame = depth_frame[0].squeeze()

            albedo_frame = albedo_frame[0][:,:,0]
            
            # Format frames as 2D NumPy arrays wrapped in Pint Quantities
            depth_qty = _prepare_frame_quantity(depth_frame, "meter")
            albedo_qty = _prepare_frame_quantity(albedo_frame, "dimensionless")
           

            histogram_data = emulator.process_frame(depth_qty, albedo_qty)

            output_file = output_dir / "frames" / f"histogram_{frame_idx:04d}.npy"
            np.save(output_file, histogram_data)

            progress.advance(task, 1)