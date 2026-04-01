from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np


def spad(
    input_dir: Path,
    output_dir: Path,
    flux_gain: float = 1.0,
    bitplanes: int = 1,
    bitdepth: int | None = None,
    force_gray: bool = False,
    seed: int = 2147483647,
    pattern: str | None = None,
    max_size: int = 1000,
    force: bool = False,
) -> None:
    """Perform binomial sampling on linearized RGB frames to yield (summed) single photon frames

    This will save numpy files which may be bitpacked (when bitplanes == 1) and may have different dtypes
    depending on the number of summed bitplanes. The shape of the output arrays will be (max_size, h, w, c) or (remainder, h, w, c)
    where remainder = len(dataset) % max_size, where the width dimension is ceil(width / 8) when bitpacked.

    If the input contains alpha channel (determined by the last dimension of the input images), it will be stripped.

    Args:
        input_dir: directory in which to look for frames
        output_dir: directory in which to save single photon frames
        pattern: used to find source image files to convert to single photon frames,
            not needed when ``input_dir`` points to a valid dataset.
        flux_gain: multiplicative factor controlling dynamic range of output
        bitplanes: number of summed binary measurements
        bitdepth: if set, ``bitplanes`` will be overridden to ``2**bitdepth - 1``
        force_gray: to disable RGB sensing even if the input images are color
        seed: random seed to use while sampling, ensures reproducibility
        max_size: maximum number of frames per output array before rolling over to new file
        force: if true, overwrite output file(s) if present, else throw error
    """
    from numpy.lib.format import open_memmap

    from visionsim.cli import _log, _log_once
    from visionsim.dataset import Dataset, Metadata
    from visionsim.emulate.spc import emulate_spc
    from visionsim.utils.color import rgb_to_grayscale, srgb_to_linearrgb
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

    if bitdepth is not None:
        _log.info(f"Overriding bitplanes to {2**bitdepth - 1} since bitdepth is set to {bitdepth}.")
        bitplanes = 2**bitdepth - 1

    # Map bitplanes to the smallest uint type that can hold it (minimum 8 bits)
    out_dtype = next(
        dtype
        for limit, dtype in [(8, np.uint8), (16, np.uint16), (32, np.uint32), (64, np.uint64)]
        if bitplanes <= 2**limit - 1
    )

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

            if len(data.shape) == 3 and data.shape[-1] in (2, 4):  # LA/RGBA
                _log_once(data.shape, "Alpha channel detected, ignoring it.", "info")
                data = data[..., :-1]

            if force_gray:
                data = rgb_to_grayscale(data)

            imgs = emulate_spc(data, flux_gain=flux_gain, bitplanes=bitplanes, rng=rng)

            offset = i % max_size
            file_path = output_dir / f"{i // max_size:04}.npy"
            transform["file_path"] = file_path.name
            transform["bitplanes"] = bitplanes
            transform["offset"] = offset
            h, w, c = data.shape

            if bitplanes == 1:
                # Default to bitpacking width
                imgs = imgs >= 0.5
                imgs = np.packbits(imgs, axis=1)
                transform["bitpack_dim"] = 2
                w = math.ceil(transform.get("w", w) / 8)
            else:
                w = transform.get("w", w)

            if not file_path.exists():
                data = open_memmap(
                    file_path,
                    mode="w+",
                    dtype=out_dtype,
                    shape=(min(max_size, remainder), transform.get("h", h), w, c),
                )
                data[offset] = imgs
            else:
                open_memmap(file_path)[offset] = imgs

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
    cutoff_hz: int = 200,
    leak_rate_hz: float = 1.0,
    shot_noise_rate_hz: float = 10.0,
    seed: int = 2147483647,
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
        seed: random seed to use while sampling, ensures reproducibility
        preview_step: accumulate events over this many frames before saving a visualization preview. If the
            number of input frames is not a multiple of the preview step, the last few frames will be dropped.
            If None, preview is disabled.
        only_preview: if true, only generate the preview and do not generate the full event file, if `preview_step` is not set, assume 1.
        force: if true, overwrite output file(s) if present, else throw error
    """
    import json

    import imageio.v3 as iio

    from visionsim.dataset import Dataset
    from visionsim.emulate.dvs import EventEmulator
    from visionsim.simulate.blender import INDEX_PADDING, ITEMS_PER_SUBFOLDER
    from visionsim.utils.color import rgb_to_grayscale
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

    if fps is None:
        if dataset.cameras:
            framerates = set(cam.fps for cam in dataset.cameras)
            if len(framerates) > 1:
                raise ValueError("Multiple cameras with different frame rates found.")
            fps = framerates.pop()
    if fps is None:
        raise ValueError("FPS not provided and could not be inferred from dataset, please specify.")

    if only_preview and preview_step is None:
        preview_step = 1

    emulator_kwargs = dict(
        pos_thres=pos_thres,
        neg_thres=neg_thres,
        sigma_thres=sigma_thres,
        cutoff_hz=cutoff_hz,
        leak_rate_hz=leak_rate_hz,
        shot_noise_rate_hz=shot_noise_rate_hz,
        seed=seed,
    )
    emulator = EventEmulator(**emulator_kwargs)  # type: ignore

    with open(output_dir / "params.json", "w") as f:
        json.dump(emulator_kwargs | dict(fps=fps), f, indent=2)

    with open(events_path, "a+") as out, ElapsedProgress() as progress:
        task = progress.add_task("Writing DVS data...", total=len(dataset))
        viz = None

        for idx, (frame, _) in enumerate(dataset):  # type: ignore
            luma = rgb_to_grayscale(frame)
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
    shutter_frac: float = 1.0,
    readout_std: float = 16.0,
    fwc: float | None = None,
    flux_gain: float = 2.0**12,
    sensor_gain: float = 1.0,
    adc_bitdepth: int = 12,
    mosaic: bool = False,
    demosaic: Literal["off", "bilinear", "MHC04"] = "MHC04",
    denoise_sigma: float = 0.0,
    sharpen_weight: float = 0.0,
    pattern: str | None = None,
    force: bool = False,
) -> None:
    """Simulate real camera, adding read/poisson noise and tonemapping

    For more information on the camera model, see :func:`emulate_rgb_from_sequence 
    <visionsim.emulate.rgb.emulate_rgb_from_sequence>`.

    Args:
        input_dir: directory in which to look for frames
        output_dir: directory in which to save binary frames
        chunk_size: number of consecutive frames to average together
        shutter_frac: fraction of inter-frame duration shutter is active (0 to 1)
        readout_std: standard deviation of gaussian read noise in photoelectrons
        fwc: full well capacity of sensor in photoelectrons
        flux_gain: factor to scale the input images before Poisson simulation
        sensor_gain: gain for photo-electron reading after Poisson rng (in units of electrons per digital level)
        adc_bitdepth: ADC bitdepth, typically 12 or 14
        mosaic: implement mosaiced R-/G-/B- pixels or an innately 3-channel sensor
        demosaic: demosaicing method (default Malvar et al.'s method)
        denoise_sigma: Gaussian blur with this sigma will be used (default 0.0 disables this)
        sharpen_weight: weight used in sharpening (default 0.0 disables this)
        pattern: used to find source image files to convert to rgb frames,
            not needed when ``input_dir`` points to a valid dataset.
        force: if true, overwrite output file(s) if present
    """
    import imageio.v3 as iio
    import more_itertools as mitertools

    from visionsim.cli import _log_once
    from visionsim.dataset import Dataset, Metadata
    from visionsim.emulate.rgb import emulate_rgb_from_sequence
    from visionsim.interpolate.pose import pose_interp
    from visionsim.simulate.blender import INDEX_PADDING, ITEMS_PER_SUBFOLDER
    from visionsim.utils.color import linearrgb_to_srgb, srgb_to_linearrgb
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

            if len(imgs.shape) == 4 and imgs.shape[-1] in (2, 4):  # LA/RGBA
                _log_once(imgs.shape, "Alpha channel detected, ignoring it.", "info")
                imgs = imgs[..., :-1]

            rgb_img = emulate_rgb_from_sequence(
                imgs,
                readout_std=readout_std,
                fwc=fwc or np.inf,
                shutter_frac=shutter_frac,
                flux_gain=flux_gain,
                sensor_gain=sensor_gain,
                adc_bitdepth=adc_bitdepth,
                mosaic=mosaic,
                demosaic=demosaic,
                denoise_sigma=denoise_sigma,
                sharpen_weight=sharpen_weight,
            )

            if not pattern:
                # We checked that there's only a single camera, just re-use any transforms dict
                (transform, *_), transforms_iter = mitertools.spy(transforms_iter)
                poses = np.array([t["transform_matrix"] for t in transforms_iter])

                if len(poses) > 1:
                    transform["transform_matrix"] = pose_interp(poses, k=min(len(poses) - 1, 3))(0.5)
                else:
                    transform["transform_matrix"] = poses[0]

                transform["file_path"] = outpath.relative_to(output_dir)
                transforms.append(transform)

            outpath.parent.mkdir(exist_ok=True, parents=True)
            iio.imwrite(outpath, (linearrgb_to_srgb(rgb_img) * 255).astype(np.uint8))
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
