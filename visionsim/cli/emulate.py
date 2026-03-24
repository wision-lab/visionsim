from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

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
            # Manually grayscale as we've already converted to floating point pixel values
            # Values from http://en.wikipedia.org/wiki/Grayscale
            r, g, b, *_ = np.transpose(frame, (2, 0, 1))
            luma = 0.0722 * b + 0.7152 * g + 0.2126 * r
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

    if output_file and not force:
        raise FileExistsError("Output file already exists.")

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
