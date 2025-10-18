from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np
from typing_extensions import Literal

from visionsim.emulate.rgb import emulate_rgb_from_sequence


def _spad_collate(batch, *, mode, rng, factor, is_tonemapped=True):
    """Use default collate function on batch and then simulate SPAD, enabling compute to be done in threads"""
    from visionsim.dataset import default_collate
    from visionsim.emulate.spc import emulate_spc
    from visionsim.utils.color import srgb_to_linearrgb

    idxs, imgs, poses = default_collate(batch)

    if is_tonemapped:
        # Image has been tonemapped so undo mapping
        imgs = srgb_to_linearrgb((imgs / 255.0).astype(float))
    else:
        imgs = imgs.astype(float) / 255.0

    binary_img = emulate_spc(imgs, factor=factor, rng=rng) * 255
    binary_img = binary_img.astype(np.uint8)

    if mode.lower() == "npy":
        binary_img = binary_img >= 128
        binary_img = np.packbits(binary_img, axis=2)
    return idxs, binary_img, poses


def spad(
    input_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    pattern: str = "frame_{:06}.png",
    factor: float = 1.0,
    seed: int = 2147483647,
    mode: Literal["npy", "img"] = "npy",
    batch_size: int = 4,
    force: bool = False,
):
    """Perform bernoulli sampling on linearized RGB frames to yield binary frames

    Args:
        input_dir: directory in which to look for frames
        output_dir: directory in which to save binary frames
        pattern: filenames of frames should match this
        factor: multiplicative factor controlling dynamic range of output
        seed: random seed to use while sampling, ensures reproducibility
        mode: how to save binary frames
        batch_size: number of frames to write at once
        force: if true, overwrite output file(s) if present
    """
    import copy

    from rich.progress import Progress
    from torch.utils.data import DataLoader

    from visionsim.dataset import Dataset, ImgDatasetWriter, NpyDatasetWriter

    from . import _validate_directories

    input_path, output_path, *_ = _validate_directories(input_dir, output_dir)
    dataset = Dataset.from_path(input_path)
    transforms_new = copy.deepcopy(dataset.transforms or {})
    shape = np.array(dataset.full_shape)
    shape[-1] = transforms_new["c"] = 3

    if mode.lower() == "img":
        ...
    elif mode.lower() == "npy":
        # Default to bitpacking width
        transforms_new["bitpack"] = True
        transforms_new["bitpack_dim"] = 2
        shape[2] /= 8
    else:
        raise ValueError(f"Mode should be one of 'img' or 'npy', got {mode}.")

    is_tonemapped = all(not str(p).endswith(".exr") for p in getattr(dataset, "paths", []))

    rng = np.random.default_rng(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=os.cpu_count() or 1,
        collate_fn=functools.partial(_spad_collate, mode=mode, rng=rng, factor=factor, is_tonemapped=is_tonemapped),
    )

    with (
        ImgDatasetWriter(output_path, transforms=transforms_new, force=force, pattern=pattern)
        if mode.lower() == "img"
        else NpyDatasetWriter(output_path, np.ceil(shape).astype(int), transforms=transforms_new, force=force) as writer,
        Progress() as progress,
    ):
        task1 = progress.add_task("Writing SPAD frames", total=len(dataset))
        for idxs, imgs, poses in loader:
            writer[idxs] = (imgs, poses)
            progress.update(task1, advance=len(idxs))


def events(
    input_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    fps: int,
    pos_thres: float = 0.2,
    neg_thres: float = 0.2,
    sigma_thres: float = 0.03,
    cutoff_hz: int = 200,
    leak_rate_hz: float = 1.0,
    shot_noise_rate_hz: float = 10.0,
    seed: int = 2147483647,
    force: bool = False,
):
    """Emulate an event camera using v2e and high speed input frames

    Args:
        input_dir: directory in which to look for frames
        output_dir: directory in which to save events
        fps: frame rate of input sequence
        pos_thres: nominal threshold of triggering positive event in log intensity
        neg_thres: nominal threshold of triggering negative event in log intensity
        sigma_thres: std deviation of threshold in log intensity
        cutoff_hz: 3dB cutoff frequency in Hz of DVS photoreceptor, default: 200,
        leak_rate_hz: leak event rate per pixel in Hz, from junction leakage in reset switch
        shot_noise_rate_hz: shot noise rate in Hz
        seed: random seed to use while sampling, ensures reproducibility
        force: if true, overwrite output file(s) if present
    """
    import json

    import imageio.v3 as iio
    from rich.progress import Progress

    from visionsim.dataset import Dataset
    from visionsim.emulate.dvs import EventEmulator

    from . import _validate_directories

    input_path, output_path, *_ = _validate_directories(input_dir, output_dir)
    (output_path / "frames").mkdir(parents=True, exist_ok=True)
    events_path = output_path / "events.txt"
    dataset = Dataset.from_path(input_path)

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

    if events_path.exists():
        if force:
            events_path.unlink()
        else:
            raise FileExistsError(f"Event file already exists in {output_path}")

    with open(output_path / "params.json", "w") as f:
        json.dump(emulator_kwargs | dict(fps=fps), f, indent=2)

    with open(events_path, "a+") as out, Progress() as progress:
        task = progress.add_task("Processing @ N/A KEV/s", total=len(dataset))

        for idx, frame, _ in dataset:  # type: ignore
            # Manually grayscale as we've already converted to floating point pixel values
            # Values from http://en.wikipedia.org/wiki/Grayscale
            r, g, b, *_ = np.transpose(frame, (2, 0, 1))
            luma = 0.0722 * b + 0.7152 * g + 0.2126 * r
            events = emulator.generate_events(luma, idx / int(fps))

            if events is not None:
                events[:, 0] *= 1e6
                np.savetxt(out, events.astype(int), fmt="%d")
                rate = len(events) * int(fps) / 1e3

                viz = np.ones_like(frame) * 255
                _, px, py, _ = events[events[:, -1] == 1].T.astype(int)
                _, nx, ny, _ = events[events[:, -1] == -1].T.astype(int)
                viz[ny, nx, :3] = [255, 0, 0]
                viz[py, px, :3] = [0, 0, 255]
                iio.imwrite(output_path / "frames" / f"event_{idx:06}.png", viz)
            else:
                rate = 0

            progress.update(task, description=f"Processing @ {rate:.1f} KEV/s", advance=1)


def rgb(
    input_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    chunk_size: int = 10,
    factor: float = 1.0,
    readout_std: float = 20.0,
    fwc: int | None = None,
    duplicate: float = 1.0,
    pattern: str = "frame_{:06}.png",
    mode: Literal["npy", "img"] = "npy",
    force: bool = False,
):
    """Simulate real camera, adding read/poisson noise and tonemapping

    Args:
        input_dir: directory in which to look for frames
        output_dir: directory in which to save binary frames
        chunk_size: number of consecutive frames to average together
        factor: multiply image's linear intensity by this weight
        readout_std: standard deviation of gaussian read noise
        fwc: full well capacity of sensor in arbitrary units (relative to factor & chunk_size)
        duplicate: when chunk size is too small, this model is ill-suited and creates unrealistic noise. This parameter artificially increases the chunk size by using each input image `duplicate` number of times
        pattern: filenames of frames should match this
        mode: how to save binary frames
        force: if true, overwrite output file(s) if present
    """
    import copy

    import more_itertools as mitertools
    from rich.progress import Progress
    from torch.utils.data import DataLoader

    from visionsim.dataset import Dataset, ImgDatasetWriter, NpyDatasetWriter, default_collate
    from visionsim.interpolate import pose_interp
    from visionsim.utils.color import srgb_to_linearrgb

    from . import _validate_directories

    input_path, output_path, *_ = _validate_directories(input_dir, output_dir)
    dataset = Dataset.from_path(input_path)
    transforms_new = copy.deepcopy(dataset.transforms or {})
    shape = np.array(dataset.full_shape)
    shape[-1] = transforms_new["c"] = 3
    shape[0] = np.ceil(shape[0] / chunk_size).astype(int)
    transforms_new = transforms_new if dataset.transforms else {}

    if mode.lower() not in ("img", "npy"):
        raise ValueError(f"Mode should be one of 'img' or 'npy', got {mode}.")

    if any(str(p).endswith(".exr") for p in getattr(dataset, "paths", [])):
        # TODO: This is due to the alpha blending below, we need alpha in [0, 1] to blend.
        raise NotImplementedError("Task does not yet support EXRs")

    loader = DataLoader(dataset, batch_size=1, num_workers=os.cpu_count() or 1, collate_fn=default_collate)

    with (
        ImgDatasetWriter(output_path, transforms=transforms_new, force=force, pattern=pattern)
        if mode.lower() == "img"
        else NpyDatasetWriter(output_path, np.ceil(shape).astype(int), transforms=transforms_new, force=force) as writer,
        Progress() as progress,
    ):
        task = progress.add_task("Writing RGB frames", total=len(dataset))
        for i, batch in enumerate(mitertools.ichunked(loader, chunk_size)):
            # Batch is an iterable of (idx, img, pose) that we need to reduce
            idxs_iter, imgs_iter, poses_iter = mitertools.unzip(batch)
            imgs = np.array([(i.astype(float) / 255.0).astype(float) for i in imgs_iter])
            idxs, poses = np.concatenate(list(idxs_iter)), np.concatenate(list(poses_iter))

            # Assume images have been tonemapped and undo mapping
            imgs = srgb_to_linearrgb(imgs)

            rgb_img = emulate_rgb_from_sequence(
                imgs * duplicate,
                readout_std=readout_std,
                fwc=fwc or (chunk_size * duplicate),
                factor=factor,
            )
            pose = pose_interp(poses)(0.5) if transforms_new else None

            if rgb_img.shape[-1] == 1:
                rgb_img = np.repeat(rgb_img, 3, axis=-1)

            writer[i] = ((rgb_img * 255).astype(np.uint8), pose)
            progress.update(task, advance=len(idxs))


def imu(
    input_dir: str | os.PathLike = ".",
    output_file: str | os.PathLike = "",
    seed: int = 2147483647,
    gravity: str = "(0.0, 0.0, -9.8)",
    dt: float = 0.00125,
    init_bias_acc: str = "(0.0, 0.0, 0.0)",
    init_bias_gyro: str = "(0.0, 0.0, 0.0)",
    std_bias_acc: float = 5.5e-5,
    std_bias_gyro: float = 2e-5,
    std_acc: float = 8e-3,
    std_gyro: float = 1.2e-3,
):
    """Simulate data from a co-located IMU using the poses in transforms.json.

    Args:
        input_dir: directory in which to look for transforms.json,
        output_file: file in which to save simulated IMU data. Prints to stdout if empty. default: '',
        seed: RNG seed value for reproducibility. default: 2147483647,
        gravity: gravity vector in world coordinate frame. Given in m/s^2. default: [0,0,-9.8],
        dt: time between consecutive transforms.json poses (assumed regularly spaced). Given in seconds. default: 0.00125,
        init_bias_acc: initial bias/drift in accelerometer reading. Given in m/s^2. default: [0,0,0],
        init_bias_gyro: initial bias/drift in gyroscope reading. Given in rad/s. default: [0,0,0],
        std_bias_acc: stdev for random-walk component of error (drift) in accelerometer. Given in m/(s^3 sqrt(Hz))
        std_bias_gyro: stdev for random-walk component of error (drift) in gyroscope. Given in rad/(s^2 sqrt(Hz))
        std_acc: stdev for white-noise component of error in accelerometer. Given in m/(s^2 sqrt(Hz))
        std_gyro: stdev for white-noise component of error in gyroscope. Given in rad/(s sqrt(Hz))
    """

    import ast
    import sys

    from visionsim.dataset import Dataset

    if not Path(input_dir).resolve().exists():
        raise RuntimeError("Input directory path doesn't exist!")
    dataset = Dataset.from_path(input_dir)
    if dataset.transforms is None:
        raise RuntimeError("dataset.transforms not found!")

    rng = np.random.default_rng(int(seed))
    gravity_ = np.array(ast.literal_eval(gravity))
    init_bias_acc_ = np.array(ast.literal_eval(init_bias_acc))
    init_bias_gyro_ = np.array(ast.literal_eval(init_bias_gyro))

    from visionsim.emulate.imu import emulate_imu

    data_gen = emulate_imu(
        dataset.poses,
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
