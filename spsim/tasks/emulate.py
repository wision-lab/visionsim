import functools

import numpy as np
from invoke import task

from spsim.emulate.rgb import emulate_rgb_from_sequence


def _spad_collate(batch, *, mode, rng, factor, is_tonemapped=True):
    """Use default collate function on batch and then simulate SPAD, enabling compute to be done in threads"""
    from spsim.dataset import default_collate
    from spsim.emulate.spc import emulate_spc
    from spsim.utils.color import srgb_to_linearrgb

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


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save binary frames",
        "pattern": "filenames of frames should match this, default: 'frame_{:06}.png'",
        "factor": "multiplicative factor controlling dynamic range of output, default: 1.0",
        "seed": "random seed to use while sampling, ensures reproducibility. default: 2147483647",
        "mode": "how to save binary frames, either as 'img' or as 'npy', default: 'npy'",
        "batch_size": "number of frames to write at once, default: 4",
        "force": "if true, overwrite output file(s) if present, default: False",
    }
)
def spad(
    c,
    input_dir,
    output_dir,
    pattern="frame_{:06}.png",
    factor=1.0,
    seed=2147483647,
    mode="npy",
    batch_size=4,
    force=False,
):
    """Perform bernoulli sampling on linearized RGB frames to yield binary frames"""
    import copy

    from rich.progress import Progress
    from torch.utils.data import DataLoader

    from spsim.dataset import Dataset, ImgDatasetWriter, NpyDatasetWriter

    from .common import _validate_directories

    input_dir, output_dir = _validate_directories(input_dir, output_dir)
    dataset = Dataset.from_path(input_dir)
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
        num_workers=c.get("max_threads"),
        collate_fn=functools.partial(_spad_collate, mode=mode, rng=rng, factor=factor, is_tonemapped=is_tonemapped),
    )

    with (
        ImgDatasetWriter(output_dir, transforms=transforms_new, force=force, pattern=pattern)
        if mode.lower() == "img"
        else NpyDatasetWriter(output_dir, np.ceil(shape).astype(int), transforms=transforms_new, force=force) as writer,
        Progress() as progress,
    ):
        task1 = progress.add_task("Writing SPAD frames", total=len(dataset))
        for idxs, imgs, poses in loader:
            writer[idxs] = (imgs, poses)
            progress.update(task1, advance=len(idxs))


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save events",
        "fps": "frame rate of input sequence",
        "pos_thres": "nominal threshold of triggering positive event in log intensity, default: 0.2",
        "neg_thres": "nominal threshold of triggering negative event in log intensity, default: 0.2",
        "sigma_thres": "std deviation of threshold in log intensity, default: 0.03",
        "cutoff_hz": "3dB cutoff frequency in Hz of DVS photoreceptor, default: 200",
        "leak_rate_hz": "leak event rate per pixel in Hz, from junction leakage in reset switch, default: 1",
        "shot_noise_rate_hz": "shot noise rate in Hz, default: 10",
        "seed": "random seed to use while sampling, ensures reproducibility, default: 2147483647",
        "force": "if true, overwrite output file(s) if present, default: False",
    }
)
def events(
    _,
    input_dir,
    output_dir,
    fps,
    pos_thres=0.2,
    neg_thres=0.2,
    sigma_thres=0.03,
    cutoff_hz=200,
    leak_rate_hz=1.0,
    shot_noise_rate_hz=10.0,
    seed=2147483647,
    force=False,
):
    """Emulate an event camera using v2e and high speed input frames"""
    import json

    import imageio.v3 as iio
    from rich.progress import Progress

    from spsim.dataset import Dataset
    from spsim.emulate.dvs import EventEmulator

    from .common import _validate_directories

    input_dir, output_dir = _validate_directories(input_dir, output_dir)
    (output_dir / "frames").mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.txt"
    dataset = Dataset.from_path(input_dir)

    emulator_kwargs = dict(
        pos_thres=pos_thres,
        neg_thres=neg_thres,
        sigma_thres=sigma_thres,
        cutoff_hz=cutoff_hz,
        leak_rate_hz=leak_rate_hz,
        shot_noise_rate_hz=shot_noise_rate_hz,
        seed=seed,
    )
    emulator = EventEmulator(**emulator_kwargs)

    if events_path.exists():
        if force:
            events_path.unlink()
        else:
            raise FileExistsError(f"Event file already exists in {output_dir}")

    with open(output_dir / "params.json", "w") as f:
        json.dump(emulator_kwargs | dict(fps=fps), f, indent=2)

    with open(events_path, "a+") as out, Progress() as progress:
        task = progress.add_task("Processing @ N/A KEV/s", total=len(dataset))

        for idx, frame, _ in dataset:
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
                viz[ny, nx] = [255, 0, 0]
                viz[py, px] = [0, 0, 255]
                iio.imwrite(output_dir / "frames" / f"event_{idx:06}.png", viz)
            else:
                rate = 0

            progress.update(task, description=f"Processing @ {rate:.1f} KEV/s", advance=1)


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save binary frames",
        "chunk_size": "number of consecutive frames to average together, default: 10",
        "factor": "multiply image's linear intensity by this weight, default: 1.0",
        "readout_std": "standard deviation of gaussian read noise, default: 20",
        "fwc": "full well capacity of sensor in arbitrary units (relative to factor & chunk_size), default: chunk_size",
        "duplicate": (
            "when chunk size is too small, this model is ill-suited and creates unrealistic noise. "
            "This parameter artificially increases the chunk size by using each input image `duplicate` "
            "number of times. default: 1"
        ),
        "pattern": "filenames of frames should match this, default: 'frame_{:06}.png'",
        "mode": "how to save binary frames, either as 'img' or as 'npy', default: 'npy'",
        "force": "if true, overwrite output file(s) if present, default: False",
    }
)
def rgb(
    c,
    input_dir,
    output_dir,
    chunk_size=10,
    factor=1.0,
    readout_std=20.0,
    fwc=None,
    duplicate=1,
    pattern="frame_{:06}.png",
    mode="img",
    force=False,
):
    """Simulate real camera, adding read/poisson noise and tonemapping"""
    import copy

    import more_itertools as mitertools
    from rich.progress import Progress
    from torch.utils.data import DataLoader

    from spsim.dataset import Dataset, ImgDatasetWriter, NpyDatasetWriter, default_collate
    from spsim.interpolate import pose_interp
    from spsim.utils.color import srgb_to_linearrgb

    from .common import _validate_directories

    input_dir, output_dir = _validate_directories(input_dir, output_dir)
    dataset = Dataset.from_path(input_dir)
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

    loader = DataLoader(dataset, batch_size=1, num_workers=c.get("max_threads"), collate_fn=default_collate)

    with (
        ImgDatasetWriter(output_dir, transforms=transforms_new, force=force, pattern=pattern)
        if mode.lower() == "img"
        else NpyDatasetWriter(output_dir, np.ceil(shape).astype(int), transforms=transforms_new, force=force) as writer,
        Progress() as progress,
    ):
        task = progress.add_task("Writing RGB frames", total=len(dataset))
        for i, batch in enumerate(mitertools.ichunked(loader, chunk_size)):
            # Batch is an iterable of (idx, img, pose) that we need to reduce
            idxs, imgs, poses = mitertools.unzip(batch)
            imgs = [(i.astype(float) / 255.0).astype(float) for i in imgs]
            idxs, poses = np.concatenate(list(idxs)), np.concatenate(list(poses))

            # Assume images have been tonemapped and undo mapping
            imgs = [srgb_to_linearrgb(i) for i in imgs]

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


@task(
    help={
        "input_dir": "directory in which to look for transforms.json",
        "output_file": "file in which to save simulated IMU data. Prints to stdout if empty. default: ''",
        "seed": "RNG seed value for reproducibility. default: 2147483647",
        "gravity": "gravity vector in world coordinate frame. Given in m/s^2. default: [0,0,-9.8]",
        "dt": "time between consecutive transforms.json poses (assumed regularly spaced). Given in seconds. default: 0.00125",
        "init_bias_acc": "initial bias/drift in accelerometer reading. Given in m/s^2. default: [0,0,0]",
        "init_bias_gyro": "initial bias/drift in gyroscope reading. Given in rad/s. default: [0,0,0]",
        "std_bias_acc": (
            "stdev for random-walk component of error (drift) in accelerometer. "
            "Given in m/(s^3 sqrt(Hz)). default: 5.5e-5"
        ),
        "std_bias_gyro": (
            "stdev for random-walk component of error (drift) in gyroscope. Given in rad/(s^2 sqrt(Hz)). default: 2e-5"
        ),
        "std_acc": (
            "stdev for white-noise component of error in accelerometer. Given in m/(s^2 sqrt(Hz)). default: 8e-3"
        ),
        "std_gyro": (
            "stdev for white-noise component of error in gyroscope. Given in rad/(s sqrt(Hz)). default: 1.2e-3"
        ),
    }
)
def imu(
    _,
    input_dir,
    output_file="",
    seed=2147483647,
    gravity="(0.0, 0.0, -9.8)",
    dt=0.00125,
    init_bias_acc="(0.0,0.0,0.0)",
    init_bias_gyro="(0.0,0.0,0.0)",
    std_bias_acc=5.5e-5,
    std_bias_gyro=2e-5,
    std_acc=8e-3,
    std_gyro=1.2e-3,
):
    """Simulate data from a co-located IMU using the poses in transforms.json."""

    import ast
    import sys
    from pathlib import Path

    from spsim.dataset import Dataset

    if not Path(input_dir).resolve().exists():
        raise RuntimeError("Input directory path doesn't exist!")
    dataset = Dataset.from_path(input_dir)
    if dataset.transforms is None:
        raise RuntimeError("dataset.transforms not found!")

    rng = np.random.default_rng(int(seed))
    gravity = np.array(ast.literal_eval(gravity))
    init_bias_acc = np.array(ast.literal_eval(init_bias_acc))
    init_bias_gyro = np.array(ast.literal_eval(init_bias_gyro))

    from spsim.emulate.imu import emulate_imu

    data_gen = emulate_imu(
        dataset.poses,
        dt=dt,
        std_acc=std_acc,
        std_gyro=std_gyro,
        std_bias_acc=std_bias_acc,
        std_bias_gyro=std_bias_gyro,
        init_bias_acc=init_bias_acc,
        init_bias_gyro=init_bias_gyro,
        gravity=gravity,
        rng=rng,
    )

    with open(output_file, "w") if output_file else sys.stdout as out:
        out.write("t,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,bias_ax,bias_ay,bias_az,bias_gx,bias_gy,bias_gz\n")
        for d in data_gen:
            out.write(
                "{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    d["t"],
                    d["acc_reading"][0],
                    d["acc_reading"][1],
                    d["acc_reading"][2],
                    d["gyro_reading"][0],
                    d["gyro_reading"][1],
                    d["gyro_reading"][2],
                    d["acc_bias"][0],
                    d["acc_bias"][1],
                    d["acc_bias"][2],
                    d["gyro_bias"][0],
                    d["gyro_bias"][1],
                    d["gyro_bias"][2],
                )
            )
