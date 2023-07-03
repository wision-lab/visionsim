import functools
from pathlib import Path

import numpy as np
from invoke import task
from tqdm.auto import tqdm

from tasks.common import _validate_directories


def _emulate_blur_single(item, grayscale=False, chunk_size=10, output_dir=None, ext=None):
    from spsim.io import read_img, write_img

    i, chunk_files = item

    stream = (read_img(p, apply_alpha=True, grayscale=grayscale) for p in chunk_files)
    avg_img, avg_alpha = next(stream)

    for img, alpha in stream:
        avg_img += img
        avg_alpha += alpha

    avg_img = (avg_img / chunk_size * 255).astype(np.uint8)
    avg_alpha = (avg_alpha / chunk_size * 255).astype(np.uint8)
    path = output_dir / f"frame_{i:06}{ext}"
    return write_img(str(path), np.concatenate([avg_img, avg_alpha], axis=-1))


def _emulate_spad_single(item, grayscale=False, factor=1.0, output_dir=None, ext=None):
    from spsim.color import srgb_to_linearrgb  # Lazy load
    from spsim.io import read_img, write_img  # Lazy load
    from spsim.utils import img_to_tensor, tensor_to_img  # Lazy Load

    in_file, seed = item
    rng = np.random.default_rng(int(seed))
    img, alpha = read_img(in_file, apply_alpha=True, grayscale=grayscale)

    # If the image filetype is anything but exr, assume it's been tonemapped and undo mapping
    if not in_file.endswith(".exr"):
        img = tensor_to_img(srgb_to_linearrgb(img_to_tensor(img)))

    # If image is a PNG, the alpha channel might cause issues. We separate it and use it to modulate
    # the bernoulli sampling such that areas with zero alpha have no photon detections
    binary_img = rng.binomial(1, alpha - np.exp(-img[:, :, :3] * factor) * alpha)
    binary_img = np.concatenate([binary_img, alpha], axis=-1) * 255
    binary_img = binary_img.astype(np.uint8)

    path = output_dir / Path(in_file).stem
    write_img(str(path.with_suffix(ext)), binary_img)


def _emulate_rgb_single(
    item, grayscale=False, factor=1.0, chunk_size=10, readout_std=20, fwc=500, output_dir=None, ext=None
):
    from spsim.color import emulate_rgb_from_merged, srgb_to_linearrgb  # Lazy load
    from spsim.io import read_img, write_img  # Lazy Load
    from spsim.utils import img_to_tensor, tensor_to_img  # Lazy Load

    i, chunk_files = item
    stream = (read_img(p, apply_alpha=True, grayscale=grayscale) for p in chunk_files)
    avg_img, avg_alpha = None, None

    for p, (img, alpha) in zip(chunk_files, stream):
        # If the image filetype is anything but exr, assume it's been tonemapped and undo mapping
        if not p.endswith(".exr"):
            img = tensor_to_img(srgb_to_linearrgb(img_to_tensor(img)))

        if avg_img is not None:
            avg_img += img * factor
        else:
            avg_img = img * factor

        if avg_alpha is not None:
            avg_alpha += alpha
        else:
            avg_alpha = alpha

    rgb_img = emulate_rgb_from_merged(
        img_to_tensor(avg_img / chunk_size), burst_size=chunk_size, readout_std=readout_std, fwc=fwc, factor=factor
    )

    path = output_dir / f"frame_{i:06}{ext}"
    return write_img(str(path), np.concatenate([tensor_to_img(rgb_img * 255), avg_alpha], axis=-1).astype(np.uint8))


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save binary frames",
        "grayscale": "if true, first convert to grayscale then sample. default: False",
        "pattern": "filenames of frames should match this, default: 'frame_*.png'",
        "ext": "which format to save colorized frames as, default: '.png'",
        "factor": "multiplicative factor controlling dynamic range of output, default: 1.0",
        "seed": "random seed to use while sampling, ensures reproducibility. default: 2147483647",
    }
)
def spad(
    _,
    input_dir,
    output_dir,
    grayscale=False,
    pattern="frame_*.png",
    ext=".png",
    factor=1.0,
    seed=2147483647,
):
    """Perform bernoulli sampling on linearized RGB frames to yield binary frames"""
    import multiprocessing

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern)

    np.random.seed(seed)
    seeds = np.random.randint(low=0, high=10e6, size=len(in_files))

    emulate_single = functools.partial(
        _emulate_spad_single, grayscale=grayscale, factor=factor, output_dir=output_dir, ext=ext
    )

    with multiprocessing.Pool() as p:
        tasks = p.imap(emulate_single, zip(in_files, seeds))
        list(tqdm(tasks, total=len(in_files)))


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save binary frames",
        "chunk_size": "number of consecutive frames to average together, default: 10",
        "grayscale": "if true, first convert to grayscale then sample. default: False",
        "pattern": "filenames of frames should match this, default: 'frame_*.png'",
        "ext": "which format to save colorized frames as, default: '.png'",
    }
)
def blur(_, input_dir, output_dir, chunk_size=10, grayscale=False, pattern="frame_*.png", ext=".png"):
    """Average together frames to create motion blur, similar to `emulate_rgb` but with no camera modeling"""
    import multiprocessing

    import more_itertools as mitertools

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern)
    emulate_single = functools.partial(
        _emulate_blur_single, grayscale=grayscale, chunk_size=chunk_size, output_dir=output_dir, ext=ext
    )

    # TODO: This would probably be faster if we used a DataLoader to load and average image batches...
    with multiprocessing.Pool() as p:
        tasks = p.imap(emulate_single, enumerate(mitertools.chunked(in_files, chunk_size)))
        list(tqdm(tasks, total=len(in_files) // chunk_size))


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save binary frames",
        "chunk_size": "number of consecutive frames to average together, default: 10",
        "factor": "multiple image's linear intensity by this weight, default: 1.0",
        "readout_std": "standard deviation of gaussian read noise, default: 20",
        "fwc": "full well capacity of sensor in arbitrary units (relative to factor & chunk_size), default: 500",
        "grayscale": "if true, first convert to grayscale then sample. default: False",
        "pattern": "filenames of frames should match this, default: 'frame_*.png'",
        "ext": "which format to save colorized frames as, default: '.png'",
    }
)
def rgb(
    _,
    input_dir,
    output_dir,
    chunk_size=10,
    factor=1.0,
    readout_std=20,
    fwc=500,
    grayscale=False,
    pattern="frame_*.png",
    ext=".png",
):
    """Simulate real camera, adding read/poisson noise and tonemapping"""
    import multiprocessing

    import more_itertools as mitertools

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern)
    emulate_single = functools.partial(
        _emulate_rgb_single,
        grayscale=grayscale,
        chunk_size=chunk_size,
        factor=factor,
        readout_std=readout_std,
        fwc=fwc,
        output_dir=output_dir,
        ext=ext,
    )

    # TODO: This would probably be faster if we used a DataLoader to load and average image batches...
    #   It would at least appear faster to the user as each image will be created with
    #   multiple threads (making pbar advance bit by bit) instead of having one thread per image.
    with multiprocessing.Pool() as p:
        tasks = p.imap(emulate_single, enumerate(mitertools.chunked(in_files, chunk_size)))
        list(tqdm(tasks, total=len(in_files) // chunk_size))
