from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Literal

import numpy as np
import OpenEXR  # type: ignore
from rich.progress import Progress, track


def _read_exr(path):
    # imageio and cv2's cannot read an exr file when the data is stored in any other channel than RGB(A)
    # but as of blender 4.x depth maps are correctly saved as single channel exrs, in the V channel.
    with OpenEXR.File(path) as f:
        if len(f.channels()) and list(f.channels().keys())[0] == "RGBA":
            return f.channels()["RGBA"].pixels.transpose(2, 0, 1)
        return np.array([c.pixels for c in f.channels().values()])


def _tonemap_collate(batch, *, hdr_quantile=0.01):
    """Use default collate function on batch and then tonemap, enabling compute to be done in threads"""
    from visionsim.dataset import default_collate
    from visionsim.utils.color import linearrgb_to_srgb

    idxs, imgs, poses = default_collate(batch)
    high, low = np.quantile(imgs, [1 - hdr_quantile, hdr_quantile])
    imgs = linearrgb_to_srgb(imgs)
    imgs = (np.clip(imgs, 0, 1) * 255).astype(np.uint8)

    return idxs, imgs, poses, high / low


def _estimate_distribution(in_files, percentage=0.2, transform=None):
    from tdigest import TDigest  # type: ignore

    digest = TDigest()
    probe_files = np.random.choice(in_files, size=int(len(in_files) * percentage), replace=False)

    for in_file in track(probe_files, description="Probing Files..."):
        im = _read_exr(in_file)
        values = transform(im) if transform is not None else im.flatten()
        digest.batch_update(values)
    return digest


def colorize_depths(
    input_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    pattern: str = "depth_*.exr",
    cmap: str = "turbo",
    ext: str = ".png",
    vmin: float | None = None,
    vmax: float | None = None,
    percentage: float = 0.2,
    sample: float = 0.01,
    step: int = 1,
):
    """Convert .exr depth maps into color-coded images for visualization

    Args:
        input_dir: directory in which to look for frames
        output_dir: directory in which to save colorized frames
        pattern: filenames of frames should match this
        cmap: which matplotlib colormap to use
        ext: which format to save colorized frames as
        vmin: minimum expected depth used to normalize colormap
        vmax: maximum expected depth used to normalize colormap
        percentage: if vmin/vmax are None, sample a subset of frames to determine range. This sets the sampling amount
        sample: proportion of pixels to sample per depth map when auto-setting vmin/vmax
        step: drop some frames when colorizing, use frames 0+step*n
    """
    # TODO: Multiprocess this
    # Lazy load imports to improve CLI responsiveness
    import imageio.v3 as iio
    import matplotlib as mpl
    import matplotlib.cm as cm

    from visionsim.cli import _validate_directories

    DEPTH_CUTOFF = 10000000000

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern)
    in_files = in_files[::step]

    def transform_depth(d):
        # Filter out large depths, this is a render bug in CYCLES
        # See: https://blender.stackexchange.com/questions/325007
        d = d[d < DEPTH_CUTOFF]
        return np.random.choice(d.flatten(), size=int(d.size * sample))

    if vmin is None and vmax is None:
        digest = _estimate_distribution(in_files, percentage=percentage, transform=transform_depth)
        vmin, vmax = digest.percentile(1), digest.percentile(99)
        print(f"Using depth range [{vmin:0.2f}, {vmax:0.2f}]\n")

    colormap = getattr(cm, cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    for in_file in track(in_files):
        # Open with imageio, convert to color using matplotlib's cmaps and save as png.
        depth = _read_exr(in_file)
        depth[depth >= DEPTH_CUTOFF] = np.nan
        img = (colormap(norm(depth)) * 255).astype(np.uint8)
        path = output_dir / Path(in_file).stem
        iio.imwrite(str(path.with_suffix(ext)), img)


def colorize_flows(
    input_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    direction: Literal["forward", "backward"] = "forward",
    pattern: str = "flow_*.exr",
    ext: str = ".png",
    vmax: float | None = None,
    percentage: float = 0.2,
    sample: float = 0.01,
    step: int = 1,
):
    """Convert .exr optical flow maps into color-coded images for visualization

    Args:
        input_dir: directory in which to look for frames
        output_dir: directory in which to save colorized frames
        direction: direction of flow to colorize
        pattern: filenames of frames should match this
        ext: which format to save colorized frames as
        vmax: maximum expected flow magnitude
        percentage: if vmax is None, sample a subset of frames to determine range. This sets the sampling amount
        sample: proportion of pixels to sample per flow map when auto-setting vmin/vmax
        step: drop some frames when colorizing, use frames 0+step*n
    """

    # TODO: Multiprocess this
    # Lazy load imports to improve CLI responsiveness
    import colorsys

    import imageio.v3 as iio

    from visionsim.cli import _validate_directories

    if direction.lower() not in ("forward", "backward"):
        raise ValueError("Direction needs to be either 'forward' or 'backwards'.")

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern)
    in_files = in_files[::step]
    convert = np.vectorize(colorsys.hsv_to_rgb)

    def magnitude(flows):
        fx, fy, bx, by = flows
        x, y = (fx, fy) if direction.lower() == "forward" else (bx, by)
        mag = np.sqrt(x**2 + y**2)
        return np.random.choice(mag.flatten(), size=int(mag.size * sample))

    if vmax is None:
        digest = _estimate_distribution(in_files, percentage=percentage, transform=magnitude)
        vmax = digest.percentile(99)
        print(f"Using a maximum magnitude of {vmax:0.2f}\n")

    for in_file in track(in_files):
        fx, fy, bx, by = _read_exr(in_file)
        x, y = (fx, fy) if direction.lower() == "forward" else (bx, by)
        h = np.arctan2(y, x) / (2 * np.pi) + 0.5
        v = np.minimum(np.sqrt(x**2 + y**2) / vmax, 1.0)
        img = np.stack(convert(h, np.ones_like(h), v), axis=-1)
        img = (img * 255).astype(np.uint8)
        path = output_dir / Path(in_file).stem
        iio.imwrite(str(path.with_suffix(ext)), img)


def colorize_normals(
    input_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    pattern: str = "normal_*.exr",
    ext: str = ".png",
    step: int = 1,
):
    """Convert .exr normal maps into color-coded images for visualization

    Args:
        input_dir: directory in which to look for frames
        output_dir: directory in which to save colorized frames
        pattern: filenames of frames should match this
        ext: which format to save colorized frames as
        step: drop some frames when colorizing, use frames 0+step*n
    """
    # TODO: Multiprocess this
    # Lazy load imports to improve CLI responsiveness
    import imageio.v3 as iio

    from visionsim.cli import _validate_directories

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern)
    in_files = in_files[::step]

    for in_file in track(in_files):
        img = np.stack(_read_exr(in_file) / 2 + 0.5, axis=-1)
        img = (img * 255).astype(np.uint8)
        path = output_dir / Path(in_file).stem
        iio.imwrite(str(path.with_suffix(ext)), img)


def colorize_segmentations(
    input_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    pattern: str = "segmentation_*.exr",
    ext: str = ".png",
    num_objects: int | None = None,
    shuffle: bool = True,
    seed: int = 1234,
    step: int = 1,
):
    """Convert .exr segmentation maps into color-coded images for visualization

    Args:
        input_dir: directory in which to look for frames
        output_dir: directory in which to save colorized frames
        pattern: filenames of frames should match this
        ext: which format to save colorized frames as
        num_objects: number of unique objects to expect in the scene
        shuffle: if true, colorize items in a random order
        seed: seed used when shuffling colors
        step: drop some frames when colorizing, use frames 0+step*n
    """
    # TODO: Multiprocess this
    # Lazy load imports to improve CLI responsiveness
    import colorsys

    import imageio.v3 as iio

    from visionsim.cli import _validate_directories

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern)
    in_files = in_files[::step]

    if num_objects is None:
        digest = _estimate_distribution(in_files, percentage=1, transform=np.unique)
        num_objects = int(digest.percentile(100))
        print(f"Found {num_objects} objects.\n")

    indices = np.arange(num_objects)

    if shuffle:
        np.random.seed(seed=seed)
        np.random.shuffle(indices)

    convert = np.vectorize(colorsys.hsv_to_rgb)
    r, g, b = convert(np.arange(num_objects) / num_objects, np.ones(num_objects), np.ones(num_objects))
    r, g, b = np.insert(r, 0, 0), np.insert(g, 0, 0), np.insert(b, 0, 0)

    for in_file in track(in_files):
        idx = _read_exr(in_file).astype(int).squeeze()

        if idx.shape[-1] != 1 and idx.ndim == 3:
            idx = idx[..., 0]

        img = np.stack([r[idx], g[idx], b[idx]], axis=-1)
        img = (img * 255).astype(np.uint8)
        path = output_dir / Path(in_file).stem
        iio.imwrite(str(path.with_suffix(ext)), img)


def tonemap_exrs(
    input_dir: str | os.PathLike,
    output_dir: str | os.PathLike | None = None,
    batch_size: int = 4,
    hdr_quantile: float = 0.01,
    force: bool = False,
):
    """Convert .exr linear intensity frames into tone-mapped sRGB images

    Args:
        input_dir: directory in which to look for frames
        output_dir: directory in which to save tone mapped frames, if not specified the dynamic range is calculated and no tonemapping occurs
        batch_size: number of frames to write at once
        hdr_quantile: calculate dynamic range using brightness quantiles instead of extrema
        force: if true, overwrite output file(s) if present
    """
    from torch.utils.data import DataLoader

    from visionsim.cli import _validate_directories
    from visionsim.dataset import Dataset, ImgDatasetWriter

    input_path, output_path, *_ = _validate_directories(input_dir, output_dir)
    dataset = Dataset.from_path(input_path)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=os.cpu_count() or 1,
        collate_fn=functools.partial(_tonemap_collate, hdr_quantile=hdr_quantile),
    )
    hdrs = []

    with Progress() as progress:
        pbar = progress.add_task(description="Processing Frames...", total=len(dataset))

        with ImgDatasetWriter(
            output_path, transforms=dataset.transforms, force=force, pattern="frame_{:06}.png"
        ) as writer:
            for idxs, imgs, poses, hdr in loader:
                writer[idxs] = (imgs, poses)
                hdrs.append(hdr)
                progress.update(pbar, advance=len(idxs))

    hdrs_ = np.array(hdrs)
    print(f"Mean dynamic range is {hdrs_.mean():0.2f}, with range ({hdrs_.min():0.2f}, {hdrs_.max():0.2f})")
