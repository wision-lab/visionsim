from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from rich.progress import track


def _estimate_distribution(in_files, transform=None):
    from fastdigest import TDigest

    from visionsim.dataset import Dataset

    digest = TDigest()

    for in_file in track(in_files, description="Probing Files..."):
        im = Dataset.load_data(in_file)
        values = transform(im) if transform is not None else im.flatten()
        digest.batch_update(values)
    return digest


def colorize_depths(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "depth_*.exr",
    cmap: str = "turbo",
    ext: str = ".png",
    vmin: float | None = None,
    vmax: float | None = None,
    quantile: float = 0.01,
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
        quantile: if vmin/vmax are None, use this quantile to estimate them
        step: drop some frames when colorizing, use frames 0+step*n
    """
    # TODO: Multiprocess this
    # Lazy load imports to improve CLI responsiveness
    import imageio.v3 as iio
    import matplotlib as mpl
    import matplotlib.cm as cm

    from visionsim.cli import _log, _validate_directories
    from visionsim.dataset import Dataset

    DEPTH_CUTOFF = 10000000000

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern)
    in_files = in_files[::step]

    def transform_depth(d):
        # Filter out large depths, this is a render bug in CYCLES
        # See: https://blender.stackexchange.com/questions/325007
        d = d[d < DEPTH_CUTOFF]
        return d.flatten()

    if vmin is None or vmax is None:
        digest = _estimate_distribution(in_files, transform=transform_depth)
        vmin_, vmax_ = digest.quantile(quantile), digest.quantile(1 - quantile)
        vmin = vmin_ if vmin is None else vmin
        vmax = vmax_ if vmax is None else vmax
        _log.info(f"Found depth range [{vmin_:0.2f}, {vmax_:0.2f}]")
        _log.info(f"Using depth range [{vmin:0.2f}, {vmax:0.2f}]\n")

    colormap = getattr(cm, cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    for in_file in track(in_files):
        # Open with imageio, convert to color using matplotlib's cmaps and save as png.
        depth = Dataset.load_data(in_file)
        depth[depth >= DEPTH_CUTOFF] = np.nan
        img = (colormap(norm(depth)) * 255).astype(np.uint8)
        path = output_dir / Path(in_file).stem
        iio.imwrite(str(path.with_suffix(ext)), img)


def colorize_flows(
    input_dir: Path,
    output_dir: Path,
    direction: Literal["forward", "backward"] = "forward",
    pattern: str = "**/*.exr",
    ext: str = ".png",
    vmax: float | None = None,
    quantile: float = 0.01,
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
        quantile: if vmax is None, use this quantile to estimate it
        step: drop some frames when colorizing, use frames 0+step*n
    """

    # TODO: Multiprocess this
    # Lazy load imports to improve CLI responsiveness
    import colorsys

    import imageio.v3 as iio

    from visionsim.cli import _log, _validate_directories
    from visionsim.dataset import Dataset

    if direction.lower() not in ("forward", "backward"):
        raise ValueError("Direction needs to be either 'forward' or 'backwards'.")

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern)
    in_files = in_files[::step]
    convert = np.vectorize(colorsys.hsv_to_rgb)

    def magnitude(flows):
        fx, fy, bx, by = flows
        x, y = (fx, fy) if direction.lower() == "forward" else (bx, by)
        mag = np.sqrt(x**2 + y**2)
        return mag.flatten()

    if vmax is None:
        digest = _estimate_distribution(in_files, transform=magnitude)
        vmax = digest.quantile(1 - quantile)
        _log.info(f"Using a maximum magnitude of {vmax:0.2f}\n")

    for in_file in track(in_files):
        fx, fy, bx, by = Dataset.load_data(in_file)
        x, y = (fx, fy) if direction.lower() == "forward" else (bx, by)
        h = np.arctan2(y, x) / (2 * np.pi) + 0.5
        v = np.minimum(np.sqrt(x**2 + y**2) / vmax, 1.0)
        img = np.stack(convert(h, np.ones_like(h), v), axis=-1)
        img = (img * 255).astype(np.uint8)
        path = output_dir / Path(in_file).stem
        iio.imwrite(str(path.with_suffix(ext)), img)


def colorize_normals(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "**/*.exr",
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
    from visionsim.dataset import Dataset

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern)
    in_files = in_files[::step]

    for in_file in track(in_files):
        img = Dataset.load_data(in_file).transpose(1, 2, 0) / 2 + 0.5
        img = (img * 255).astype(np.uint8)
        path = output_dir / Path(in_file).stem
        iio.imwrite(str(path.with_suffix(ext)), img)


def colorize_segmentations(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "**/*.exr",
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

    from visionsim.cli import _log, _validate_directories
    from visionsim.dataset import Dataset

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern)
    in_files = in_files[::step]

    if num_objects is None:
        digest = _estimate_distribution(in_files, transform=np.unique)
        num_objects = int(digest.max())
        _log.info(f"Found {num_objects} objects.\n")

    indices = np.arange(num_objects)

    if shuffle:
        np.random.seed(seed=seed)
        np.random.shuffle(indices)

    convert = np.vectorize(colorsys.hsv_to_rgb)
    r, g, b = convert(np.arange(num_objects) / num_objects, np.ones(num_objects), np.ones(num_objects))
    r, g, b = np.insert(r, 0, 0), np.insert(g, 0, 0), np.insert(b, 0, 0)

    for in_file in track(in_files):
        idx = Dataset.load_data(in_file).astype(int).squeeze()

        if idx.shape[-1] != 1 and idx.ndim == 3:
            idx = idx[..., 0]

        img = np.stack([r[idx], g[idx], b[idx]], axis=-1)
        img = (img * 255).astype(np.uint8)
        path = output_dir / Path(in_file).stem
        iio.imwrite(str(path.with_suffix(ext)), img)


def tonemap_frames(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "**/*.exr",
    ext: str = ".png",
    hdr_quantile: float = 0.01,
):
    """Convert .exr linear intensity frames (or composites) into tone-mapped sRGB images

    Args:
        input_dir: directory in which to look for frames
        output_dir: directory in which to save tone mapped frames
        pattern: filenames of frames should match this
        ext: which format to save colorized frames as
        hdr_quantile: calculate dynamic range using brightness quantiles instead of extrema
    """
    import imageio.v3 as iio

    from visionsim.cli import _log, _validate_directories
    from visionsim.dataset import Dataset
    from visionsim.utils.color import linearrgb_to_srgb

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern)
    hdrs = []

    for in_file in track(in_files):
        img = Dataset.load_data(in_file)
        high, low = np.quantile(img, [1 - hdr_quantile, hdr_quantile])
        img = linearrgb_to_srgb(img)
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        hdrs.append(high / low)

        path = output_dir / Path(in_file).stem
        iio.imwrite(str(path.with_suffix(ext)), img)

    hdrs_ = np.array(hdrs)
    _log.info(f"Mean dynamic range is {hdrs_.mean():0.2f}, with range ({hdrs_.min():0.2f}, {hdrs_.max():0.2f})")
