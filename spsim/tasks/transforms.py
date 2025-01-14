import functools
from pathlib import Path

import numpy as np
import OpenEXR
from invoke import task
from rich.progress import Progress, track

from spsim.tasks.common import _validate_directories


def _read_exr(path):
    # imageio and cv2's cannot read an exr file when the data is stored in any other channel than RGB(A)
    # but as of blender 4.x depth maps are correctly saved as single channel exrs, in the V channel.
    with OpenEXR.File(path) as f:
        if len(f.channels()) and list(f.channels().keys())[0] == "RGBA":
            return f.channels()["RGBA"].pixels.transpose(2, 0, 1)
        return np.array([c.pixels for c in f.channels().values()])


def _tonemap_collate(batch, *, hdr_quantile=0.01):
    """Use default collate function on batch and then tonemap, enabling compute to be done in threads"""
    from spsim.color import linearrgb_to_srgb
    from spsim.dataset import default_collate

    idxs, imgs, poses = default_collate(batch)
    high, low = np.quantile(imgs, [1 - hdr_quantile, hdr_quantile])
    imgs = linearrgb_to_srgb(imgs)
    imgs = (np.clip(imgs, 0, 1) * 255).astype(np.uint8)

    return idxs, imgs, poses, high / low


def _estimate_distribution(in_files, percentage=0.2, transform=None):
    from tdigest import TDigest

    digest = TDigest()
    probe_files = np.random.choice(in_files, size=int(len(in_files) * percentage), replace=False)

    for in_file in track(probe_files, description="Probing Files..."):
        im = _read_exr(in_file)
        values = transform(im) if transform is not None else im.flatten()
        digest.batch_update(values)
    return digest


@task(
    auto_shortflags=False,
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save colorized frames",
        "pattern": "filenames of frames should match this, default: 'depth_*.exr'",
        "cmap": "which matplotlib colormap to use, default: 'turbo'",
        "ext": "which format to save colorized frames as, default: '.png'",
        "vmin": "minimum expected depth used to normalize colormap, default: None",
        "vmax": "maximum expected depth used to normalize colormap, default: None",
        "percentage": "if vmin/vmax are None, sample a subset of frames to determine range. "
        "This sets the sampling amount, default: 0.2",
        "sample": "proportion of pixels to sample per depth map when auto-setting vmin/vmax, default: 0.01",
        "step": "drop some frames when colorizing, use frames 0+step*n, default: 1",
    },
)
def colorize_depths(
    _,
    input_dir,
    output_dir,
    pattern="depth_*.exr",
    cmap="turbo",
    ext=".png",
    vmin=None,
    vmax=None,
    percentage=0.2,
    sample=0.01,
    step=1,
):
    """Convert .exr depth maps into color-coded images for visualization"""
    # TODO: Multiprocess this
    # Lazy load imports to improve CLI responsiveness
    import imageio.v3 as iio
    import matplotlib as mpl
    import matplotlib.cm as cm

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern)
    in_files = in_files[::step]

    if vmin is None and vmax is None:
        digest = _estimate_distribution(in_files, percentage=percentage, transform=lambda a: np.random.choice(a.flatten(), size=int(a.size*sample)))
        vmin, vmax = digest.percentile(1), digest.percentile(99)
        print(f"Using depth range [{vmin:0.2f}, {vmax:0.2f}]\n")

    cmap = getattr(cm, cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    for in_file in track(in_files):
        # Open with imageio, convert to color using matplotlib's cmaps and save as png.
        depth = _read_exr(in_file)
        depth[depth == 10000000000] = np.nan
        img = (cmap(norm(depth)) * 255).astype(np.uint8)
        path = output_dir / Path(in_file).stem
        iio.imwrite(str(path.with_suffix(ext)), img)


@task(
    auto_shortflags=False,
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save colorized frames",
        "direction": "direction of flow to colorize, either 'forward' or 'backwards', default: 'forward'",
        "pattern": "filenames of frames should match this, default: 'flow_*.exr'",
        "ext": "which format to save colorized frames as, default: '.png'",
        "vmax": "maximum expected flow magnitude, default: None",
        "percentage": "if vmax is None, sample a subset of frames to determine range. "
        "This sets the sampling amount, default: 0.2",
        "sample": "proportion of pixels to sample per flow map when auto-setting vmin/vmax, default: 0.01",
        "step": "drop some frames when colorizing, use frames 0+step*n, default: 1",
    },
)
def colorize_flows(
    _,
    input_dir,
    output_dir,
    direction="forward",
    pattern="flow_*.exr",
    ext=".png",
    vmax=None,
    percentage=0.2,
    sample=0.01,
    step=1,
):
    """Convert .exr optical flow maps into color-coded images for visualization"""
    # TODO: Multiprocess this
    # Lazy load imports to improve CLI responsiveness
    import colorsys
    import imageio.v3 as iio

    if direction.lower() not in ("forward", "backward"):
        raise ValueError("Direction needs to be either 'forward' or 'backwards'.")

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern)
    in_files = in_files[::step]
    convert = np.vectorize(colorsys.hsv_to_rgb)

    def magnitude(flows):
        fx, fy, bx, by = flows
        x, y = (fx, fy) if direction.lower() == "forward" else (bx, by) 
        mag = np.sqrt(x**2 + y**2)
        return np.random.choice(mag.flatten(), size=int(mag.size*sample))

    if vmax is None:
        digest = _estimate_distribution(in_files, percentage=percentage, transform=magnitude)
        vmax = digest.percentile(99)
        print(f"Using a maximum magnitude of {vmax:0.2f}\n")

    for in_file in track(in_files):
        fx, fy, bx, by = _read_exr(in_file)
        x, y = (fx, fy) if direction.lower() == "forward" else (bx, by)
        h = np.arctan2(y, x) / (2*np.pi) + 0.5
        v = np.minimum(np.sqrt(x**2 + y**2) / vmax, 1.0)
        img = np.stack(convert(h, np.ones_like(h), v), axis=-1)
        img = (img*255).astype(np.uint8)
        path = output_dir / Path(in_file).stem
        iio.imwrite(str(path.with_suffix(ext)), img)


@task(
    auto_shortflags=False,
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save colorized frames",
        "pattern": "filenames of frames should match this, default: 'normal_*.exr'",
        "ext": "which format to save colorized frames as, default: '.png'",
        "step": "drop some frames when colorizing, use frames 0+step*n, default: 1",
    },
)
def colorize_normals(
    _,
    input_dir,
    output_dir,
    pattern="normal_*.exr",
    ext=".png",
    step=1,
):
    """Convert .exr normal maps into color-coded images for visualization"""
    # TODO: Multiprocess this
    # Lazy load imports to improve CLI responsiveness
    import imageio.v3 as iio

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern)
    in_files = in_files[::step]

    for in_file in track(in_files):
        img = np.stack(_read_exr(in_file)/2 + 0.5, axis=-1)
        img = (img*255).astype(np.uint8)
        path = output_dir / Path(in_file).stem
        iio.imwrite(str(path.with_suffix(ext)), img)


@task(
    auto_shortflags=False,
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save colorized frames",
        "pattern": "filenames of frames should match this, default: 'segmentation_*.exr'",
        "ext": "which format to save colorized frames as, default: '.png'",
        "num_objects": "number of unique objects to expect in the scene, default: None",
        "shuffle": "if true, colorize items in a random order, default: True",
        "seed": "seed used when shuffling colors, default: 1234",
        "step": "drop some frames when colorizing, use frames 0+step*n, default: 1",
    },
)
def colorize_segmentations(
    _,
    input_dir,
    output_dir,
    pattern="segmentation_*.exr",
    ext=".png",
    num_objects=None,
    shuffle=True,
    seed=1234,
    step=1,
):
    """Convert .exr segmentation maps into color-coded images for visualization"""
    # TODO: Multiprocess this
    # Lazy load imports to improve CLI responsiveness
    import colorsys

    import imageio.v3 as iio

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
        idx = _read_exr(in_file).astype(int)
        img = np.stack([r[idx], g[idx], b[idx]], axis=-1)
        img = (img*255).astype(np.uint8)
        path = output_dir / Path(in_file).stem
        iio.imwrite(str(path.with_suffix(ext)), img)


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save tone mapped frames, if not specified the dynamic range is calculated and no tonemapping occurs, default: None",
        "batch_size": "number of frames to write at once, default: 4",
        "hdr_quantile": "calculate dynamic range using brightness quantiles instead of extrema, default: 0.01",
        "force": "if true, overwrite output file(s) if present, default: False",
    }
)
def tonemap_exrs(c, input_dir, output_dir=None, batch_size=4, hdr_quantile=0.01, force=False):
    """Convert .exr linear intensity frames into tone-mapped sRGB images"""
    from torch.utils.data import DataLoader

    from spsim.dataset import ImgDatasetWriter, dataset_dispatch

    input_dir, output_dir = _validate_directories(input_dir, output_dir)
    dataset = dataset_dispatch(input_dir)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=c.get("max_threads"),
        collate_fn=functools.partial(_tonemap_collate, hdr_quantile=hdr_quantile),
    )
    hdrs = []

    with Progress() as progress:
        pbar = progress.add_task(total=len(dataset))

        with ImgDatasetWriter(output_dir, transforms=dataset.transforms, force=force, pattern="frame_{:06}.png") as writer:
            for idxs, imgs, poses, hdr in loader:
                writer[idxs] = (imgs, poses)
                hdrs.append(hdr)
                progress.update(pbar, advance=len(idxs))

    hdrs = np.array(hdrs)
    print(f"Mean dynamic range is {hdrs.mean():0.2f}, with range ({hdrs.min():0.2f}, {hdrs.max():0.2f})")
