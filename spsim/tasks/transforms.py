import functools
from pathlib import Path

import numpy as np
import OpenEXR
from invoke import task
from tqdm.auto import tqdm

from spsim.tasks.common import _validate_directories


def _read_exr(path):
    # imageio and cv2's cannot read an exr file when the data is stored in any other channel than RGB(A)
    # but as of blender 4.x depth maps are correctly saved as single channel exrs, in the V channel.
    with OpenEXR.File(path) as f:
        if len(f.channels()) != 1:
            raise RuntimeError("Encountered EXR file with multiple channels!")
        c, *_ = f.channels().keys()
        return f.channels()[c].pixels.squeeze()


def _tonemap_collate(batch, *, hdr_quantile=0.01):
    """Use default collate function on batch and then tonemap, enabling compute to be done in threads"""
    from spsim.color import linearrgb_to_srgb
    from spsim.dataset import default_collate

    idxs, imgs, poses = default_collate(batch)
    high, low = np.quantile(imgs, [1 - hdr_quantile, hdr_quantile])
    imgs = linearrgb_to_srgb(imgs)
    imgs = (np.clip(imgs, 0, 1) * 255).astype(np.uint8)

    return idxs, imgs, poses, high / low


@task(
    auto_shortflags=False,
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save colorized frames",
        "pattern": "filenames of frames should match this, default: 'depth_*.exr'",
        "cmap": "which matplotlib colormap to use, default: 'turbo'",
        "ext": "which format to save colorized frames as, default: '.jpg'",
        "vmin": "minimum expected depth used to normalize colormap, default: None",
        "vmax": "maximum expected depth used to normalize colormap, default: None",
        "percentage": "if vmin/vmax are None, sample a subset of frames to determine range. "
        "This sets the sampling amount, default: 0.2",
        "step": "drop some frames when colorizing, use frames 0+step*n, default: 1",
    },
)
def colorize_depth(
    _,
    input_dir,
    output_dir,
    pattern="depth_*.exr",
    cmap="turbo",
    ext=".png",
    vmin=None,
    vmax=None,
    percentage=0.2,
    step=1,
):
    """Convert .exr depth maps into color-coded images for visualization"""
    # TODO: Multiprocess this
    # Lazy load imports to improve CLI responsiveness
    import matplotlib as mpl
    import matplotlib.cm as cm

    from spsim.io import write_img

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern)
    in_files = in_files[::step]

    if not vmin and not vmax:
        print("Estimating depth range...")
        vmin, vmax = np.inf, -np.inf
        probe_files = np.random.choice(in_files, size=int(len(in_files) * percentage), replace=False)

        for in_file in tqdm(probe_files):
            # Open with opencv, convert to color using matplotlib's cmaps and save as png.
            depth = _read_exr(in_file)
            depth[depth == 10000000000] = np.nan
            vmin = np.minimum(vmin, np.nanmin(depth))
            vmax = np.maximum(vmax, np.nanmax(depth))
        print(f"\nUsing range [{vmin}, {vmax}]\n")

    cmap = getattr(cm, cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    for in_file in tqdm(in_files):
        # Open with imageio, convert to color using matplotlib's cmaps and save as png.
        depth = _read_exr(in_file)
        depth[depth == 10000000000] = np.nan
        img = (cmap(norm(depth)) * 255).astype(np.uint8)
        path = output_dir / Path(in_file).stem
        write_img(str(path.with_suffix(ext)), img)


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
    from tqdm.auto import tqdm

    from spsim.dataset import ImgDatasetWriter, dataset_dispatch

    from .common import _validate_directories

    input_dir, output_dir = _validate_directories(input_dir, output_dir)
    dataset = dataset_dispatch(input_dir)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=c.get("max_threads"),
        collate_fn=functools.partial(_tonemap_collate, hdr_quantile=hdr_quantile),
    )
    pbar = tqdm(total=len(dataset))
    hdrs = []

    with ImgDatasetWriter(output_dir, transforms=dataset.transforms, force=force, pattern="frame_{:06}.png") as writer:
        for idxs, imgs, poses, hdr in loader:
            writer[idxs] = (imgs, poses)
            hdrs.append(hdr)
            pbar.update(len(idxs))

    hdrs = np.array(hdrs)
    print(f"Mean dynamic range is {hdrs.mean():0.2f}, with range ({hdrs.min():0.2f}, {hdrs.max():0.2f})")
