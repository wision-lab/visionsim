import functools
from pathlib import Path

import numpy as np
from invoke import task
from tqdm.auto import tqdm

from tasks.common import _validate_directories


def _tonemap_single(in_file, output_dir=None, ext=None):
    from spsim.color import linearrgb_to_srgb  # Lazy load
    from spsim.io import read_img, write_img  # Lazy load
    from spsim.utils import img_to_tensor, tensor_to_img  # Lazy Load

    # Open with opencv, apply tone mapping and save as png.
    img, _ = read_img(in_file, apply_alpha=True)
    img = tensor_to_img(linearrgb_to_srgb(img_to_tensor(img)))
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    path = output_dir / Path(in_file).stem
    write_img(str(path.with_suffix(ext)), img)


@task(
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
    }
)
def colorize_depth(
    _,
    input_dir,
    output_dir,
    pattern="depth_*.exr",
    cmap="turbo",
    ext=".jpg",
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

    from spsim.io import read_img, write_img

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern)
    in_files = in_files[::step]

    if not vmin and not vmax:
        print("Estimating depth range...")
        vmin, vmax = np.inf, -np.inf
        probe_files = np.random.choice(in_files, size=int(len(in_files) * percentage), replace=False)

        for in_file in tqdm(probe_files):
            # Open with opencv, convert to color using matplotlib's cmaps and save as png.
            depth, _ = read_img(in_file, apply_alpha=False)
            depth[depth == 10000000000] = np.nan
            vmin = np.minimum(vmin, np.nanmin(depth[:, :, 0]))
            vmax = np.maximum(vmax, np.nanmax(depth[:, :, 0]))
        print(f"\nUsing range [{vmin}, {vmax}]\n")

    cmap = getattr(cm, cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    for in_file in tqdm(in_files):
        # Open with imageio, convert to color using matplotlib's cmaps and save as png.
        depth, _ = read_img(in_file, apply_alpha=False)
        depth[depth == 10000000000] = np.nan
        img = (cmap(norm(depth[:, :, 0])) * 255).astype(np.uint8)
        path = output_dir / Path(in_file).stem
        write_img(str(path.with_suffix(ext)), img)


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save tone mapped frames",
        "pattern": "filenames of frames should match this, default: 'frame_*.exr'",
        "ext": "which format to save colorized frames as, default: '.png'",
        "step": "drop some frames when tone mapping, use frames 0+step*n, default: 1",
    }
)
def tonemap_exrs(_, input_dir, output_dir, pattern="frame_*.exr", ext=".png", step=1):
    """Convert .exr linear intensity frames into tone-mapped sRGB images"""
    import multiprocessing

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern)

    tonemap_single = functools.partial(_tonemap_single, output_dir=output_dir, ext=ext)

    with multiprocessing.Pool() as p:
        tasks = p.imap(tonemap_single, in_files[::step])
        list(tqdm(tasks, total=len(in_files) // step))
