# TODO: This file is getting very long (although most of it is help
#  and docstrings), and really needs refactoring into a tasks folder.

import functools
import json
import platform
import re
from pathlib import Path

import numpy as np
from invoke import task
from tqdm.auto import tqdm

from spsim.cli import modify_signature
from spsim.render import parser_config


def _run(c, command, watchers=None, **kwargs):
    watchers = (
        [
            watchers,
        ]
        if watchers is not None and not isinstance(watchers, list)
        else watchers
    )
    return c.run(command, pty=platform.system() != "Windows", watchers=watchers, **kwargs)


def _log_run(c, command, log_path, echo=True, **kwargs):
    Path(log_path).mkdir(parents=True, exist_ok=True)
    log_out = Path(log_path).resolve() / "out.log"
    log_err = Path(log_path).resolve() / "err.log"

    with open(str(log_out), "w") as f_out:
        if echo:
            f_out.write(f"$ {command}\n")
        if platform.system() != "Windows":
            _run(c, command, out_stream=f_out, **kwargs)
        else:
            with open(str(log_err), "w") as f_err:
                _run(c, command, out_stream=f_out, err_stream=f_err, **kwargs)


def _raise_callback(*args, err_type=ValueError, message="", **kwargs):
    raise err_type(message)


def _validate_directories(input_dir, output_dir, pattern=None):
    import glob  # Lazy import

    from natsort import natsorted

    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise RuntimeError(f"Input directory {input_dir} does not exist.")

    if pattern:
        # Pattern might be ffmpeg-style like "frames_%06d.png", convert to "frames_*.png".
        pattern = re.sub(r"(%\d+d)", "*", pattern)
        if not (in_files := glob.glob(str(input_dir / pattern))):
            raise FileNotFoundError(f"No files matching {pattern} found in {input_dir}.")
        in_files = natsorted(in_files)
        return input_dir, output_dir, in_files
    return input_dir, output_dir


def _generate_mask_single(in_file, output_dir=None):
    from spsim.io import read_img, write_img

    img, alpha = read_img(in_file, apply_alpha=True)
    mask = (alpha * 255).astype(np.uint8)
    path = output_dir / Path(in_file).with_suffix(".png").name
    return write_img(str(path), mask)


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


# Dynamically populate arguments of the `render-views` task
conf = parser_config()
render_views_help = {arg["name"].lstrip("-"): arg["help"] for arg in conf["arguments"]}
render_views_help["autoexec"] = "if true, enable the execution of bundled code. default: False"
render_views_args = [arg["name"].lstrip("-").replace("-", "_") for arg in conf["arguments"] if "default" not in arg]
render_views_kwargs = {
    arg["name"].lstrip("-").replace("-", "_"): arg["default"] for arg in conf["arguments"] if "default" in arg
}
render_views_kwargs.pop("blend_file")


def render_views(c, blend_file, *args, autoexec=False, **kwargs):
    """Wrapper that calls the `render.py` CLI within blender. All arguments
    are first verified here using invoke and then again using argparse."""
    # TODO: Enable parallelization by spinning up different workers (and blender
    #  instances), each focusing on different frames.
    import sys  # Lazy load

    # Runtime checks and gard rails
    if _run(c, "blender --version", hide=True).failed:
        raise RuntimeError(f"No blender installation found on path!")
    if not (blend_file := Path(blend_file).resolve()).exists():
        raise FileNotFoundError(f"Blender file {blend_file} not found.")
    if "render-views" not in sys.argv[1]:
        raise RuntimeError(f"Task `render-views` must run first if running multiple tasks simultaneously.")

    # Call `render.py` script through blender's python interpreter
    path = Path(__file__).parent / "src" / "spsim" / "render.py"
    autoexec = "--enable-autoexec" if autoexec else "--disable-autoexec"
    cmd = f"blender --background {autoexec} --python {path} -- {' '.join(sys.argv[3:])} --blend-file={blend_file}"
    _run(c, cmd)


render_views_full = modify_signature(
    *render_views_args,
    remove_var_args=True,
    remove_var_kwargs=True,
    # TODO: Make the docstring for invoke usage only here and have a
    #   default docstring within argparse in the `parser_conf` dict.
    docstring=conf["parser"]["prog"],
    **render_views_kwargs,
)(render_views)
render_views = task(help=render_views_help, auto_shortflags=False)(render_views)


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "pattern": "filenames of frames should match this, default: 'frame_%06d.png'",
        "outfile": "where to save generated mp4, default: 'out.mp4'",
        "fps": "frames per second in video, default: 25",
        "crf": "constant rate factor for video encoding (0-51), lower is better quality but more memory, default: 22",
        "vcodec": "video codec to use (either libx264 or libx265), default: libx264",
        "step": "drop some frames when making video, use frames 0+step*n, default: 1",
        "force": "if true, overwrite output file if present, default: False",
        "bg_color": "for images with transparencies, namely PNGs, use this color as a background, default: 'black'",
        "png_filter": "if false, do not pre-process PNGs to remove transparencies, default: True",
        "auto_tonemap": "if true and images are .exr (linear intensity), apply tonemapping first, default: True",
        "hide": "if true, hide ffmpeg output, default: False",
    }
)
def ffmpeg_animate(
    c,
    input_dir,
    pattern="frame_*.png",
    outfile="out.mp4",
    fps=25,
    crf=22,
    vcodec="libx264",
    step=1,
    force=False,
    bg_color="black",
    png_filter=True,
    auto_tonemap=True,
    hide=False,
):
    """Combine generated frames into an MP4 using ffmpeg wizardry"""
    # TODO: Auto-tonemap for depth colorization
    import tempfile  # Lazy import

    from natsort import natsorted

    if _run(c, "ffmpeg -version", hide=True).failed:
        raise RuntimeError(f"No ffmpeg installation found on path!")

    input_dir, output_dir, in_files = _validate_directories(input_dir, Path(outfile).parent, pattern=pattern)

    # See: https://stackoverflow.com/questions/52804749
    png_filter = (
        (
            f'-filter_complex "color={bg_color},format=rgb24[c];[c][0]scale2ref[c][i];'
            f'[c][i]overlay=format=auto:shortest=1,setsar=1" '
        )
        if (pattern.endswith(".png") or (pattern.endswith(".exr") and auto_tonemap)) and png_filter
        else ""
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        # There's no easy way to select out a subset of frames to use.
        # The select filter (-vf "select=not(mod(n\,step))") interferes with
        # the PNG alpha channel removal, and the concat muxer doesn't work
        # with images or leads to errors.
        # As a quick fix, we create a tmpdir with symlinks to the frames we
        # want to include and point ffmpeg to those.

        tmpdirname = Path(tmpdirname)
        ext = str(Path(pattern).suffix)

        if pattern.endswith(".exr") and auto_tonemap:
            # Apply tone mapping (linear -> sRGB) to all files
            # Note: `tonemap_exrs` does not rename files, so we have to do it here...
            print("Applying tonemapping to images...")
            tonemap_exrs(c, input_dir, str(tmpdirname), pattern=pattern, ext=".png", step=step)
            for i, p in enumerate(natsorted(tmpdirname.glob("*.png"))):
                p.rename(tmpdirname / f"{i:09}.png")
            ext = ".png"
        else:
            # No transformation needed, simply symlink files
            for i, p in enumerate(in_files[::step]):
                (tmpdirname / f"{i:09}{ext}").symlink_to(p)

        cmd = (
            f"ffmpeg -framerate {fps} -f image2 -i {tmpdirname / ('%09d' + ext)} {png_filter}"
            f"{'-y' if force else ''} -vcodec {vcodec} -crf {crf} -pix_fmt yuv420p {outfile}"
        )
        _run(c, cmd, hide=hide)


@task(help={"input_file": "video file input"})
def count_frames(c, input_file):
    """Count the number of frames a video file contains using ffprobe"""
    # See: https://stackoverflow.com/questions/2017843
    if _run(c, "ffprobe -version", hide=True).failed:
        raise RuntimeError(f"No ffprobe installation found on path!")

    cmd = (
        f"ffprobe -v error -select_streams v:0 -count_packets -show_entries "
        f"stream=nb_read_packets -of csv=p=0 {input_file}"
    )
    result = _run(c, cmd, hide=True)
    print(f"Video contains {result.stdout.strip()} frames.")


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
        # Open with opencv, convert to color using matplotlib's cmaps and save as png.
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


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save masks",
        "pattern": "filenames of frames should match this, default: 'frame_*.png'",
    }
)
def generate_masks(
    _,
    input_dir,
    output_dir,
    pattern="frame_*.png",
):
    """Extract alpha channel from frames and create PNG masks that COLMAP understands"""
    import multiprocessing

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern)
    in_files = list(filter(lambda p: not Path(p).is_dir(), in_files))
    generate_mask_single = functools.partial(_generate_mask_single, output_dir=output_dir)

    with multiprocessing.Pool() as p:
        tasks = p.imap(generate_mask_single, in_files)
        list(tqdm(tasks, total=len(in_files)))


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
def emulate_spad(
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
def emulate_blur(_, input_dir, output_dir, chunk_size=10, grayscale=False, pattern="frame_*.png", ext=".png"):
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
def emulate_rgb(
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
    """Simulate real camera, adding read/poisson noise and tonemapping."""
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


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save interpolated frames",
        "method": "interpolation method to use, only RIFE (ECCV22) is supported for now",
        "file_name": "name of file containing transforms, default: 'transforms.json'",
        "n": "interpolation factor, must be a multiple of 2, default: 2",
    }
)
def interpolate_ds(_, input_dir, output_dir, method="rife", file_name="transforms.json", n=2):
    """Interpolate between frames and poses (up to 16x) using RIFE (ECCV22)"""
    # TODO: Enable interpolation of only transforms or only frames
    from natsort import natsorted

    from spsim.interpolate import pose_interp, rife

    if method.lower() not in ("rife",):
        raise NotImplementedError(f"Only rife is currently supported as an interpolation method.")
    if n < 2 or not n & (n - 1) == 0:
        raise ValueError(f"Can only interpolate by a power of 2, greater or equal to 2, not {n}.")
    input_dir, output_dir = _validate_directories(input_dir, output_dir)

    with (input_dir / file_name).open("r") as f:
        transforms = json.load(f)

    # Expect either blender or nerf style transforms, giving priority to blender-style.
    img_paths = [f["file_paths"][0] if "file_paths" in f else f["file_path"] for f in transforms["frames"]]
    frames = natsorted(transforms["frames"], key=lambda f: f["file_paths"][0] if "file_paths" in f else f["file_path"])
    img_paths = natsorted(str(input_dir / p) for p in img_paths)
    is_blender = all("file_paths" in f for f in frames)
    is_nerf = all("file_path" in f for f in frames)
    exts = set(Path(p).suffix for p in img_paths)

    if len(exts) != 1:
        raise RuntimeError(f"All images must have same extension but found {exts}.")

    if not is_blender and not is_nerf:
        raise ValueError(f"Format not understood.")

    # Perform pose interpolation
    #   Ex for 4 frames, and n=4:
    #       [0.0, 1.0, 2.0, 3.0, 4.0]
    #       [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]
    num_frames = len(transforms["frames"])
    indices = np.linspace(0, num_frames - 1, num_frames)
    interp_indices = np.linspace(0, num_frames - 1, n * num_frames - (n - 1))
    pose_spline = pose_interp([f["transform_matrix"] for f in frames], ts=indices)
    new_poses = pose_spline(interp_indices)
    keys = set().union(*[f.keys() for f in frames])

    if len(keys) != 2:
        raise RuntimeError(
            f"Expected only two keys per frame ('transform_matrix' and 'file_paths'/'file_path')" f"but got {keys}."
        )

    # Perform image interpolation
    rife(img_paths, output_dir / "frames", exp=np.log2(n).astype(int))

    # Assemble new transforms.json
    new_paths = natsorted(output_dir.glob(f"frames/*{exts.pop()}"))

    if len(new_paths) != len(new_poses):
        raise RuntimeError(
            f"Image and pose mismatch! Found {len(new_poses)} new poses " f"and {len(new_paths)} new images."
        )

    new_frames = [
        {"file_path": str(path), "transform_matrix": pose.tolist()}
        if is_nerf
        else {"file_paths": [str(path)], "transform_matrix": pose.tolist()}
        for path, pose in zip(new_paths, new_poses)
    ]
    transforms["frames"] = new_frames

    with (output_dir / file_name).open("w") as f:
        json.dump(transforms, f, indent=2)


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save colmap results, default='.'",
        "camera_model": "one of SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV. Default: OPENCV",
        "camera_params": "intrinsic parameters, depends on the chosen model. default: ''",
        "mask_dir": "if present use these image masks when extracting features, will generate them "
        "from image alpha channel if needed. default: ''",
        "matcher": "which matcher to use. One of exhaustive, sequential, spatial, transitive, vocab_tree. "
        "Usually, sequential for videos, exhaustive for adhoc images. Default: sequential",
        "vocab_path": "vocabulary tree path, default: None",
        "mapper": "which mapper to use, either '' or 'hierarchical'. default: ''.",
        "dense_reconstruction": "if true, perform dense reconstruction after sparse one. default: False",
        "cuda": "use cuda acceleration where possible, default: True",
        "force": "if true, overwrite output file if present, default: False",
    }
)
def run_colmap(
    c,
    input_dir,
    output_dir,
    camera_model="OPENCV",
    camera_params="",
    mask_dir="",
    matcher="sequential",
    vocab_path=None,
    mapper="",
    dense_reconstruction=False,
    skip_ba=False,
    cuda=True,
    force=False,
):
    """Run colmap on the provided images to get a rough pose/scene estimate"""
    import shutil

    from spsim.cli import StreamWatcherTqdmPbar  # Lazy import

    input_dir, output_dir, in_files = _validate_directories(input_dir, output_dir, pattern="*")
    db = Path(output_dir) / "colmap.db"
    text = Path(output_dir) / "text"
    sparse = Path(output_dir) / "sparse"
    dense = Path(output_dir) / "dense"
    log = Path(output_dir) / "log"
    mask_dir = Path(mask_dir) if mask_dir else None

    if _run(c, "colmap help", hide=True).failed:
        raise RuntimeError(f"No COLMAP installation found on path!")
    if not force and (db.exists() or text.exists() or sparse.exists() or dense.exists()):
        raise RuntimeError(
            f"COLMAP databases {db}, {text}, {sparse} or {dense} already exist. To overwrite rerun with '--force'"
        )

    db.unlink(missing_ok=True)
    shutil.rmtree(str(text), ignore_errors=True)
    shutil.rmtree(str(sparse), ignore_errors=True)
    shutil.rmtree(str(dense), ignore_errors=True)
    shutil.rmtree(str(log), ignore_errors=True)
    text.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)
    dense.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)

    # Generate masks if not present at mask_dir
    if mask_dir and not mask_dir.exists():
        print(f"No masks were found. Generating them...")
        generate_masks(c, input_dir, mask_dir, pattern="*")

    # If camera params is the path to a valid json transforms file (as generated via render-views)
    # then load it up and extract ground truth camera parameters from it.
    if camera_params and camera_params.endswith(".json"):
        if Path(camera_params).exists():
            print(f"\nArgument `camera_params` points to a json file, extracting camera model...")

            with open(camera_params, "r") as f:
                camera_params_data = json.load(f).get("camera")
                params = [camera_params_data.get(k) for k in ("fx", "fy", "cx", "cy")]
                if any(v is None for v in params):
                    raise ValueError(f"Missing one of fx, fy, cx, cy in {camera_params}.")
                camera_params = ", ".join(str(i) for i in params + [0, 0, 0, 0])
                print("Using parameters", camera_params)
        else:
            raise FileNotFoundError("Supplied `camera_params` path not found!")

    # CPU only as we are estimating affine shapes.
    # See: https://github.com/colmap/colmap/issues/1110
    sift_cmd = (
        f"colmap feature_extractor --ImageReader.camera_model {camera_model.upper()} "
        f'--ImageReader.camera_params "{camera_params}" '
        f"--SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true "
        f"--ImageReader.single_camera 1 --database_path {db} --image_path {input_dir} "
        f"--SiftExtraction.use_gpu={'false' if not cuda else 'true'} "
        f"--SiftExtraction.peak_threshold=0.001 "
    )
    sift_cmd += f"--ImageReader.mask_path {mask_dir} " if mask_dir else ""

    match_cmd = (
        f"colmap {matcher}_matcher --SiftMatching.guided_matching=true --database_path {db} "
        f"--SiftMatching.use_gpu={'false' if not cuda else 'true'} "  # GPU requires attached display
        # f"--SiftMatching.max_error=20 --SiftMatching.max_distance=20.0 --SiftMatching.confidence=0.0 "
        # f"--SiftMatching.min_inlier_ratio=0.1 --SiftMatching.min_num_inliers=4"
    )
    match_cmd += f" --VocabTreeMatching.vocab_tree_path {vocab_path}" if vocab_path else ""

    # Mapper can't run on GPU either, only `hierarchical_mapper` can but results in poor reconstruction.
    mapper_cmd = f"colmap {mapper + '_' if mapper else ''}mapper "
    mapper_cmd += f"--database_path {db} --image_path {input_dir} --output_path {sparse} "
    mapper_cmd += (
        f"--Mapper.ba_refine_focal_length=0 --Mapper.ba_refine_principal_point=0 "
        f"--Mapper.ba_refine_extra_params=0 "
        f"--Mapper.min_num_matches=5 --Mapper.init_min_num_inliers=15 "
        # if camera_params
        # else ""
    )

    ba_cmd = (
        f"colmap bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 "
        f"--BundleAdjustment.refine_principal_point {int(not bool(camera_params))}"
    )

    convert_cmd = f"colmap model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT"

    undistort_cmd = (
        f"colmap image_undistorter --image_path {input_dir} --input_path {sparse}/0 "
        f"--output_path {dense} --output_type COLMAP --max_image_size 2000"
    )

    patch_match_cmd = (
        f"colmap patch_match_stereo --workspace_path {dense} "
        f"--workspace_format COLMAP --PatchMatchStereo.geom_consistency true"
    )

    fusion_cmd = (
        f"colmap stereo_fusion --workspace_path {dense} --workspace_format COLMAP "
        f"--input_type geometric --output_path {dense}/fused.ply"
    )

    poisson_mesher_cmd = f"colmap poisson_mesher --input_path {dense}/fused.ply --output_path {dense}/meshed-poisson.ply"

    delaunay_mesher_cmd = f"colmap delaunay_mesher --input_path {dense} --output_path {dense}/meshed-delaunay.ply"

    print(
        f"\nRunning colmap with:\n"
        f"  -images from: {input_dir}\n"
        f"  -output will be in: {output_dir}\n"
        f"  -logs will be saved in: {log}\n"
    )
    print(f"\nFound {len(in_files)} files in {input_dir}...")

    print("\nExtracting Features...")
    with StreamWatcherTqdmPbar(r"Processed file \[(?P<n>\d+)/\d+\]", total=len(in_files)) as pbar_watcher:
        _log_run(c, sift_cmd, log / "feature_extractor", watchers=pbar_watcher)

    print("\nMatching Features...")
    with StreamWatcherTqdmPbar(r"Matching image \[\d+/\d+\]", total=len(in_files)) as pbar_watcher:
        _log_run(c, match_cmd, log / f"{matcher}_matcher", watchers=pbar_watcher)

    print("\nRunning Mapper...")
    with StreamWatcherTqdmPbar(
        r"Registering image #\d+ \((?P<n>\d+)\)",
        on_match={
            "No good initial image pair found.": functools.partial(
                _raise_callback, err_type=RuntimeError, message="No good initial image pair found."
            )
        },
        total=len(in_files),
    ) as pbar_watcher:
        _log_run(c, mapper_cmd, log / "mapper", watchers=pbar_watcher)

    if not skip_ba:
        print("\nBundle Adjusting...")
        _log_run(c, ba_cmd, log / "bundle_adjuster")
    else:
        print("\nSkipping bundle adjustments...")
        if not camera_params:
            print("WARNING: Skipping BA is not recommended if no camera parameters are specified.")

    print("\nConverting results to text format...")
    _log_run(c, convert_cmd, log / "convert")

    print("\nConverting results to NeRF format...")
    colmap_to_nerf_format(c, input_dir, text, keep_colmap_coords=False, aabb_scale=16)

    # Early exit if needed
    if not dense_reconstruction:
        return

    print("\nStarting dense reconstruction process (this will take a while)...")
    print("\nUndistorting Images...")
    with StreamWatcherTqdmPbar(r"Undistorting image \[(?P<n>\d+)/\d+\]", total=len(in_files)) as pbar_watcher:
        _log_run(c, undistort_cmd, log / "undistorter", watchers=pbar_watcher)

    print("\nMatching Image Patches...")
    # This is twice as long as it matches based on geometric and photometric loss
    with StreamWatcherTqdmPbar(r"Processing view \d+ / \d+", total=len(in_files) * 2) as pbar_watcher:
        _log_run(c, patch_match_cmd, log / "patch_match", watchers=pbar_watcher)

    print("\nFusing Image Patches...")
    with StreamWatcherTqdmPbar(r"Fusing image \[(?P<n>\d+)/\d+\]", total=len(in_files)) as pbar_watcher:
        _log_run(c, fusion_cmd, log / "fusion", watchers=pbar_watcher)

    print("\nConverting results to a Poisson Mesh...")
    _log_run(c, poisson_mesher_cmd, log / "poisson_mesher")

    print("\nConverting results to a Delaunay Mesh...")
    with StreamWatcherTqdmPbar(r"Integrating image \[(?P<n>\d+)/\d+\]", total=len(in_files)) as pbar_watcher:
        _log_run(c, delaunay_mesher_cmd, log / "delaunay_mesher", watchers=pbar_watcher)


@task
def colmap_to_nerf_format(
    _,
    input_dir,
    text_dir,
    keep_colmap_coords=False,
    aabb_scale=16,
    indices=None,
    file_name="transforms.json",
    sharpness=False,
):
    """Convert transform.json from colmap format to nerf-style format"""
    import ast

    from spsim.colmaptools import convert_from_colmap  # Lazy import

    transforms_path = (Path(text_dir).parent / file_name).resolve()
    transforms_path.unlink(missing_ok=True)

    convert_from_colmap(
        input_dir,
        text_dir,
        transforms_path,
        aabb_scale=aabb_scale,
        indices=ast.literal_eval(indices) if indices else slice(None),
        keep_colmap_coords=keep_colmap_coords,
        sharpness=sharpness,
    )


@task
def blender_to_nerf_format(
    _,
    input_dir,
    infile="transforms_blender.json",
    outfile="transforms.json",
    aabb_scale=16,
    sharpness=False,
):
    """Convert transform.json from blender format to nerf-style format

    Flatten "camera" entry, add "w", "h", "fl_x" and "fl_y", and convert "file_paths" to "file_path"
    """
    from spsim.colmaptools import compute_sharpness

    if Path(input_dir).is_file():
        print("Path provided is file, ignoring `infile` argument.")
        infile = Path(input_dir).name
        input_dir = Path(input_dir).parent

    transforms_in_path = (Path(input_dir) / infile).resolve()
    transforms_out_path = (Path(input_dir) / outfile).resolve()

    if transforms_out_path == transforms_in_path:
        raise RuntimeError("Input and output files are the same!")
    if transforms_out_path.exists():
        raise RuntimeError(f"Output file {transforms_out_path} already exists.")

    with transforms_in_path.open("r") as f:
        out = json.load(f)

    # Flatten out camera section
    camera = out.pop("camera")
    camera["cx"] = camera.get("cx", np.array(camera["intrinsics"])[0, 2])
    camera["cy"] = camera.get("cy", np.array(camera["intrinsics"])[1, 2])
    out.update(camera)

    if camera["type"] != "PERSP":
        raise NotImplementedError("Only perspective cameras are supported!")

    out["aabb_scale"] = aabb_scale
    out["w"] = 2 * (camera["cx"] - camera["shift_x"])
    out["h"] = 2 * (camera["cy"] - camera["shift_y"])
    out["fl_x"] = camera["fx"]
    out["fl_y"] = camera["fy"]

    out["frames"] = [
        {
            "file_path": frame["file_paths"][0],
            "sharpness": compute_sharpness(str(Path(input_dir) / frame["file_paths"][0])) if sharpness else None,
            "transform_matrix": frame["transform_matrix"],
        }
        for frame in out["frames"]
    ]

    with transforms_out_path.open("w") as outfile:
        json.dump(out, outfile, indent=2)


@task(
    help={
        "input_dir": "directory in which to look for frames",
        "output_dir": "directory in which to save npy file",
        "file_name": "name of file containing transforms, default: 'transforms.json'",
    }
)
def frames_to_npy(_, input_dir, output_dir, file_name="transforms.json"):
    """Convert an image folder based dataset to a NPY dataset (experimental)"""
    # TODO: Deal with alpha channel, bitpack if binary data.
    import imageio.v3 as iio
    from natsort import natsorted
    from numpy.lib.format import open_memmap
    from tqdm.auto import tqdm

    input_dir, output_dir = _validate_directories(input_dir, output_dir)

    with (input_dir / file_name).open("r") as f:
        transforms = json.load(f)

    # Expect either blender or nerf style transforms, giving priority to blender-style.
    img_paths = [f["file_paths"][0] if "file_paths" in f else f["file_path"] for f in transforms["frames"]]
    frames = natsorted(transforms["frames"], key=lambda f: f["file_paths"][0] if "file_paths" in f else f["file_path"])
    img_paths = natsorted(str(input_dir / p) for p in img_paths)

    w = transforms["w"] if "w" in transforms else 2 * (transforms["camera"]["cx"] - transforms["camera"]["shift_x"])
    h = transforms["h"] if "h" in transforms else 2 * (transforms["camera"]["cy"] - transforms["camera"]["shift_y"])

    frames_array = open_memmap(
        output_dir / "frames.npy", mode="w+", dtype=np.uint8, shape=(len(img_paths), int(h), int(w), 3)
    )

    for i, path in enumerate(tqdm(img_paths)):
        frames_array[i] = iio.imread(path)[:, :, :3]

    transforms["file_path"] = "frames.npy"
    new_frames = [{k: v for k, v in f.items() if k not in ("file_path", "file_paths")} for f in frames]
    transforms["frames"] = new_frames

    with (output_dir / file_name).open("w") as f:
        json.dump(transforms, f, indent=2)
