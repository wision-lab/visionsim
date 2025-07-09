from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import tyro
from typing_extensions import Annotated, List, Literal, TypeAlias

from visionsim.dataset import Dataset
from visionsim.simulate.blender import BlenderClient, BlenderClients
from visionsim.types import UpdateFn
from visionsim.utils.progress import ElapsedProgress

_SIZE_SYMBOLS = ("B", "K", "M", "G", "T", "P", "E", "Z", "Y")
_SIZE_BOUNDS = [(1024**i, sym) for i, sym in enumerate(_SIZE_SYMBOLS)]
_SIZE_DICT = {sym: val for val, sym in _SIZE_BOUNDS}
_SIZE_RANGES = list(zip(_SIZE_BOUNDS, _SIZE_BOUNDS[1:]))


def _bytes_from_str(size: str | List[str]) -> int:
    # Based on https://stackoverflow.com/a/60708339
    if isinstance(size, list):
        assert len(size) == 1
        return _bytes_from_str(size[0])
    try:
        return int(size)
    except ValueError:
        size = size.upper()
        if not re.match(r" ", size):
            symbols = "".join(_SIZE_SYMBOLS)
            size = re.sub(rf"([{symbols}]?B)", r" \1", size)
        number, unit = [string.strip() for string in size.split()]
        return int(float(number) * _SIZE_DICT[unit[0]])


def _bytes_to_str(nbytes: int, ndigits: int = 1) -> str:
    # Based on https://boltons.readthedocs.io/en/latest/strutils.html#boltons.strutils.bytes2human
    abs_bytes = abs(nbytes)
    for (size, symbol), (next_size, _) in _SIZE_RANGES:
        if abs_bytes <= next_size:
            break
    hnbytes = float(nbytes) / size
    return f"{hnbytes:.{ndigits}f}{symbol}"


MemSize: TypeAlias = Annotated[
    int,
    tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="BYTES",
        instance_from_str=_bytes_from_str,
        is_instance=lambda instance: isinstance(instance, int),
        str_from_instance=_bytes_to_str,
    ),
]


@dataclass
class RenderConfig:
    executable: str | os.PathLike | None = None
    """Path to blender executable"""
    height: int = 512
    """Height of rendered frames"""
    width: int = 512
    """Width of rendered frames"""
    bit_depth: int = 8
    """Bit depth for intensity frames. Usually 8 for pngs, 32 or 16 bits for OPEN_EXR"""
    file_format: str = "PNG"
    """File format to use for intensity frames"""
    depths: bool = False
    """If true, enable depth map outputs"""
    normals: bool = False
    """If true, enable normal map outputs"""
    flows: bool = False
    """If true, enable optical flow outputs"""
    flow_direction: Literal["forward", "backward", "both"] = "forward"
    """Direction of flow to colorize for debug visualization. Only used when debug is true"""
    segmentations: bool = False
    """If true, enable segmentation map outputs"""
    debug: bool = True
    """If true, also save debug visualizations for auxiliary outputs"""
    keyframe_multiplier: float = 1.0
    """Stretch keyframes by this amount, eg: 2.0 will slow down time"""
    timeout: int = -1
    """Maximum allowed time in seconds to wait to connect to render instance"""
    autoexec: bool = True
    """If true, allow python execution of embedded scripts (warning: potentially dangerous)"""
    device_type: Literal["cpu", "cuda", "optix", "metal"] = "optix"
    """Name of device to use, one of "cpu", "cuda", "optix", "metal", etc"""
    adaptive_threshold: float = 0.05
    """Noise threshold of rendered images, for higher quality frames make this threshold smaller. 
    The default value is intentionally a little high to speed up renders"""
    max_samples: int = 256
    """Maximum number of samples per pixel to take"""
    use_denoising: bool = True
    """If enabled, a denoising pass will be used"""
    log_dir: str | os.PathLike = "logs/"
    """Directory to use for logging"""
    allow_skips: bool = True
    """If true, skip rendering a frame if it already exists"""
    unbind_camera: bool = False
    """Free the camera from it's parents, any constraints and animations it may have. 
    Ensures it uses the world's coordinate frame and the provided camera trajectory"""
    use_animations: bool = True
    """Allow any animations to play out, if false, scene will be static"""
    use_motion_blur: bool | None = None
    """Enable realistic motion blur. cannot be used if also rendering optical flow"""
    addons: list[str] | None = None
    """List of extra addons to enable"""
    jobs: int = 1
    """Number of concurrent render jobs"""
    autoscale: bool = False
    """Set number of jobs automatically based on available VRAM and `max_job_vram` when enabled"""
    max_job_vram: MemSize | None = None
    """Maximum allowable VRAM per job in bytes (limit is not enforced, simply used for `autoscale`)"""


def _render_job(
    client: BlenderClient | BlenderClients,
    blend_file: str | os.PathLike,
    root: str | os.PathLike,
    *,
    config: RenderConfig,
    frame_start: int | None = None,
    frame_end: int | None = None,
    frame_step: int | None = None,
    output_blend_file: str | os.PathLike | None = None,
    dry_run: bool = False,
    update_fn: UpdateFn | None = None,
):
    """Render a sequence from a given blender-file.

    Args:
        client (BlenderClient | BlenderClients): The blender client(s) which will be used for rendering.
            These should already be connected to a `BlenderServer`, and will get automagically passed
            in when using this function with `BlenderClients.pool` or similar.
        blend_file (str | os.PathLike): Path to blender file to use.
        root (str | os.PathLike): Location at which to save all outputs.
        config (RenderConfig): Render configuration.
        frame_start (int | None, optional): Frame index to start capture at (inclusive).
            If None, use start of animation range.
        frame_end (int | None, optional): frame number to stop capture at (inclusive).
            If None, use end of animation range.
        frame_step (int | None, optional): Step with which to capture frames.
            If None, use step of animation range.
        output_blend_file (str | os.PathLike | None, optional): If set, write the modified blend file to
            this path. Helpful for troubleshooting. Defaults to not saving.
        dry_run (bool, optional): If enabled, do not render any frames or ground truth annotations.
        update_fn (UpdateFn | None, optional): callback function to track render progress.
            Will first be called with `total` kwarg, indicating number of steps to be taken,
            then will be called with `advance=1` at every step. Closely mirrors the `rich.Progress
            API <https://rich.readthedocs.io/en/stable/reference/progress.html#rich.progress.Progress.update>`_.
    """
    client.initialize(blend_file, root)
    client.set_resolution(height=config.height, width=config.width)
    client.image_settings(file_format=config.file_format, bit_depth=config.bit_depth)
    client.use_animations(config.use_animations)
    client.load_addons(*(config.addons or []))

    client.cycles_settings(
        device_type=config.device_type,
        adaptive_threshold=config.adaptive_threshold,
        use_denoising=config.use_denoising,
        max_samples=config.max_samples,
        use_cpu=True,
    )

    if config.depths:
        client.include_depths(debug=config.debug)
    if config.normals:
        client.include_normals(debug=config.debug)
    if config.flows:
        client.include_flows(debug=config.debug, direction=config.flow_direction)
    if config.segmentations:
        client.include_segmentations(debug=config.debug)

    if config.unbind_camera:
        client.unbind_camera()
    if config.use_motion_blur is not None:
        client.use_motion_blur(config.use_motion_blur)

    client.move_keyframes(scale=config.keyframe_multiplier)

    if output_blend_file is not None:
        client.save_file(output_blend_file)

    transforms = client.render_animation(
        frame_start=frame_start,
        frame_end=frame_end,
        frame_step=frame_step,
        allow_skips=config.allow_skips,
        dry_run=dry_run,
        update_fn=update_fn,
    )

    with open(root / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=2)


def sequence_info(
    dataset: str | os.PathLike,
    keyframe_multiplier: float = 1.0,
    original_fps: int = 50,
    output: str | os.PathLike | None = None,
):
    """Query dataset to collect some extra metadata, write it to a json file

    Args:
        dataset (str | os.PathLike): Root pathy of dataset
        keyframe_multiplier (float, optional): Keyframe stretch amount.
        original_fps (int, optional): Framerate of native blender animation. Defaults to 50fps.
        output (str | os.PathLike | None, optional): Path of output info file. Defaults to
            "info.json" in the dataset's root directory.
    """
    ds = Dataset.from_path(dataset)
    dt = len(ds) / (original_fps * keyframe_multiplier)
    output = Path(dataset) / "info.json" if output is None else output

    info = {
        "frame_rate": int(original_fps * keyframe_multiplier),
        "distance_traveled": ds.arclength,
        "average_velocity": ds.arclength / dt,
        "elapsed_time": dt,
    }

    with open(output, "w") as f:
        json.dump(info, f, indent=2)


def render_animation(
    blend_file: str | os.PathLike,
    root_path: str | os.PathLike,
    /,
    render_config: RenderConfig,
    frame_start: int | None = None,
    frame_end: int | None = None,
    dry_run: bool = False,
):
    """Create datasets by rendering out a sequence from a _single_ blend-file.

    Args:
        blend_file (str | os.PathLike): Path to blend file.
        root_path (str | os.PathLike): Dataset output folder.
        render_config (RenderConfig): Render configuration.
        frame_start (int): Start rendering at this frame index (inclusive).
        frame_end (int): Stop rendering at this frame index (inclusive).
        dry_run (bool, optional): if true, nothing will be rendered at all. Defaults to False.
    """
    from visionsim.cli import _log, _run  # avoid circular import

    # Runtime checks and gard rails
    if _run(f"{render_config.executable or 'blender'} --version", shell=True).returncode != 0:
        raise RuntimeError("No blender installation found on path!")
    if not (blend_file := Path(blend_file).resolve()).exists():
        raise FileNotFoundError(f"Blender file {blend_file} not found.")

    root_path = Path(root_path).resolve()
    root_path.mkdir(parents=True, exist_ok=True)

    if render_config.autoscale:
        if torch.cuda.device_count() != 1:
            _log.warning("Cannot autoscale when using multi-gpu. Falling back on using a single render job.")
            render_config.autoscale = False
            render_config.max_job_vram = None
            render_config.jobs = 1
        else:
            idx = torch.cuda.current_device()
            device = torch.device(idx)
            free, _ = torch.cuda.mem_get_info(device)
            render_config.jobs = free // render_config.max_job_vram
            _log.info(f"Auto-scaling to using {render_config.jobs} render jobs on {torch.cuda.get_device_name(idx)}.")

    if render_config.jobs <= 0:
        raise RuntimeError(f"At least one render job is needed, got `render_config.jobs={render_config.jobs}`.")

    with (
        BlenderClients.spawn(
            jobs=render_config.jobs,
            log_dir=Path(render_config.log_dir),
            timeout=render_config.timeout,
            executable=render_config.executable,
            autoexec=render_config.autoexec,
        ) as clients,
        ElapsedProgress() as progress,
    ):
        task = progress.add_task(f"Rendering {Path(blend_file).stem}...")
        _render_job(
            clients,
            blend_file,
            root_path,
            frame_start=frame_start,
            frame_end=frame_end,
            config=render_config,
            dry_run=dry_run,
            update_fn=partial(progress.update, task),
        )
        original_fps, *_ = clients.original_fps()
    sequence_info(root_path, keyframe_multiplier=render_config.keyframe_multiplier, original_fps=original_fps)


# TODO: enable using spline to move camera, i.e:
# "num_frames": "number of frame to capture, this argument is affected by `keyframe-multiplier`. default: 100",
# "location_points": (
#     "points defining the spline the camera follows. Expected to be json-str or path "
#     "to json file. Default is circular obit at Z=1 with radius=5."
# ),
# "viewing-points": (
#     "points defining the spline the camera looks at. Expected to be json-str or path "
#     "to json file. Default is static origin."
# ),
