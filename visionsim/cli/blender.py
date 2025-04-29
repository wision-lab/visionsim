from __future__ import annotations

import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import Literal


def render_animation(
    blend_file: str | os.PathLike,
    root_path: str | os.PathLike,
    frame_start: int | None = None,
    frame_end: int | None = None,
    frame_step: int = 1,
    height: int | None = None,
    width: int | None = None,
    bit_depth: int = 8,
    device: Literal["cpu", "cuda", "optix", "metal"] | None = None,
    dry_run: bool = False,
    jobs: int = 1,
    unbind_camera: bool = False,
    use_animations: bool = True,
    use_motion_blur: bool | None = None,
    keyframe_multiplier: float = 1.0,
    allow_skips: bool = True,
    depths: bool = False,
    normals: bool = False,
    flows: bool = False,
    segmentations: bool = False,
    file_format: str = "PNG",
    log_dir: str | None = None,
    addons: list[str] | None = None,
    adaptive_threshold: float | None = None,
    autoexec: bool = False,
    output_blend_file: str | os.PathLike | None = None,
    executable: str | os.PathLike | None = None,
):
    """Render views of a .blend file while moving camera along an animated trajectory

    Args:
        blend_file: path to blender file to use
        root_path: location at which to save dataset
        frame_start: frame number to start capture at (inclusive)
        frame_end: frame number to stop capture at (exclusive)
        frame_step: step with which to capture frames
        height: height of frame to capture
        width: width of frame to capture
        bit_depth: bit depth for frames, usually 8 for pngs
        device: which device type to use, one of none (meaning no change), cpu, cuda, optix
        dry_run: if enabled, do not render frames
        jobs: number of blender instances to spawn and render from
        unbind_camera: free the camera from it's parents, any constraints and animations it may have. Ensures it uses the world's coordinate frame and the provided camera trajectory
        use_animations: allow any animations to play out, if false, scene will be static
        use_motion_blur: enable realistic motion blur
        keyframe_multiplier: slow down animations by this factor
        allow_skips: whether or not to skip rendering a frame if it already exists
        depths: whether or not to capture depth images
        normals: whether or not to capture normals images
        flows: whether or not to capture optical flow images
        segmentations: whether or not to capture segmentation images
        file_format: frame file format to use. Depth is always 'OPEN_EXR' thus is unaffected by this setting
        log_dir: where to save log to
        addons: list of extra addons to enable
        adaptive_threshold: noise threshold of rendered images, for higher quality frames make this threshold smaller. The default value is intentionally a little high to speed up renders
        autoexec: if enabled, allow any embedded python scripts to run
        output_blend_file: if set, write the modified blend file to this path. Helpful for troubleshooting
        executable: use a different blender executable that the one on PATH

    Example:
        visionsim blender.render_animation --blend-file=<blend-file> --root-path=<output-path>
    """
    from rich.progress import Progress

    from visionsim.cli import _run
    from visionsim.simulate.blender import BlenderClients

    # Runtime checks and gard rails
    if _run(f"{executable or 'blender'} --version", shell=True).returncode != 0:
        raise RuntimeError("No blender installation found on path!")
    if not (blend_file := Path(blend_file).resolve()).exists():
        raise FileNotFoundError(f"Blender file {blend_file} not found.")
    if "blender.render_animation" not in sys.argv[1]:
        raise RuntimeError("Task `blender.render_animation` must run first if running multiple tasks simultaneously.")

    with (
        BlenderClients.spawn(
            jobs=jobs, timeout=30, autoexec=autoexec, log_dir=log_dir, executable=executable
        ) as clients,
        Progress() as progress,
    ):
        clients.initialize(blend_file, Path(root_path).resolve())
        clients.set_resolution(width=width, height=height)
        clients.image_settings(file_format, bit_depth)
        clients.use_animations(use_animations)
        clients.load_addons(*(addons or []))

        clients.cycles_settings(
            device_type=device,
            use_cpu=True,
            adaptive_threshold=adaptive_threshold if adaptive_threshold is None else float(adaptive_threshold),
            use_denoising=True,
        )

        if depths:
            clients.include_depths()
        if normals:
            clients.include_normals()
        if flows:
            clients.include_flows(direction="both")
        if segmentations:
            clients.include_segmentations()

        if unbind_camera:
            clients.unbind_camera()
        if use_motion_blur is not None:
            clients.use_motion_blur(use_motion_blur)

        clients.move_keyframes(scale=keyframe_multiplier)

        if output_blend_file is not None:
            output_blend_file = Path(output_blend_file).resolve()
            print(f"Saving to {output_blend_file}...")
            clients[0].save_file(output_blend_file)

        task = progress.add_task(f"Rendering {Path(blend_file).name}...")
        transforms = clients.render_animation(
            frame_start=frame_start if frame_start is None else int(frame_start),
            frame_end=frame_end if frame_end is None else int(frame_end),
            frame_step=frame_step,
            allow_skips=allow_skips,
            dry_run=dry_run,
            update_fn=partial(progress.update, task, advance=1),
        )

        with open(str(Path(root_path) / "transforms.json"), "w") as f:
            json.dump(transforms, f, indent=2)


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
