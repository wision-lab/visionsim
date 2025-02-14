import json
import sys
from functools import partial
from pathlib import Path

from invoke import task
from rich.progress import Progress

from spsim.simulate.blender import BlenderClients
from spsim.tasks.common import _run


@task(
    help={
        "blend_file": "path to blender file to use",
        "root_path": "location at which to save dataset",
        "frame_start": "frame number to start capture at (inclusive), default: None",
        "frame_end": "frame number to stop capture at (exclusive), default: None",
        "frame_step": "step with which to capture frames, default: 1",
        "height": "height of frame to capture, default: None",
        "width": "width of frame to capture, default: None",
        "bit_depth": "bit depth for frames, usually 8 for pngs, default: 8",
        "device": "which device type to use, one of none (meaning no change), cpu, cuda, optix. default: None",
        "dry_run": "if enabled, do not render frames, default: False",
        "jobs": "number of blender instances to spawn and render from, default: 1",
        "unbind_camera": (
            "free the camera from it's parents, any constraints and animations it may have. Ensures it "
            "uses the world's coordinate frame and the provided camera trajectory. default: False"
        ),
        "use_animations": "allow any animations to play out, if false, scene will be static. default: True",
        "use_motion_blur": "enable realistic motion blur, default: None",
        "keyframe_multiplier": "slow down animations by this factor, default: 1.0 (no slowdown)",
        "allow_skips": "whether or not to skip rendering a frame if it already exists, default: True",
        "depths": "whether or not to capture depth images, default: False",
        "normals": "whether or not to capture normals images, default: False",
        "flows": "whether or not to capture optical flow images, default: False",
        "segmentations": "whether or not to capture segmentation images, default: False",
        "file_format": (
            "frame file format to use. Depth is always 'OPEN_EXR' thus is unaffected by this setting, default: PNG"
        ),
        "log_dir": "where to save log to, default: None (no log is saved)",
        "addons": "list of extra addons to enable, default: None",
        "adaptive_threshold": (
            "noise threshold of rendered images, for higher quality frames make this threshold smaller. "
            "The default value is intentionally a little high to speed up renders. default: 0.1"
        ),
        "autoexec": "if enabled, allow any embedded python scripts to run, default: False",
        "executable": "use a different blender executable that the one on PATH, default: None",
    },
    auto_shortflags=False,
    iterable=["addons"],
)
def render_animation(
    c,
    blend_file,
    root_path,
    frame_start=None,
    frame_end=None,
    frame_step=1,
    height=None,
    width=None,
    bit_depth=8,
    device=None,
    dry_run=False,
    jobs=1,
    unbind_camera=False,
    use_animations=True,
    use_motion_blur=None,
    keyframe_multiplier=1.0,
    allow_skips=True,
    depths=False,
    normals=False,
    flows=False,
    segmentations=False,
    file_format="PNG",
    log_dir=None,
    addons=None,
    adaptive_threshold=None,
    autoexec=False,
    executable=None,
):
    """Render views of a .blend file while moving camera along an animated trajectory
    Example:
        spsim blender.render-animation <blend-file> <output-path>
    """

    # Runtime checks and gard rails
    if _run(c, f"{executable or 'blender'} --version", hide=True).failed:
        raise RuntimeError("No blender installation found on path!")
    if not (blend_file := Path(blend_file).resolve()).exists():
        raise FileNotFoundError(f"Blender file {blend_file} not found.")
    if "blender.render" not in sys.argv[1]:
        raise RuntimeError("Task `blender.render-animation` must run first if running multiple tasks simultaneously.")

    with BlenderClients.spawn(
        jobs=jobs, timeout=30, autoexec=autoexec, log_dir=log_dir, executable=executable
    ) as clients, Progress() as progress:
        clients.initialize(blend_file, Path(root_path).resolve())
        clients.set_resolution(width=width, height=height)
        clients.image_settings(file_format, bit_depth)
        clients.use_animations(use_animations)
        clients.load_addons(*(addons or []))

        clients.cycles_settings(
            device_type=device,
            use_cpu=True,
            adaptive_threshold=adaptive_threshold,
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
