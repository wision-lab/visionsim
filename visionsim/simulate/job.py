import json
import os
from pathlib import Path

from visionsim.simulate.blender import BlenderClient, BlenderClients
from visionsim.simulate.config import RenderConfig
from visionsim.types import UpdateFn


def render_job(
    client: BlenderClient | BlenderClients,
    blend_file: str | os.PathLike,
    root: str | os.PathLike,
    *,
    config: RenderConfig = RenderConfig(),
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
        client.include_depths(debug=config.debug, exr_codec=config.exr_codec)
    if config.normals:
        client.include_normals(debug=config.debug, exr_codec=config.exr_codec)
    if config.flows:
        client.include_flows(debug=config.debug, direction=config.flow_direction, exr_codec=config.exr_codec)
    if config.segmentations:
        client.include_segmentations(debug=config.debug, exr_codec=config.exr_codec)

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

    with open(Path(root) / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=2)
