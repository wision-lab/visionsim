from __future__ import annotations

import os
from dataclasses import asdict

from visionsim.simulate.blender import BlenderClient, BlenderClients
from visionsim.simulate.config import RenderConfig
from visionsim.types import UpdateFn


def render_job(
    client: BlenderClient | BlenderClients,
    blend_file: str | os.PathLike,
    root: str | os.PathLike,
    config: RenderConfig,
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
    frame_step: int | None = None,
    output_blend_file: str | os.PathLike | None = None,
    dry_run: bool = False,
    update_fn: UpdateFn | None = None,
) -> None:
    """Render a sequence from a given blender-file.

    Args:
        client (BlenderClient | BlenderClients): The blender client(s) which will be used for rendering.
            These should already be connected to a ``BlenderServer``, and will get automagically passed
            in when using this function with ``BlenderClients.pool`` or similar.
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
            Will first be called with ``total`` kwarg, indicating number of steps to be taken,
            then will be called with ``advance=1`` at every step. Closely mirrors the `rich.Progress
            API <https://rich.readthedocs.io/en/stable/reference/progress.html#rich.progress.Progress.update>`_.
    """
    client.initialize(blend_file, root)
    client.set_resolution(height=config.height, width=config.width, resolution_percentage=config.resolution_percentage)
    client.use_animations(config.use_animations)
    client.load_addons(*(config.addons or []))

    client.cycles_settings(
        device_type=config.device_type,
        adaptive_threshold=config.adaptive_threshold,
        use_denoising=config.use_denoising,
        max_samples=config.max_samples,
        use_cpu=True,
    )

    if config.include_composites:
        client.include_composites(**asdict(config.composites))
    if config.include_frames:
        client.include_frames(**asdict(config.frames))
    if config.include_depths:
        client.include_depths(**asdict(config.depths))
    if config.include_normals:
        client.include_normals(**asdict(config.normals))
    if config.include_flows:
        client.include_flows(**asdict(config.flows))
    if config.include_segmentations:
        client.include_segmentations(**asdict(config.segmentations))
    if config.include_materials:
        client.include_materials(**asdict(config.materials))
    if config.include_diffuse_pass:
        client.include_diffuse_pass(**asdict(config.diffuse_pass))
    if config.include_specular_pass:
        client.include_specular_pass(**asdict(config.specular_pass))
    if config.include_points:
        client.include_points(**asdict(config.points))
    if config.include_thermal:
        client.prepare_thermal(**asdict(config.thermal))
        client.include_thermal(**asdict(config.thermal))

    if config.unbind_camera:
        client.unbind_camera()
    if config.use_motion_blur is not None:
        client.use_motion_blur(config.use_motion_blur)
    if config.camera_offset is not None:
        for frame_number in client.common_animation_range():
            client.set_current_frame(frame_number)
            client.set_camera_keyframe(frame_number)
        for frame_number in client.common_animation_range():
            client.set_current_frame(frame_number)
            client.offset_camera(config.camera_offset)
            client.set_camera_keyframe(frame_number)
    client.move_keyframes(scale=config.keyframe_multiplier)

    if output_blend_file is not None:
        client.save_file(output_blend_file)

    client.render_animation(
        frame_start=frame_start,
        frame_end=frame_end,
        frame_step=frame_step,
        allow_skips=config.allow_skips,
        dry_run=dry_run,
        update_fn=update_fn,
    )
