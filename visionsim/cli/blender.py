from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from visionsim.simulate.config import RenderConfig


def render_animation(
    blend_file: Path,
    output_dir: Path,
    /,
    config: RenderConfig,
    frame_start: int | None = None,
    frame_end: int | None = None,
    output_file: Path | None = None,
    dry_run: bool = False,
) -> None:
    """Create datasets by rendering out a sequence from a single blend-file.

    Args:
        blend_file: Path to blend file.
        output_dir: Dataset output folder.
        config: Render configuration.
        frame_start: Start rendering at this frame index (inclusive).
        frame_end: Stop rendering at this frame index (inclusive).
        output_file: If set, write the modified blend file to
            this path. Helpful for troubleshooting. Defaults to not saving.
        dry_run: if true, nothing will be rendered at all. Defaults to False.
    """
    from visionsim.cli import _log, _run  # avoid circular import
    from visionsim.simulate.blender import BlenderClients
    from visionsim.simulate.job import render_job
    from visionsim.utils.progress import ElapsedProgress

    # Runtime checks and gard rails
    if _run(f"{config.executable or 'blender'} --version", shell=True, hide=True).returncode != 0:
        raise RuntimeError("No blender installation found on path!")
    if not (blend_file := blend_file.resolve()).exists():
        raise FileNotFoundError(f"Blender file {blend_file} not found.")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_file.resolve() if output_file else None

    if config.autoscale:
        if not torch.cuda.is_available():
            _log.warning("No GPU devices found, cannot autoscale. Falling back on using a single render job.")
            config.autoscale = False
            config.max_job_vram = None
            config.jobs = 1
        elif torch.cuda.device_count() != 1:
            _log.warning("Cannot autoscale when using multi-gpu. Falling back on using a single render job.")
            config.autoscale = False
            config.max_job_vram = None
            config.jobs = 1
        else:
            idx = torch.cuda.current_device()
            device = torch.device(idx)
            free, _ = torch.cuda.mem_get_info(device)
            config.jobs = free // config.max_job_vram
            _log.info(f"Auto-scaling to using {config.jobs} render jobs on {torch.cuda.get_device_name(idx)}.")

    if config.jobs <= 0:
        raise RuntimeError(f"At least one render job is needed, got `render_config.jobs={config.jobs}`.")

    with (
        BlenderClients.spawn(
            jobs=config.jobs,
            log=config.log_dir,
            timeout=config.timeout,
            executable=config.executable,
            autoexec=config.autoexec,
        ) as clients,
        ElapsedProgress() as progress,
    ):
        task = progress.add_task(f"Rendering {blend_file.stem}...")
        render_job(
            clients,
            blend_file,
            output_dir,
            frame_start=frame_start,
            frame_end=frame_end,
            config=config,
            output_blend_file=output_file,
            dry_run=dry_run,
            update_fn=partial(progress.update, task),
        )
