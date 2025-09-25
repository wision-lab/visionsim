from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path

import torch

from visionsim.dataset import Dataset
from visionsim.simulate.blender import BlenderClients
from visionsim.simulate.config import RenderConfig
from visionsim.simulate.job import render_job
from visionsim.utils.progress import ElapsedProgress


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
        if not torch.cuda.is_available():
            _log.warning("No GPU devices found, cannot autoscale. Falling back on using a single render job.")
            render_config.autoscale = False
            render_config.max_job_vram = None
            render_config.jobs = 1
        elif torch.cuda.device_count() != 1:
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
        render_job(
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
