from __future__ import annotations

import json
import os
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from visionsim.simulate.config import RenderConfig

if TYPE_CHECKING:
    from collections.abc import Generator

    from visionsim.simulate.blender import BlenderClients
    from visionsim.utils.progress import ElapsedProgress


def _validate_inputs(blend_file: Path, output_dir: Path, output_file: Path | None) -> tuple[Path, Path, Path | None]:
    """Validate user-supplied paths and resolve them to absolute locations.

    Args:
        blend_file: Path to blend file.
        output_dir: Dataset output folder, created if it doesn't exist.
        output_file: If set, path to write the modified blend file to.

    Returns:
        tuple[Path, Path, Path | None]: Resolved blend file, output directory and output file.

    Raises:
        FileNotFoundError: raised if the blend file does not exist.
    """
    if not (blend_file := blend_file.resolve()).exists():
        raise FileNotFoundError(f"Blender file {blend_file} not found.")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    return blend_file, output_dir, output_file.resolve() if output_file else None


def _resolve_autoscale(config: RenderConfig) -> RenderConfig:
    """Resolve the number of render jobs when autoscaling is requested.

    Falls back to a single job when autoscaling isn't possible (no GPU, or multiple GPUs), otherwise
    divides the free VRAM by ``config.max_job_vram``. A no-op when ``config.autoscale`` is false.

    Args:
        config: Render configuration.

    Returns:
        RenderConfig: A copy of ``config`` with the resolved ``jobs``. The passed-in config is
            left untouched.
    """
    from visionsim.cli import _log  # avoid circular import

    if not config.autoscale:
        return config

    config = deepcopy(config)

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

    return config


def _require_blender(executable: str | os.PathLike | None = None) -> None:
    """Check that a blender installation is available.

    Args:
        executable: Path to the blender executable. Defaults to looking for one on ``$PATH``.

    Raises:
        RuntimeError: raised if no blender installation is found.
    """
    from visionsim.cli import _run  # avoid circular import

    if _run(f"{executable or 'blender'} --version", shell=True, hide=True).returncode != 0:
        raise RuntimeError("No blender installation found on path!")


@contextmanager
def _spawn_clients(*, config: RenderConfig) -> Generator[tuple[BlenderClients, ElapsedProgress]]:
    """Spawn connected blender clients along with a progress bar.

    The number of clients is driven by ``config.jobs``.

    Args:
        config: Render configuration.

    Yields:
        tuple[BlenderClients, ElapsedProgress]: Connected clients and a progress instance.
    """
    from visionsim.simulate.blender import BlenderClients
    from visionsim.utils.progress import ElapsedProgress

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
        yield clients, progress


def render_animation(
    blend_file: Path,
    output_dir: Path,
    /,
    config: RenderConfig,
    frame_start: int | None = None,
    frame_end: int | None = None,
    frame_step: int | None = None,
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
        frame_step: Step to render frames by. Defaults to internal value.
        output_file: If set, write the modified blend file to
            this path. Helpful for troubleshooting. Defaults to not saving.
        dry_run: if true, nothing will be rendered at all. Defaults to False.
    """
    from visionsim.simulate.job import render_job

    if config.jobs <= 0:
        raise RuntimeError(f"At least one render job is needed, got `config.jobs={config.jobs}`.")

    blend_file, output_dir, output_file = _validate_inputs(blend_file, output_dir, output_file)
    config = _resolve_autoscale(config)
    _require_blender(config.executable)

    with _spawn_clients(config=config) as (clients, progress):
        task = progress.add_task(f"Rendering {blend_file.stem}...")
        render_job(
            clients,
            blend_file,
            output_dir,
            frame_start=frame_start,
            frame_end=frame_end,
            frame_step=frame_step,
            config=config,
            output_blend_file=output_file,
            dry_run=dry_run,
            update_fn=partial(progress.update, task),
        )


def render_frame(
    blend_file: Path,
    output_dir: Path,
    /,
    config: RenderConfig,
    frame: int,
    output_file: Path | None = None,
    dry_run: bool = False,
) -> None:
    """Render a single frame from a single blend-file into a dataset folder.

    Unlike ``blender.render-animation``, which renders a whole frame range, this renders exactly one
    frame index, writing the same output layout and metadata as the animation path would for that
    frame. Animations stay enabled, so the rendered frame matches what ``render-animation`` would
    produce at the same index.

    This always runs in a single render job, ignoring ``config.jobs`` and ``config.autoscale``.

    Args:
        blend_file: Path to blend file.
        output_dir: Dataset output folder.
        config: Render configuration. ``jobs`` and ``autoscale`` are ignored.
        frame: Index of the frame to render.
        output_file: If set, write the modified blend file to
            this path. Helpful for troubleshooting. Defaults to not saving.
        dry_run: if true, nothing will be rendered at all. Defaults to False.

    Raises:
        RuntimeError: raised if no blender installation is found on path.
        FileNotFoundError: raised if the blend file does not exist.
    """
    from visionsim.cli import _log
    from visionsim.simulate.blender import BlenderClient
    from visionsim.simulate.job import frame_job

    blend_file, output_dir, output_file = _validate_inputs(blend_file, output_dir, output_file)

    # A single frame is rendered by one client, so any job fan-out is meaningless here.
    if config.autoscale or config.jobs != 1:
        _log.warning(
            f"Rendering a single frame always uses a single render job, ignoring "
            f"`config.autoscale={config.autoscale}` and `config.jobs={config.jobs}`."
        )
    _require_blender(config.executable)

    with BlenderClient.spawn(
        timeout=config.timeout,
        log=config.log_dir,
        executable=config.executable,
        autoexec=config.autoexec,
    ) as client:
        frame_job(
            client,
            blend_file,
            output_dir,
            config,
            frame_number=frame,
            output_blend_file=output_file,
            dry_run=dry_run,
        )


def render_playblast(
    blend_file: Path,
    output_dir: Path,
    /,
    config: RenderConfig,
    frame_start: int | None = None,
    frame_end: int | None = None,
    frame_step: int | None = None,
    video: bool = True,
    output_file: Path | None = None,
    dry_run: bool = False,
) -> None:
    """Create a fast preview (playblast) of an animation from a single blend-file.

    This uses Blender's viewport/OpenGL renderer instead of a full render, which is much quicker
    but produces no ground truth annotations. Output is written to a dedicated ``playblast/``
    folder inside ``output_dir``, either as a single video file or a PNG frame sequence.

    This always runs in a single render job, overriding ``config.jobs`` and ``config.autoscale``.
    Playblasts are fast, need a GL context (ie: a non-background Blender), and write a single
    shared output file that is not safe to render in parallel.

    Tip: There is no need for a physical display. If one isn't already available, on linux you can
    start this headlessly with a virtual X server, eg:
    ``DISPLAY="" WAYLAND_DISPLAY="" xvfb-run -a --server-args="-screen 0 1920x1080x24" visionsim blender.render-playblast ...``

    Args:
        blend_file: Path to blend file.
        output_dir: Dataset output folder.
        config: Render configuration. ``jobs`` and ``autoscale`` are ignored.
        frame_start: Start rendering at this frame index (inclusive).
        frame_end: Stop rendering at this frame index (inclusive).
        frame_step: Step to render frames by. Defaults to internal value.
        video: If true, encode the preview as a single video file, otherwise save a PNG
            frame sequence along with a metadata database. Defaults to True.
        output_file: If set, write the modified blend file to
            this path. Helpful for troubleshooting. Defaults to not saving.
        dry_run: if true, nothing will be rendered at all. Defaults to False.

    Raises:
        RuntimeError: raised if no blender installation is found on path.
        FileNotFoundError: raised if the blend file does not exist.
    """
    from visionsim.cli import _log
    from visionsim.simulate.blender import BlenderClient
    from visionsim.simulate.job import playblast_job
    from visionsim.utils.progress import ElapsedProgress

    blend_file, output_dir, output_file = _validate_inputs(blend_file, output_dir, output_file)

    # Work on a copy: the job overrides below must not leak into the caller's config.
    config = deepcopy(config)

    # Playblasts are fast, there is no point in spreading the work across multiple render jobs.
    # They also need a GL context, which Blender only has outside of background mode, and their single
    # output file is not parallel-safe, so the job count is forced to one regardless of the config.
    if config.autoscale or config.jobs != 1:
        _log.warning(
            f"Playblast rendering always uses a single render job, ignoring "
            f"`config.autoscale={config.autoscale}` and `config.jobs={config.jobs}`."
        )
    config.autoscale = False
    config.max_job_vram = None
    config.jobs = 1

    # The viewport renderer only writes the color preview, so every other `include_*` is dropped.
    # Detect them by exclusion rather than by name, so a new pass is caught without touching this.
    ignored = [
        name
        for name in vars(config)
        if name.startswith("include_") and name != "include_frames" and getattr(config, name)
    ]
    if ignored:
        _log.warning(f"Playblast rendering produces no ground truth annotations, ignoring {', '.join(sorted(ignored))}.")
    _require_blender(config.executable)

    with (
        BlenderClient.spawn(
            timeout=config.timeout,
            log=config.log_dir,
            executable=config.executable,
            autoexec=config.autoexec,
            background=False,
        ) as client,
        ElapsedProgress() as progress,
    ):
        task = progress.add_task(f"Playblasting {blend_file.stem}...")
        playblast_job(
            client,
            blend_file,
            output_dir,
            frame_start=frame_start,
            frame_end=frame_end,
            frame_step=frame_step,
            video=video,
            config=config,
            output_blend_file=output_file,
            dry_run=dry_run,
            update_fn=partial(progress.update, task),
        )


def optimize_rate(
    blend_file: Path,
    /,
    config: RenderConfig,
    frame_start: int | None = None,
    frame_end: int | None = None,
    frame_step: int | None = None,
    resolution_percentage: int = 10,
    percentile: float = 95.0,
    target: float = 1.0,
    tolerance: float = 0.1,
    init_k: float = 5.0,
    max_iterations: int = 15,
    max_scale_factor: float = 15.0,
    scale_decay: float = 0.95,
    stall_tolerance: float = 0.05,
    max_depth: float | None = None,
    debug_path: Path | None = None,
) -> float:
    """Find the keyframe multiplier ``k`` such that the ``percentile``-th percentile optical flow magnitude
    is about ``target`` pixels.

    This method uses a simplified Newton method to find the keyframe multiplier, assuming flow is roughly
    proportional to `1/k`. At every step, `render-animation` at a coarse resolution and low sample count is run,
    and the flow is estimated by scaling by the coarse resolution flow by `1/resolution_percentage`.

    Note:
        Some render config parameters are set automatically for the prob such as low samples, no denoising,
        no preview, and only flows enabled.

    Args:
        blend_file: Path to blend file.
        config: Render configuration.
        frame_start: Start rendering at this frame index (inclusive).
        frame_end: Stop rendering at this frame index (inclusive).
        frame_step: Step to render frames by.
        resolution_percentage: Render resolution as a percentage of the
            scene's configured resolution.
        percentile: Percentile of optical flow magnitudes to use.
        target: Target optical flow magnitude to achieve.
        tolerance: Tolerance for the optical flow magnitude.
        init_k: Initial guess for the keyframe multiplier.
        max_iterations: Maximum number of iterations to run.
        max_scale_factor: Maximum factor by which to scale the keyframe
            multiplier in a single iteration.
        scale_decay: Decay the scale factor multiplier by this much after each iteration,
            helps prevent oscillations.
        stall_tolerance: If ``p_flow``, the percentile-th percentile flow, changes by less than this
            fraction of ``target`` between consecutive iterations, the search is considered stalled and
            terminates early with a warning.
        max_depth: If set, pixels with depth greater than this limit are ignored when computing the
            percentile flow. Depth rendering will be enabled.
        debug_path: Optional path to write debug information to. If provided, flow digests for each iteration
            are written to `debug_path / f"iter_{i:02d}_k_{k:.4f}.json"`.

    Returns:
        float: Estimated keyframe multiplier.
    """
    import tempfile

    import numpy as np
    from fastdigest import TDigest

    from visionsim.cli import _log
    from visionsim.dataset import Dataset

    if scale_decay <= 0 or scale_decay > 1:
        raise ValueError(f"Parameter `scale_decay` ({scale_decay}) must be in (0, 1]")
    if max_scale_factor <= 1:
        raise ValueError(f"Parameter `max_scale_factor` ({max_scale_factor}) must be >= 1")
    if init_k <= 0:
        raise ValueError(f"Parameter `init_k` ({init_k}) must be positive.")

    # Build a lightweight probe config: coarse resolution, flows only, single job, no previews, low samples.
    probe_config = deepcopy(config)
    probe_config.resolution_percentage = resolution_percentage
    probe_config.max_samples = 1
    probe_config.adaptive_threshold = False
    probe_config.include_flows = True

    probe_config.use_denoising = False
    probe_config.include_frames = False
    probe_config.include_composites = False
    probe_config.include_diffuse_pass = False
    probe_config.include_specular_pass = False
    probe_config.include_depths = max_depth is not None
    probe_config.include_normals = False
    probe_config.include_segmentations = False
    probe_config.include_materials = False
    probe_config.include_points = False
    probe_config.include_segmentations = False
    probe_config.previews = False
    probe_config.autoscale = False
    probe_config.jobs = 1

    # Scale factor from low-res pixel coords to full-res pixel coords.
    scale_to_full = 100.0 / resolution_percentage
    prev_p_flow: float | None = None
    k = init_k

    # Render animation into a tempdir, start at k=init_k and scale it by max_flow/threshold where max_flow is the
    # percentile-th percentile optical flow in previous step, scaled to full resolution.
    # K does not need to be doubled every time, we know how much to scale it by.
    for i in range(max_iterations):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            _log.debug(f"[#{i + 1}] Using temp directory: {tmpdir_str}")
            tmpdir = Path(tmpdir_str)
            iter_digest = TDigest()

            probe_config.keyframe_multiplier = k
            _log.info(f"[#{i + 1}] Probing with keyframe_multiplier={k:.4f} ...")

            try:
                render_animation(
                    blend_file,
                    tmpdir,
                    probe_config,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    frame_step=frame_step,
                )
            except (ConnectionError, TimeoutError, OSError, EOFError, ValueError, RuntimeError) as e:
                _log.error(f"Failed to render animation for keyframe_multiplier={k:.4f}: {e}", exc_info=True)
                return k
            except KeyboardInterrupt:
                _log.info("Optimization interrupted by user.")
                return k

            # Gather all rendered flow EXRs (shape H x W x 4: fx, fy, bx, by).
            flow_dataset = Dataset.from_path(tmpdir / "flows")
            depth_dataset = Dataset.from_path(tmpdir / "depths") if max_depth is not None else [None] * len(flow_dataset)

            for j, ((flow, _), depth_item) in enumerate(zip(flow_dataset, depth_dataset)):
                try:
                    fx, fy, bx, by = flow.transpose(2, 0, 1)
                    fw_mag = np.sqrt(fx**2 + fy**2).ravel() * scale_to_full
                    bw_mag = np.sqrt(bx**2 + by**2).ravel() * scale_to_full

                    # Skip flow values from frames beyond the `max_depth`
                    if max_depth is not None:
                        depth = depth_item[0]
                        valid = (depth <= max_depth).ravel()
                        fw_mag, bw_mag = fw_mag[valid], bw_mag[valid]

                    # Skip the flow to/from a non-existent frame
                    if j != 0:
                        iter_digest.batch_update(fw_mag)
                    if j != len(flow_dataset) - 1:
                        iter_digest.batch_update(bw_mag)
                except ValueError:
                    _log.warning(f"Skipping corrupted flow/depth for index {j}")
                    continue

            p_flow = iter_digest.quantile(percentile / 100.0)
            _log.info(f"Estimated {percentile:.0f}th-percentile flow at full resolution: {p_flow:.4f} px")
            _log.debug(
                "Flow stats (min, max, mean, median, 95%, 99%): "
                f"{iter_digest.min():.4f}, {iter_digest.max():.4f}, {iter_digest.mean():.4f}, "
                f"{iter_digest.median():.4f}, {iter_digest.quantile(0.95):.4f}, {iter_digest.quantile(0.99):.4f}"
            )

            if debug_path is not None:
                debug_dir = Path(debug_path)
                debug_dir.mkdir(parents=True, exist_ok=True)

                with open(digest_path := debug_dir / f"iter_{i:02d}_k_{k:.4f}.json", "w") as f:
                    json.dump(iter_digest.to_dict(), f, indent=2)
                    _log.debug(f"Wrote digest to {digest_path}")

            if abs(p_flow - target) <= tolerance:
                _log.info(f"Found suitable keyframe_multiplier={k:.4f}")
                return k

            if prev_p_flow is not None and abs(p_flow - prev_p_flow) < stall_tolerance * target:
                _log.warning(
                    f"Optimization stalled: {percentile:.0f}th-percentile flow changed by "
                    f"less than {stall_tolerance * target:.4f} px since last iteration. "
                    f"Returning current keyframe_multiplier={k:.4f}."
                )
                return k

            prev_p_flow = p_flow
            factor = p_flow / target
            if factor > max_scale_factor:
                _log.warning(
                    f"Flow is too high, limiting scale factor to {max_scale_factor} to prevent overshooting (scale factor: {factor:.4f})"
                )
                factor = max_scale_factor
            elif factor < 1.0 / max_scale_factor:
                _log.warning(
                    f"Flow is too low, limiting scale factor to {1.0 / max_scale_factor:.4f} to prevent undershooting (scale factor: {factor:.4f})"
                )
                factor = 1.0 / max_scale_factor
            k += (k * (factor - 1)) * scale_decay**i

    _log.warning(
        f"Failed to find suitable keyframe_multiplier within {max_iterations} iterations. Returning last value."
    )
    return k
