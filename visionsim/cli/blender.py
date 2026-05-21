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
            frame_step=frame_step,
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
    resolution_percentage: int = 25,
    percentile: float = 95.0,
    target: float = 1.0,
    tolerance: float = 0.1,
    max_iterations: int = 15,
    max_scale_factor: float = 10.0,
) -> float:
    """Find the keyframe multiplier `k` such that the `percentile`-th percentile optical flow magnitude
    is about `target` pixels.

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
        max_iterations: Maximum number of iterations to run.
        max_scale_factor: Maximum factor by which to scale the keyframe
            multiplier in a single iteration.

    Returns:
        float: Estimated keyframe multiplier.
    """
    import copy
    import tempfile

    import numpy as np
    from fastdigest import TDigest

    from visionsim.cli import _log
    from visionsim.dataset import Dataset

    # Build a lightweight probe config: coarse resolution, flows only, single job, no previews, low samples.
    probe_config = copy.deepcopy(config)
    probe_config.resolution_percentage = resolution_percentage
    probe_config.max_samples = 1
    probe_config.adaptive_threshold = False
    probe_config.include_flows = True

    probe_config.use_denoising = False
    probe_config.include_frames = False
    probe_config.include_composites = False
    probe_config.include_diffuse_pass = False
    probe_config.include_specular_pass = False
    probe_config.include_depths = False
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
    k = 1.0

    # Render animation into a tempdir, start at k=1.0 and scale it by max_flow/threshold where max_flow is the
    # percentile-th percentile optical flow in previous step, scaled to full resolution.
    # K does not need to be doubled every time, we know how much to scale it by.
    for i in range(max_iterations):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            digest = TDigest()

            probe_config.keyframe_multiplier = k
            _log.info(f"[#{i + 1}] Probing with keyframe_multiplier={k:.4f} ...")

            render_animation(
                blend_file,
                tmpdir,
                probe_config,
                frame_start=frame_start,
                frame_end=frame_end,
                frame_step=frame_step,
            )

            # Gather all rendered flow EXRs (shape H x W x 4: fx, fy, bx, by).
            flow_files = sorted((tmpdir / "flows").glob("**/*.exr"))
            if not flow_files:
                raise RuntimeError("No flow files were rendered. Check the render configuration.")

            for flow_file in flow_files:
                try:
                    flow = np.array(Dataset.load_data(flow_file))  # (H, W, 4)
                    fx, fy, bx, by = flow.transpose(2, 0, 1)
                    digest.batch_update(np.sqrt(fx**2 + fy**2).ravel())
                    digest.batch_update(np.sqrt(bx**2 + by**2).ravel())
                except ValueError:
                    _log.warning(f"Failed to load flow file: {flow_file}")
                    continue

            # Scale pixel magnitudes up to what they would be at full resolution.
            p_flow = digest.quantile(percentile / 100.0) * scale_to_full
            _log.info(f"Estimated {percentile:.0f}th-percentile flow at full resolution: {p_flow:.4f} px")

            if abs(p_flow - target) <= tolerance:
                _log.info(f"Found suitable keyframe_multiplier={k:.4f}")
                return k

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
            k *= factor

    _log.warning(
        f"Failed to find suitable keyframe_multiplier within {max_iterations} iterations. Returning last value."
    )
    return k
