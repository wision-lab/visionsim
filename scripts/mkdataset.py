from __future__ import annotations

import itertools
import logging
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path

import imageio.v3 as iio
import multiprocess
import numpy as np
import numpy.typing as npt
import tyro
from natsort import natsorted
from rich.logging import RichHandler
from rich.progress import track
from rich.traceback import install

from visionsim.cli import ffmpeg
from visionsim.cli.blender import sequence_info
from visionsim.simulate.blender import BlenderClients
from visionsim.simulate.config import RenderConfig
from visionsim.simulate.job import render_job
from visionsim.utils.progress import PoolProgress

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logging.getLogger("PIL").setLevel(logging.WARNING)
log = logging.getLogger("rich")
install(suppress=[tyro])
app = tyro.extras.SubcommandApp()


def find_blends(root: str | os.PathLike) -> list[Path]:
    """Find all blend files in a root directory, allowing for blends to be in their own folder.

    Args:
        root (str | os.PathLike): Root directory to search for blend files in.

    Returns:
        list[Path]: List of resolved blender file paths.
    """
    blends = itertools.chain(Path(root).glob("*.blend"), Path(root).glob("*/*.blend"))
    return [p.resolve() for p in natsorted(blends)]


def sample_scenes(
    render_config: RenderConfig,
    sequences_per_scene: int = 1,
    num_frames: int | None = None,
    anim_start: int = 1,
    anim_end: int = 600,
) -> tuple[int, npt.NDArray]:
    """Sample an animation range and return the starting indices of subsequences
    of length `num_frames`. Validate that subsequences do not overlap."""
    frame_starts, step = np.linspace(
        anim_start * render_config.keyframe_multiplier,
        anim_end * render_config.keyframe_multiplier,
        sequences_per_scene,
        endpoint=False,
        retstep=True,
    )
    frame_starts, step = np.floor(frame_starts).astype(int), np.floor(step).astype(int)

    if num_frames is None:
        if sequences_per_scene == 1:
            num_frames = step
        else:
            log.critical("Parameter `num-frames` must be specified if using multiple `sequences-per-scene`.")
            sys.exit(2)
    elif num_frames > step:
        log.critical(
            "Parameter `num-frames` is too large and would cause sequences to overlap, got %s, expected a max of %s. "
            "Consider increasing `keyframe-multiplier`, or lowering the number of `sequences-per-scene`.",
            num_frames,
            step,
        )
        sys.exit(2)
    return frame_starts, num_frames


@app.command
def create_datasets(
    scenes_dir: str | os.PathLike,
    datasets_dir: str | os.PathLike,
    render_config: RenderConfig,
    sequences_per_scene: int = 1,
    num_frames: int | None = None,
    allow_skips: bool = False,
    dry_run: bool = False,
):
    """Create datasets by rendering out sequences from many blend-files.

    Args:
        scenes_dir (str | os.PathLike): Directory to search for blend files in (includes sub-directories 1-level deep).
            Every scene is assumed to be animated between frames 1-600.
        datasets_dir (str | os.PathLike): Dataset output folder, ground truth renders will be saved in
            `<datasets_dir>/renders/<scene_name>/<sequence_id>` where `scene_name` is the stem of the blender file (filename
            without extension), and `sequence_id` is defined as `<keyframe_multiplier>-<frame_start>-<frame_start+num_frames>`.
        render_config (RenderConfig): Render configuration.
        sequences_per_scene (int, optional): Number of sequences per scene to render. The start of each sequence is sampled
            uniformly from the animation range [1, 600].
        num_frames (int | None, optional): Number of frames to render per sequence. If None, render everything.
        allow_skips (bool, optional): If true, allow skipping over whole sequences if their corresponding root directory exists.
        dry_run (bool, optional): if true, nothing will be rendered at all. Defaults to False.
    """
    # Sample sequences and validate args.
    frame_starts, num_frames = sample_scenes(render_config, sequences_per_scene, num_frames)

    # Find all sequences, i.e: pairs of blend-files and frame ranges (start, start+num-frames)
    scenes = find_blends(scenes_dir)
    sequences = list(itertools.product(scenes, frame_starts))
    log.info(f"Generating {len(sequences)} sequences from {len(scenes)} scenes...")

    # Note: Shuffling the sequences helps with throughput as different scenes are rendered at once
    random.seed(123456789)
    random.shuffle(sequences)

    # Define helper to map each sequence to a unique path
    def get_sequence_dir(scene_name, frame_start):
        Path(datasets_dir).mkdir(parents=True, exist_ok=True)
        sequence_id = f"{int(render_config.keyframe_multiplier):03}-{frame_start:05}-{frame_start + num_frames:05}"
        sequence_dir = Path(datasets_dir) / "renders" / scene_name / sequence_id
        return sequence_dir.resolve()

    # Queue up and run all render tasks
    with (
        BlenderClients.pool(
            jobs=render_config.jobs,
            log_dir=Path(render_config.log_dir),
            timeout=render_config.timeout,
            executable=render_config.executable,
            autoexec=render_config.autoexec,
        ) as pool,
        PoolProgress() as progress,
    ):
        for blend_file, frame_start in sequences:
            sequence_dir = get_sequence_dir(blend_file.stem, frame_start)

            if not allow_skips or not (sequence_dir / "transforms.json").exists():
                # Note: The client will be automagically passed to `render` here.
                tick = progress.add_task(f"{blend_file.stem} ({frame_start}-{frame_start + num_frames})")
                pool.apply_async(
                    render_job,
                    args=(blend_file, sequence_dir),
                    kwds=dict(
                        frame_start=frame_start,
                        frame_end=frame_start + num_frames,
                        config=render_config,
                        dry_run=dry_run,
                        update_fn=tick,
                    ),
                )
            else:
                log.info(f"Skipping: {sequence_dir}")
        progress.wait()
        pool.close()
        pool.join()

    # Gather some metadata about every sequence and save it to a "info.json" file.
    with multiprocess.Pool(render_config.jobs) as pool:
        info_fn = partial(sequence_info, keyframe_multiplier=render_config.keyframe_multiplier)
        sequence_dirs = [get_sequence_dir(blend_file.stem, frame_start) for blend_file, frame_start in sequences]
        list(pool.imap(info_fn, track(sequence_dirs, description="Gathering Metadata...")))


@app.command
def preview_datasets(
    datasets_dir: str | os.PathLike,
    previews_dir: str | os.PathLike,
    fps: int = 50,
    allow_skips: bool = False,
    grids: bool = True,
    colorize: bool = True,
    clean: bool = False,
    jobs: int = 1,
):
    """Generate preview videos of the datasets

    Args:
        datasets_dir (str | os.PathLike): Datasets folder, will search for all `transforms.json` recursively.
        previews_dir (str | os.PathLike): Directory in which to save previews, files will be saved
            as eg: "previews-dir/scene-name/sequence-id/frames.mp4"
        fps (int, optional): Framerate of preview videos. Defaults to 50fps.
        allow_skips (bool, optional): If true, and outputs exist, skip over them. Defaults to False.
        grids (bool, optional): If true, assemble all previews into a grid-video saved in
            previews-dir/grids. Defaults to True.
        colorize (bool, optional): If true, colorize depths and flows using transforms.colorize-depths/flows. Defaults to False.
        clean (bool, optional): If true, clean all colorized depth and flow frames. Ignored if `colorize` is False. Defaults to False.
        jobs (int, optional): Allow multiple previews to be built in parallel. Defaults to 1.
    """
    # TODO: This should just import the relevant functions from visionsim instead of calling the CLI.
    previews_dir = Path(previews_dir)
    previews_dir.mkdir(exist_ok=True, parents=True)
    (previews_dir / "grids").mkdir(exist_ok=True, parents=True)
    sequence_dirs = [p.parent for p in Path(datasets_dir).glob("**/transforms.json")]
    run = partial(subprocess.run, check=True, capture_output=True)
    frame_types = set()
    colorized_dirs = []
    colorize_commands = []
    animate_commands = []

    for sequence_dir in sequence_dirs:
        for frame_type in sequence_dir.glob("*"):
            if frame_type.is_dir():
                preview_path = previews_dir / sequence_dir.parent.stem / sequence_dir.stem / f"{frame_type.stem}.mp4"
                frame_types.add(frame_type.stem)

                if colorize and frame_type.stem in ("depths", "flows"):
                    input_frames = frame_type / "colorized"
                    colorized_dirs.append(input_frames)

                    if not any(input_frames.glob("*.png")) or not allow_skips:
                        cmd = f"visionsim transforms.colorize-{frame_type.stem} {frame_type} {input_frames}"
                        colorize_commands.append(shlex.split(cmd))
                elif any(frame_type.glob("*.png")):
                    input_frames = frame_type

                if not preview_path.exists() or not allow_skips:
                    cmd = f'visionsim ffmpeg.animate {input_frames} -o {preview_path} --fps={fps} --pattern="*.png" --force'
                    animate_commands.append(shlex.split(cmd))

    if colorize_commands:
        with multiprocess.Pool(jobs) as pool:
            list(
                track(
                    pool.imap(partial(subprocess.run, stdout=subprocess.DEVNULL), colorize_commands),
                    description="Colorizing...",
                    total=len(colorize_commands),
                )
            )

    if animate_commands:
        with multiprocess.Pool(jobs) as pool:
            list(track(pool.imap(run, animate_commands), description="Making Previews...", total=len(animate_commands)))

    if clean:
        for d in colorized_dirs:
            shutil.rmtree(d, ignore_errors=True)

    if grids:
        for frame_type in track(frame_types, description="Making Preview Grids..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                # ffmpeg grid already searches for videos in a folder, so here
                # we just symlink all input videos into a tempdir
                files = natsorted(Path(previews_dir).glob(f"**/{frame_type}.mp4"))
                outfile = previews_dir / "grids" / f"grid-{frame_type}.mp4"
                tmpdirname = Path(tmpdir)

                for i, p in enumerate(files):
                    (tmpdirname / f"{i:09}{p.suffix}").symlink_to(p, target_is_directory=False)

                candidates = [
                    (w, int(len(files) / w)) for w in range(1, len(files) + 1) if int(len(files) / w) == (len(files) / w)
                ]
                w, h = min(reversed(candidates), key=lambda c: sum(c))
                ffmpeg.grid(tmpdirname, width=w, height=h, outfile=outfile, force=True)


@app.command
def purge_corrupted(datasets: str | os.PathLike, /, jobs: int | None = None, dry_run: bool = False):
    """Scan dataset and remove frames are unreadable or otherwise corrupted.

    Args:
        datasets (str | os.PathLike): Dataset directory
        jobs (int | None, optional): number of jobs to use for scanning, defaults to number of cores.
        dry_run (bool, optional): if set, do not remove corrupted files.
    """

    def validate_single(frame):
        try:
            iio.imread(frame)
        except Exception:
            log.warning("Corrupted: %s", frame)

            if not dry_run:
                frame.unlink()
            return frame

    with multiprocess.Pool(jobs) as pool:
        frames = list(Path(datasets).glob("**/*.png"))
        results = pool.imap(validate_single, track(frames, description="Purging..."))
        corrupted = [f for f in results if f]

    log.info(f"Found {'and removed' if not dry_run else ''} {len(corrupted)} corrupted files out of {len(frames)}.")


if __name__ == "__main__":
    app.cli()
