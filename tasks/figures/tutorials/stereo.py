"""Stereo figures: ``docs/source/tutorials/stereo.rst``.

Blender writes frames sharded into numbered subfolders and numbers them from one
(``frames/0000/001.png``), which is why the ffmpeg input takes ``-start_number 1``
and the ``%03d`` pattern from inside that shard.
"""

from __future__ import annotations

import subprocess

from .._page import CACHE, Node, dataset_check, page_task, static

# Interpupillary distance quoted on the page, halved for one eye and expressed in
# metres, which is what the render config expects.
HALF_IPD = 0.0325

# ``kitchenpack.blend`` animates frames 1..600 at 50 fps, so the render is 12 s of real
# time. Every frame is kept and played at the source rate, so the video runs at the
# speed the scene was animated at.
SOURCE_FPS = 50
FRAME_COUNT = 600

# An h264 video is an order of magnitude smaller than the equivalent gif, and is what
# the page embeds through the ``video`` directive the other sensor pages use.
VIDEO_CRF = 23


def _render_eye(eye: str, sign: float):
    """Build the recipe rendering one eye of the stereo pair.

    Args:
        eye: ``"left"`` or ``"right"``, used for the output path and log lines.
        sign: Sign of the offset along the camera's local X axis, in metres.

    Returns:
        A recipe taking the blender executable and whether to force.
    """

    def recipe(executable: str | None, force: bool = False) -> None:
        cmd = (
            f"visionsim blender.render-animation kitchenpack.blend stereo/{eye}/"
            f" --config.camera-offset {sign * HALF_IPD} 0 0"
            + (f" --config.executable={executable}" if executable else "")
            + (" --config.no-allow-skips" if force else "")
        )
        subprocess.run(cmd, shell=True, check=True, cwd=CACHE)

    return recipe


def _anaglyph(executable: str | None, force: bool = False) -> None:
    """Combine the two eye renders into a red/cyan anaglyph video.

    The eyes are stacked side by side and handed to ``stereo3d`` as ``sbsl``
    (side-by-side, left first), which emits a half-width ``arcd`` anaglyph. That
    output is what keeps the result at the per-eye resolution instead of the doubled
    width of the stack.

    Args:
        executable: Ignored; this step needs no Blender.
        force: Ignored; ffmpeg overwrites unconditionally.
    """
    frames = "stereo/{eye}/frames/0000/%03d.png"
    left, right = (frames.format(eye=e) for e in ("left", "right"))
    cmd = (
        f"ffmpeg -hide_banner -loglevel error -y"
        f" -framerate {SOURCE_FPS} -start_number 1 -i {left}"
        f" -framerate {SOURCE_FPS} -start_number 1 -i {right}"
        f" -filter_complex"
        f" '[0:v][1:v]hstack=inputs=2[stacked];[stacked]stereo3d=in=sbsl:out=arcd[out]'"
        f" -map '[out]' -c:v libx264 -crf {VIDEO_CRF} -pix_fmt yuv420p"
        f" -movflags +faststart {static('stereo-anaglyph.mp4')}"
    )
    print(f"provisioning anaglyph: {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=CACHE)


NODES = (
    Node(
        name="stereo-left",
        files=(CACHE / "stereo" / "left" / "frames",),
        check=dataset_check(CACHE / "stereo" / "left" / "frames", FRAME_COUNT),
        recipe=_render_eye("left", -1.0),
    ),
    Node(
        name="stereo-right",
        files=(CACHE / "stereo" / "right" / "frames",),
        check=dataset_check(CACHE / "stereo" / "right" / "frames", FRAME_COUNT),
        recipe=_render_eye("right", 1.0),
    ),
    Node(
        name="stereo-anaglyph",
        files=(static("stereo-anaglyph.mp4"),),
        is_figure=True,
        requires=("stereo-left", "stereo-right"),
        recipe=_anaglyph,
    ),
)

build = page_task(NODES, "stereo", "Regenerate the stereo anaglyph figure.")
