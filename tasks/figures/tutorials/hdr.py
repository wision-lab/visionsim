"""HDR single-frame figures: ``docs/source/tutorials/hdr.rst``.

``attic.blend`` animates its camera for frames 1..600 at 50 fps. Instead of
previewing the whole flythrough, this page renders one frame that holds both the
dim and the bright end of the scene, and shows that frame at several exposures.
"""

from __future__ import annotations

from pathlib import Path

from .._page import Node, _command_recipe, cached, dataset_check, encode_webp, page_task, static

# Frame 410 looks up at the roof, where the skylights are far brighter than the
# shadowed corner under the eaves, so one frame spans the range the page is about.
FRAME = 410
RESOLUTION_PERCENTAGE = 200
MAX_SAMPLES = 1024
# Exposure of the two insets, in stops, relative to the base frame.
BRIGHT_EV = -3.0
DIM_EV = 4.0
# Inset regions as fractions of the frame, ``(x0, x1, y0, y1)``: the skylights
# top-right, the shadowed corner under the left eaves.
BRIGHT_REGION = (0.70, 1.00, 0.00, 0.26)
DIM_REGION = (0.00, 0.30, 0.46, 0.80)


def _frame() -> Path:
    """Return the single EXR the figure is built from."""
    return next(iter(sorted(cached("hdr", "attic-frame", "frames").rglob("*.exr"))))


def _expose(linear, ev: float):
    """Scale a linear image by ``ev`` stops and encode it for display."""
    import numpy as np

    from visionsim.utils.color import linearrgb_to_srgb

    return np.clip(linearrgb_to_srgb(linear * 2.0**ev), 0.0, 1.0)


def _hdr_figure(executable: str | None, force: bool = False) -> None:
    """Compose the page figure from the single rendered EXR.

    The base image is the frame at zero stops, with two insets cut from the same
    linear frame: the skylights pulled down and the shadowed corner pushed up,
    each labelled with its exposure. Re-exposing one float frame is what lets a
    single render stand in for several display-referred images.

    Args:
        executable: Ignored; this step needs no Blender.
        force: Ignored; matplotlib overwrites unconditionally.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import OpenEXR  # type: ignore
    from matplotlib.patches import Rectangle

    with OpenEXR.File(str(_frame())) as handle:
        linear = next(iter(handle.channels().values())).pixels.astype(np.float64)
    height, width = linear.shape[:2]

    out = static("hdr/attic-hdr.webp")
    out.parent.mkdir(parents=True, exist_ok=True)
    # matplotlib writes the composite as a PNG, which the figure is then encoded from.
    drawn = cached("hdr", "attic-hdr.png")

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.imshow(_expose(linear, 0.0))
    ax.set_axis_off()

    insets = (
        (BRIGHT_REGION, BRIGHT_EV, (0.035, 0.05, 0.27, 0.27), "#ffd166", "skylight  -3 EV"),
        (DIM_REGION, DIM_EV, (0.695, 0.05, 0.27, 0.27), "#4cc9f0", "shadowed corner  +4 EV"),
    )
    for (x0, x1, y0, y1), ev, box, color, label in insets:
        left, right = int(x0 * width), int(x1 * width)
        top, bottom = int(y0 * height), int(y1 * height)
        ax.add_patch(Rectangle((left, top), right - left, bottom - top, fill=False, edgecolor=color, linewidth=1.6))
        inset = fig.add_axes(box)
        inset.imshow(_expose(linear[top:bottom, left:right], ev))
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2.2)
        inset.text(
            0.04,
            0.04,
            label,
            transform=inset.transAxes,
            color=color,
            fontsize=10,
            va="bottom",
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 3, "edgecolor": "none"},
        )

    fig.savefig(drawn, facecolor="black", dpi=150)
    plt.close(fig)
    encode_webp(drawn, out)


NODES = (
    Node(
        name="attic-frame",
        files=(cached("hdr", "attic-frame", "frames"),),
        check=dataset_check(cached("hdr", "attic-frame", "frames"), 1),
        recipe=_command_recipe(
            "attic-frame",
            "visionsim blender.render-frame attic.blend hdr/attic-frame/"
            f" --frame {FRAME}"
            f" --config.resolution-percentage={RESOLUTION_PERCENTAGE}"
            f" --config.max-samples={MAX_SAMPLES}"
            " --config.frames.file-format OPEN_EXR --config.frames.bit-depth 32"
            " --config.frames.exr-codec ZIP{executable}{force}",
        ),
    ),
    Node(
        name="attic-hdr",
        files=(static("hdr/attic-hdr.webp"),),
        is_figure=True,
        requires=("attic-frame",),
        recipe=_hdr_figure,
    ),
)

build = page_task(NODES, "hdr", "Regenerate the attic HDR single-frame figure.")
