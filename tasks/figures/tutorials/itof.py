"""iToF frequency-sweep figures: ``docs/source/tutorials/itof.rst``.

One figure per theme: tap 0 over decoded depth, one column per modulation
frequency, on a shared depth colorbar. It reads the ultramodern quickstart
render and adds emulation only, never re-rendering the scene.

The figure is transparent and drawn twice, with light and dark text, because a
single raster with a background sits as a light box on dark pages.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
from matplotlib.colors import Normalize

from .._page import Node, _command_recipe, cached, encode_webp, page_task, static

# Low to high. The room is 3.34 m deep, so the lowest frequency's 3.75 m range
# covers it whole and each step up folds more of it.
FREQUENCIES = (40e6, 60e6, 80e6, 120e6)

# A still needs one frame, not the whole sequence.
FRAME = 250
# The frame sits in a ``0000`` shard, named by number; Blender before 5.0
# zero-pads it to four digits (``0250.png``) where 5.0+ pad to three
# (``250.png``), so the glob leaves the padding to match either.
PATTERN = f"0000/*{FRAME}.png"
DEPTH_PATTERN = f"0000/*{FRAME}.exr"

INPUT_DIR = "quickstart/ultramodern-gt/frames"
DEPTH_DIR = "quickstart/ultramodern-gt/depths"

# Shared across the sweep, since normalizing per frequency would rescale each
# column to its own range and hide the folding the sweep exists to show.
SHARED_VMAX = scipy.constants.c / (2 * min(FREQUENCIES))

# Foreground colour furo uses per theme.
TEXT_LIGHT = "#1a1a1a"
TEXT_DARK = "#e6e6e6"


def _slug(freq: float) -> str:
    """Path-safe frequency label, e.g. ``40e6`` -> ``40mhz``."""
    return f"{freq / 1e6:.0f}mhz"


def _emulated(name: str, freq: float) -> Node:
    """Emulate one frequency from a single ground-truth frame.

    The CLI takes one frequency per invocation, so each column gets its own run,
    and ``--force`` replaces the output directory a previous run left behind.
    """
    slug = _slug(freq)
    return Node(
        name=name,
        files=(cached("itof", slug, "preview", "tap_0", "0000.png"),),
        requires=("ultramodern-gt",),
        recipe=_command_recipe(
            name,
            f"visionsim emulate.itof --input-dir={INPUT_DIR} --depth-dir={DEPTH_DIR}"
            f" --pattern={PATTERN} --depth-pattern={DEPTH_PATTERN}"
            f" --output-dir=itof/{slug}/ --scheme=convSin --n-captures=4"
            f" --freq={freq} --preview --force",
        ),
    )


def _draw(taps: list[np.ndarray], depths: list[np.ndarray], ranges: list[float], text_color: str):
    """Draw tap 0 over decoded depth, one column per frequency.

    Args:
        taps: Colorized tap-0 image per frequency.
        depths: Decoded depth, in metres, per frequency.
        ranges: Unambiguous range per frequency, for the column titles.
        text_color: Colour for every label and stroke, for the theme.

    Returns:
        The figure, on a transparent background.
    """
    columns = len(FREQUENCIES)
    fig = plt.figure(figsize=(13.5, 6.6), layout="constrained")
    # The colorbar gets its own grid column so it can span the depth row without
    # narrowing it and knocking the two rows out of alignment.
    grid = fig.add_gridspec(2, columns + 1, width_ratios=[*[1] * columns, 0.04])

    norm = Normalize(vmin=0.0, vmax=SHARED_VMAX)
    for col, (tap, depth, freq, span) in enumerate(zip(taps, depths, FREQUENCIES, ranges)):
        top = fig.add_subplot(grid[0, col])
        bottom = fig.add_subplot(grid[1, col])
        top.imshow(tap)
        image = bottom.imshow(depth, cmap="turbo", norm=norm)
        top.set_title(f"{freq / 1e6:.0f}MHz ({span:.2f}m)", color=text_color)
        for ax in (top, bottom):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("none")
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Axes are appended column by column, so the first two are row 0 and row 1 of
    # the first column: the leftmost panel of each row.
    fig.axes[0].set_ylabel("Tap 0", color=text_color, fontsize="large")
    fig.axes[1].set_ylabel("Decoded depth", color=text_color, fontsize="large")

    bar = fig.colorbar(image, cax=fig.add_subplot(grid[1, columns]))
    bar.set_label("depth [m]", color=text_color)
    bar.ax.tick_params(colors=text_color)
    bar.outline.set_edgecolor(text_color)
    return fig


def _figure(name: str) -> Node:
    """Combine the emulated runs into one shared figure, in both text colours."""
    requires = tuple(f"itof-{_slug(freq)}-emulated" for freq in FREQUENCIES)

    def recipe(executable: str | None, force: bool = False) -> None:
        from visionsim.emulate.itof import decode

        taps: list[np.ndarray] = []
        depths: list[np.ndarray] = []
        ranges: list[float] = []
        for freq in FREQUENCIES:
            run = cached("itof", _slug(freq))
            measurements = np.load(run / "0000.npy")
            params = json.loads((run / "params.json").read_text())
            _, height, width = measurements.shape

            # decode wants (n_captures, n_pixels); the frequency comes back off
            # the run so the decoder matches what the acquisition used.
            decoded = decode("convSin", measurements.reshape(measurements.shape[0], -1), freq=params["freq_hz"])
            depths.append(decoded.reshape(height, width))
            taps.append(plt.imread(run / "preview" / "tap_0" / "0000.png"))
            ranges.append(params["unambiguous_range_m"])

        # The docs pick between the two with the only-light / only-dark classes.
        for text_color, suffix in ((TEXT_LIGHT, ""), (TEXT_DARK, "-dark")):
            fig = _draw(taps, depths, ranges, text_color)
            png = cached("itof", f"sweep{suffix}.png")
            fig.savefig(png, dpi=150, transparent=True)
            plt.close(fig)
            encode_webp(png, static(f"sensors/itof/sweep{suffix}.webp"))

    return Node(
        name=name,
        files=(static("sensors/itof/sweep.webp"), static("sensors/itof/sweep-dark.webp")),
        is_figure=True,
        requires=requires,
        recipe=recipe,
    )


NODES = (
    *(_emulated(f"itof-{_slug(freq)}-emulated", freq) for freq in FREQUENCIES),
    _figure("itof-sweep"),
)

build = page_task(NODES, "itof", "Regenerate the iToF frequency-sweep figures.")
