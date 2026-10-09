"""iToF coding-scheme waveforms: ``docs/source/sections/sensors/itof.rst``."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: this task only writes files
import matplotlib.pyplot as plt
import numpy as np

# Registers the "science"/"nature"/"ieee" matplotlib styles; the import has no stubs.
import scienceplots  # type: ignore[import-untyped]  # noqa: F401

from visionsim.emulate.itof.coding import CodingScheme, make_coding_functions
from visionsim.emulate.itof.simulation import compute_correlation_function

from ..._page import STATIC, Node, page_task

N_BINS = 128
# Order 2 pushes the 2-D and 3-D curves past 900 bins at 5 captures, where they
# smear into a solid block, so every dimension stays at order 1 (64-180 bins).
HILBERT_ORDER = 1
N_TAPS_PLOTTED = 3

# Multi-frequency sinusoids need one frequency and phase shift per capture. The
# decoder expects one low-frequency tap plus three uniformly shifted taps of a
# single higher frequency, and requires that three-shift group to be the widest,
# so the low-frequency tap stays at 0.5 units.
MULT_FREQ_HZ = np.array([0.5, 1.0, 1.0, 1.0])
MULT_FREQ_SHIFTS = np.array([0.0, 0.0, 2 * np.pi / 3, 4 * np.pi / 3])

# (scheme, n_captures, extra kwargs). Capture counts differ: the 3-D Hilbert
# curve needs five and the multi-frequency scheme one frequency per tap.
SCHEMES: list[tuple[CodingScheme, int, dict]] = [
    ("convSin", 3, {}),
    ("deltaSin", 3, {}),
    ("convSquare", 3, {}),
    ("singleRamp", 3, {}),
    ("doubleRamp", 3, {}),
    ("deltaHilbertDimOne", 3, {"hilbert_order": HILBERT_ORDER}),
    ("deltaHilbertDimTwo", 4, {"hilbert_order": HILBERT_ORDER}),
    ("deltaHilbertDimThree", 5, {"hilbert_order": HILBERT_ORDER}),
    ("multFreqSin", len(MULT_FREQ_HZ), {"freq_vec": MULT_FREQ_HZ, "shifts_vec": MULT_FREQ_SHIFTS}),
]

# The dark variants are lightened to stay distinguishable against the background.
TAP_COLORS_LIGHT = ["#e5007d", "#5aae61", "#3b6fb6"]
TAP_COLORS_DARK = ["#ff6fb5", "#7fd18a", "#6fa8ff"]

_TAP_COLORS = TAP_COLORS_LIGHT


def _apply_theme(dark: bool) -> None:
    """Select the matplotlib style and tap palette for the requested theme."""
    global _TAP_COLORS
    if dark:
        plt.style.use(["science", "no-latex", "dark_background"])
        plt.rcParams.update(
            {
                "axes.edgecolor": "#c9c9c9",
                "axes.labelcolor": "#e6e6e6",
                "xtick.color": "#c9c9c9",
                "ytick.color": "#c9c9c9",
                "text.color": "#e6e6e6",
            }
        )
        _TAP_COLORS = TAP_COLORS_DARK
    else:
        plt.style.use(["science", "no-latex"])
        _TAP_COLORS = TAP_COLORS_LIGHT

    # Transparent figures let the page background show through.
    plt.rcParams.update({"figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none"})


def _codes_and_correlation(scheme: CodingScheme, n_captures: int, kwargs: dict):
    """Compute modulation codes, reference codes, correlations and their x axes.

    The two x axes both run 0-1 over one unambiguous range so the rows line up,
    but a ramp code array holds two periods, so the axes differ in bin count.

    Args:
        scheme: Coding scheme identifier.
        n_captures: Number of captures (phase shifts) for the scheme.
        kwargs: Extra scheme-specific arguments forwarded to
            :func:`~visionsim.emulate.itof.coding.make_coding_functions`, such as
            ``hilbert_order`` or ``freq_vec``.

    Returns:
        The x axis for the codes, the x axis for the correlations, the modulation
        codes, the reference codes, and the stacked correlations.
    """
    modulation_codes, reference_codes = make_coding_functions(scheme, n_captures, N_BINS, **kwargs)
    n_points = modulation_codes.shape[1]

    # Ramps carry two periods in one array, so half the array spans the range.
    bins_per_period = n_points // 2 if scheme in ("singleRamp", "doubleRamp") else n_points
    time_resolution = 1.0 / n_points

    # Each tap is correlated against its own modulation waveform: identical across
    # taps for most schemes, but multi-frequency emits a different frequency per
    # capture, so using tap 0's waveform throughout would repeat one response.
    correlations = np.stack(
        [
            compute_correlation_function(
                modulation_codes[k], reference_codes[k], time_resolution, n_depths=bins_per_period
            )
            for k in range(min(n_captures, N_TAPS_PLOTTED))
        ]
    )

    # The Hilbert code is a few bins longer than the others, and the ramps return
    # half an array, so both land off the common grid. Resample onto N_BINS so
    # every row shares a sample count.
    if correlations.shape[1] != N_BINS:
        x_src = np.linspace(0.0, 1.0, correlations.shape[1])
        x_dst = np.linspace(0.0, 1.0, N_BINS, endpoint=False)
        correlations = np.stack([np.interp(x_dst, x_src, row) for row in correlations])

    x_codes = np.linspace(0.0, 1.0, n_points, endpoint=False)
    x_corr = np.linspace(0.0, 1.0, correlations.shape[1], endpoint=False)
    return x_codes, x_corr, modulation_codes, reference_codes, correlations


def make_grid(schemes: list[tuple[CodingScheme, int, dict]]):
    """Plot every scheme as a row of modulation, demodulation and correlation.

    Args:
        schemes: One ``(scheme, n_captures, kwargs)`` triple per row.

    Returns:
        The matplotlib figure holding the grid.
    """
    n_rows = len(schemes)
    fig, axes = plt.subplots(n_rows, 3, figsize=(11, 2.3 * n_rows), squeeze=False)

    column_titles = ["Modulation", "Demodulation", "Correlation"]
    rows = [_codes_and_correlation(scheme, n_captures, kwargs) for scheme, n_captures, kwargs in schemes]

    # Schemes use incompatible code amplitudes: impulses are 0/1 while the others
    # peak near 1/n_bins. Dividing the modulation row by its own peak puts every
    # envelope at 1.0 so the rows compare. Dividing each trace by its own peak
    # instead would flatten the DC and dark taps to a line.
    panels_per_row = []
    for x_codes, x_corr, modulation, reference, correlation in rows:
        modulation_peak = float(np.abs(modulation[:N_TAPS_PLOTTED]).max()) or 1.0
        panels_per_row.append([(modulation / modulation_peak, x_codes), (reference, x_codes), (correlation, x_corr)])

    for row, (scheme, _, _) in enumerate(schemes):
        for ax, (data, x), title in zip(axes[row], panels_per_row[row], column_titles):
            for k in range(min(len(data), N_TAPS_PLOTTED)):
                ax.plot(x, data[k], color=_TAP_COLORS[k % len(_TAP_COLORS)], linewidth=1.5)
            ax.set_yticks([0, 1])
            ax.set_ylim(-0.2, 1.4)
            ax.set_xmargin(0)
            ax.tick_params(labelbottom=False, labelsize="large")
            if row == 0:
                ax.set_title(title, fontsize="x-large")
            if ax is axes[row][0]:
                ax.set_ylabel(scheme, fontsize="large")

    for ax in axes[-1]:
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("period" if ax is not axes[-1][-1] else "unambiguous range", fontsize="large")

    fig.tight_layout()

    # The tap colors repeat in every panel, so one legend serves the whole grid.
    handles = [plt.Line2D([], [], color=_TAP_COLORS[k], linewidth=1.5, label=f"tap {k}") for k in range(N_TAPS_PLOTTED)]
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=N_TAPS_PLOTTED,
        fontsize="large",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.01),
    )
    return fig


def plot_codes() -> None:
    """Write the light and dark iToF coding-scheme figures into ``_static/sensors``."""
    out = STATIC / "sensors"
    out.mkdir(parents=True, exist_ok=True)

    # The docs pick between the two with the only-light / only-dark classes.
    for dark, suffix in ((False, ""), (True, "-dark")):
        _apply_theme(dark)
        fig = make_grid(SCHEMES)
        fig.savefig(out / f"itof-codes-all{suffix}.svg", dpi=200, bbox_inches="tight")
        plt.close(fig)


NODES = (
    Node(
        name="itof-codes-all",
        files=(STATIC / "sensors" / "itof-codes-all.svg", STATIC / "sensors" / "itof-codes-all-dark.svg"),
        # The recipe writes both files above; the coding schemes and correlation
        # model it imports are not tracked, so pass --force to rebuild after
        # changing them.
        is_figure=True,
        recipe=lambda executable, force=False: plot_codes(),  # type: ignore[misc]
    ),
)

build = page_task(NODES, "itof", "Regenerate the iToF coding-scheme figures.")
