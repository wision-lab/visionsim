"""Quick-start preview figures: ``docs/source/quick-start.rst``."""

from __future__ import annotations

from ._page import Node, gif_recipe, page_task, static

# Only the figures this page owns; the datasets they need are intermediates in
# ``_page.INTERMEDIATES``, named through ``requires``.
NODES = (
    Node(
        name="lego-gt-preview",
        files=(static("lego-gt-preview.gif"),),
        is_figure=True,
        requires=("lego-gt",),
        recipe=gif_recipe("quickstart/lego-gt/frames/*/*.png", 5, "lego-gt-preview"),
    ),
    Node(
        name="lego-depth-preview",
        files=(static("lego-depth-preview.gif"),),
        is_figure=True,
        requires=("lego-gt",),
        recipe=gif_recipe("quickstart/lego-gt/previews/depths/*/*.png", 5, "lego-depth-preview"),
    ),
    Node(
        name="lego-rgb25fps-preview",
        files=(static("lego-rgb25fps-preview.gif"),),
        is_figure=True,
        requires=("lego-rgb25fps",),
        recipe=gif_recipe("quickstart/lego-rgb25fps/*/*.png", 1, "lego-rgb25fps-preview"),
    ),
    Node(
        name="lego-spc4kHz-preview",
        files=(static("lego-spc4kHz-preview.gif"),),
        is_figure=True,
        # emulate.spad writes only .npy, so previews go .npy -> mp4 -> PNGs.
        requires=("lego-spc4kHz",),
        recipe=gif_recipe("quickstart/lego-spc4kHz/preview/*.png", 160, "lego-spc4kHz-preview"),
    ),
    Node(
        name="lego-dvs125fps-preview",
        files=(static("lego-dvs125fps-preview.gif"),),
        is_figure=True,
        # Also embedded by docs/source/sections/sensors/dvs.rst, but the frames
        # come from the quick-start dataset, so the figure belongs to this page.
        requires=("lego-dvs125fps",),
        recipe=gif_recipe("quickstart/lego-dvs125fps/preview/*/*.png", 5, "lego-dvs125fps-preview"),
    ),
    Node(
        name="lego-itof-preview",
        files=(static("lego-itof-preview.gif"),),
        is_figure=True,
        # iToF writes raw .npy taps; the preview PNGs come from `--preview`.
        requires=("lego-itof",),
        recipe=gif_recipe("quickstart/lego-itof/preview/tap_0/*.png", 5, "lego-itof-preview"),
    ),
)

build = page_task(NODES, "quick-start", "Regenerate the quick-start preview figures.")
