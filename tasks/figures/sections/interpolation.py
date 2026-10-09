"""Interpolation figures: ``docs/source/sections/interpolation.rst``."""

from __future__ import annotations

from .._page import Node, gif_recipe, page_task, static

# Each figure's dataset renders and interpolates in one intermediate node, so the
# render happens once even when several figures share it.
NODES = (
    Node(
        name="lego0025-interp",
        files=(static("interpolation/lego0025-interp.gif"),),
        is_figure=True,
        requires=("lego-0025",),
        recipe=gif_recipe("interpolation/lego0025-interp/*/*.png", 8, "interpolation/lego0025-interp"),
    ),
    Node(
        name="lego0050-interp",
        files=(static("interpolation/lego0050-interp.gif"),),
        is_figure=True,
        requires=("lego-0050",),
        recipe=gif_recipe("interpolation/lego0050-interp/*/*.png", 8, "interpolation/lego0050-interp"),
    ),
    Node(
        name="lego0100-interp",
        files=(static("interpolation/lego0100-interp.gif"),),
        is_figure=True,
        requires=("lego-0100",),
        recipe=gif_recipe("interpolation/lego0100-interp/*/*.png", 8, "interpolation/lego0100-interp"),
    ),
    Node(
        name="lego0200-interp",
        files=(static("interpolation/lego0200-interp.gif"),),
        is_figure=True,
        requires=("lego-0200",),
        recipe=gif_recipe("interpolation/lego0200-interp/*/*.png", 8, "interpolation/lego0200-interp"),
    ),
)

build = page_task(NODES, "interpolation", "Regenerate the interpolation figures.")
