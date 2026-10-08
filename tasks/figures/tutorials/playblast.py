"""Playblast figures: ``docs/source/tutorials/playblast.rst``."""

from __future__ import annotations

from .._page import Node, gif_recipe, page_task, static

NODES = (
    Node(
        name="playblast-preview",
        files=(static("playblast-preview.gif"),),
        is_figure=True,
        requires=("playblast",),
        recipe=gif_recipe("playblast/playblast/*.png", 5, "playblast-preview"),
    ),
    Node(
        name="playblast-full-preview",
        files=(static("playblast-full-preview.gif"),),
        is_figure=True,
        requires=("playblast-full",),
        recipe=gif_recipe("playblast/full/frames/*/*.png", 5, "playblast-full-preview"),
    ),
)

build = page_task(NODES, "playblast", "Regenerate the playblast comparison figures.")
