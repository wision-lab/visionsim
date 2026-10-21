"""Playblast figures: ``docs/source/tutorials/playblast.rst``."""

from __future__ import annotations

from .._page import Node, page_task, static, video_recipe

NODES = (
    Node(
        name="playblast-preview",
        files=(static("playblast/playblast-preview.mp4"),),
        is_figure=True,
        requires=("playblast",),
        recipe=video_recipe("playblast/playblast/*.png", 5, "playblast/playblast-preview"),
    ),
    Node(
        name="playblast-full-preview",
        files=(static("playblast/playblast-full-preview.mp4"),),
        is_figure=True,
        requires=("playblast-full",),
        recipe=video_recipe("playblast/full/frames/*/*.png", 5, "playblast/playblast-full-preview"),
    ),
)

build = page_task(NODES, "playblast", "Regenerate the playblast comparison figures.")
