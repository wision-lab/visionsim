"""Quick-start preview figures: ``docs/source/quick-start.rst``."""

from __future__ import annotations

from ._page import Node, page_task, static, video_recipe

# Only the figures this page owns; the datasets they need are intermediates in
# ``_page.INTERMEDIATES``, named through ``requires``.  The ground-truth figures
# keep every other frame so that their 25fps playback runs at the speed of the
# 50fps render; the emulated datasets are already at their own rates.
NODES = (
    Node(
        name="ultramodern-gt-preview",
        files=(static("quick-start/ultramodern-gt-preview.mp4"),),
        is_figure=True,
        requires=("ultramodern-gt",),
        recipe=video_recipe("quickstart/ultramodern-gt/frames/*/*.png", 2, "quick-start/ultramodern-gt-preview"),
    ),
    Node(
        name="ultramodern-depth-preview",
        files=(static("quick-start/ultramodern-depth-preview.mp4"),),
        is_figure=True,
        requires=("ultramodern-gt",),
        recipe=video_recipe(
            "quickstart/ultramodern-gt/previews/depths/*/*.png", 2, "quick-start/ultramodern-depth-preview"
        ),
    ),
    Node(
        name="ultramodern-rgb25fps-preview",
        files=(static("quick-start/ultramodern-rgb25fps-preview.mp4"),),
        is_figure=True,
        requires=("ultramodern-rgb25fps",),
        recipe=video_recipe("quickstart/ultramodern-rgb25fps/*/*.png", 1, "quick-start/ultramodern-rgb25fps-preview"),
    ),
    Node(
        name="ultramodern-spc1600fps-preview",
        files=(static("quick-start/ultramodern-spc1600fps-preview.mp4"),),
        is_figure=True,
        # emulate.spad writes only .npy, so previews go .npy -> mp4 -> PNGs, and
        # those PNGs keep the interpolated rate: step by 64 to land on 25fps.
        requires=("ultramodern-spc1600fps",),
        # Single-photon previews are per-pixel noise, which costs several megabytes a
        # clip even at the default quality.
        recipe=video_recipe(
            "quickstart/ultramodern-spc1600fps/preview/*.png", 64, "quick-start/ultramodern-spc1600fps-preview", crf=28
        ),
    ),
    Node(
        name="ultramodern-dvs1600fps-preview",
        files=(static("quick-start/ultramodern-dvs1600fps-preview.mp4"),),
        is_figure=True,
        # The event-camera section embeds this figure; the frames come from the
        # quick-start dataset, so the figure belongs to this page.
        requires=("ultramodern-dvs1600fps",),
        # Event previews are per-pixel noise, so the clip is expensive at any quality.
        # Keep every 64th preview to play the trajectory at 25fps like the ground-truth
        # clip, and encode at the frames' own size since the pages stretch it to width.
        recipe=video_recipe(
            "quickstart/ultramodern-dvs1600fps/preview/*/*.png",
            64,
            "quick-start/ultramodern-dvs1600fps-preview",
            size=512,
            crf=28,
        ),
    ),
    Node(
        name="ultramodern-itof-preview",
        files=(static("quick-start/ultramodern-itof-preview.mp4"),),
        is_figure=True,
        # iToF writes raw .npy taps; the preview PNGs come from `--preview`.
        requires=("ultramodern-itof",),
        recipe=video_recipe(
            "quickstart/ultramodern-itof/preview/tap_0/*.png", 2, "quick-start/ultramodern-itof-preview"
        ),
    ),
)

build = page_task(NODES, "quick-start", "Regenerate the quick-start preview figures.")
