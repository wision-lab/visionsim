"""Event-camera figures: ``docs/source/tutorials/dvs.rst``.

Every figure is one ``emulate.events`` run over the 32-fold interpolated quickstart
sequence, reduced to a single preview frame. Interpolated input keeps the stream sparse:
each input frame covers a slice of motion far shorter than the source render's, so a
single frame's events stay readable instead of filling the image.

Only one ground-truth frame's worth of interpolated frames is fed in (``PATTERN``), and
``--only-preview`` skips the event file, because a still is all each figure needs. Two
families: a contrast-threshold sweep with all noise terms off, and a per-pixel threshold
mismatch sweep at a fixed threshold.
"""

from __future__ import annotations

from .._page import Node, _command_recipe, cached, encode_webp, page_task, static

# Interpolated frames to emulate: every sub-frame of ground-truth frame 150, which is
# 32 frames at the 1600 fps the sequence was interpolated to. Emulating the whole
# sequence would cost ~200x the work for the one still each figure keeps.
INPUT_DIR = "quickstart/ultramodern-interp"
PATTERN = "0000/150_*.png"
FPS = 1600

# ON/OFF contrast threshold in log-intensity units, raised left to right so the stream
# goes from dense to sparse. The noise terms are off, leaving the threshold as the only
# thing changing between the four.
THRESHOLDS = (0.1, 0.2, 0.4, 0.8)

# Standard deviation of the per-pixel threshold mismatch, raised left to right at a fixed
# 0.2 threshold, so the four run from a clean stream to one salted with random events.
SIGMA_THRESHOLDS = (0, 0.1, 0.2, 0.4)

# Noise-off and threshold-fixed flag sets the two sweeps start from, so each figure
# differs from its neighbours in exactly the parameter under test.
CLEAN = "--sigma-thres=0 --cutoff-hz=0 --leak-rate-hz=0 --shot-noise-rate-hz=0"
CLEAN_AT_THRESHOLD = "--pos-thres=0.2 --neg-thres=0.2 --cutoff-hz=0 --leak-rate-hz=0 --shot-noise-rate-hz=0"


def _emulated(name: str, slug: str, flags: str) -> Node:
    """Emulate one event-camera configuration from the interpolated quickstart sequence.

    ``emulate.events`` takes a single configuration per invocation, so every figure
    gets its own run; ``--force`` lets a re-run replace an output directory a previous
    run already created.
    """
    return Node(
        name=name,
        files=(cached("dvs", slug, "preview"),),
        requires=("ultramodern-interp",),
        recipe=_command_recipe(
            name,
            f"visionsim emulate.events --input-dir={INPUT_DIR} --pattern={PATTERN}"
            f" --output-dir=dvs/{slug}/ --fps={FPS} --preview-step=1"
            f" {flags} --seed=0 --only-preview --force",
        ),
    )


def _still(name: str, slug: str, figure: str) -> Node:
    """Encode the middle preview frame of an emulation into ``_static`` as the figure.

    All runs consume the same frames at the same preview step, so they write the same
    preview indices; taking the middle one rather than a guessed index keeps the four
    comparable and the figure reproducible.
    """

    def still(executable: str | None, force: bool = False) -> None:
        previews = sorted(cached("dvs", slug, "preview").rglob("*.png"))
        encode_webp(previews[len(previews) // 2], static(figure))

    return Node(name=name, files=(static(figure),), is_figure=True, requires=(f"dvs-{slug}-emulated",), recipe=still)


def _pair(prefix: str, index: int, flags: str) -> tuple[Node, Node]:
    """Build the emulation and the still that encodes one of its preview frames."""
    slug = f"{prefix}-{index}"
    return (
        _emulated(f"dvs-{slug}-emulated", slug, flags),
        _still(f"dvs-{slug}", slug, f"sensors/dvs/ultramodern-{slug}.webp"),
    )


_THRESHOLD = tuple(
    node
    for i, t in enumerate(THRESHOLDS, 1)
    for node in _pair("threshold", i, f"{CLEAN} --pos-thres={t} --neg-thres={t}")
)
_MISMATCH = tuple(
    node
    for i, s in enumerate(SIGMA_THRESHOLDS, 1)
    for node in _pair("mismatch", i, f"{CLEAN_AT_THRESHOLD} --sigma-thres={s}")
)

NODES = _THRESHOLD + _MISMATCH

build = page_task(NODES, "dvs", "Regenerate the event-camera emulation figures.")
