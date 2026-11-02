"""Conventional-camera figures: ``docs/source/sections/sensors/rgb.rst``."""

from __future__ import annotations

from ..._page import Node, _command_recipe, cached, encode_webp, page_task, static

# ``--chunk-size`` frames are averaged into each output frame, so it is the exposure
# time: more frames per output means more motion blur. Read noise is off for these
# four so blur is the only thing changing.
EXPOSURE_CHUNKS = (10, 40, 160, 640)
# Read noise at a fixed 80-frame exposure, the standard deviation lowered from the
# first figure to the last so the four run from noisiest to cleanest.
READNOISE_STDS = (80, 40, 20, 10)
READNOISE_CHUNK = 80


def _emulated(name: str, subdir: str, chunk_size: int, readout_std: int) -> Node:
    """Emulate one RGB configuration from the interpolated ultramodern sequence.

    ``emulate.rgb`` takes a single configuration per invocation, so each of the
    eight figures gets its own run; ``--force`` lets a re-run replace an output
    directory that a previous run already created.
    """
    return Node(
        name=name,
        files=(cached("rgb", subdir),),
        requires=("ultramodern-interp",),
        recipe=_command_recipe(
            name,
            f"visionsim emulate.rgb --input-dir=quickstart/ultramodern-interp/"
            f" --output-dir=rgb/{subdir}/ --chunk-size={chunk_size}"
            f" --readout-std={readout_std} --force",
        ),
    )


def _still(name: str, emulated: str, subdir: str, figure: str) -> Node:
    """Encode the first output frame of an emulation into ``_static`` as the figure.

    Every run starts at the same point on the orbit, so the first frame is the
    comparable one; taking a fixed index rather than guessing keeps the figure
    reproducible.
    """

    def still(executable: str | None, force: bool = False) -> None:
        encode_webp(cached("rgb", subdir, "0000", "000.png"), static(figure))

    return Node(name=name, files=(static(figure),), is_figure=True, requires=(emulated,), recipe=still)


def _pair(prefix: str, index: int, chunk_size: int, readout_std: int) -> tuple[Node, Node]:
    """Build the emulation and the still that encodes its first frame."""
    slug = f"{prefix}-{index}"
    return (
        _emulated(f"rgb-{slug}-emulated", slug, chunk_size, readout_std),
        _still(f"rgb-{slug}", f"rgb-{slug}-emulated", slug, f"sensors/rgb/ultramodern-{slug}.webp"),
    )


_EXPOSURE = tuple(node for i, chunk in enumerate(EXPOSURE_CHUNKS, 1) for node in _pair("exposure", i, chunk, 0))
_READNOISE = tuple(
    node for i, std in enumerate(READNOISE_STDS, 1) for node in _pair("readnoise", i, READNOISE_CHUNK, std)
)

NODES = _EXPOSURE + _READNOISE

build = page_task(NODES, "rgb", "Regenerate the conventional-camera emulation figures.")
