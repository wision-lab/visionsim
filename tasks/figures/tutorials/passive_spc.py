"""Single-photon-camera figures: ``docs/source/tutorials/passive-spc.rst``.

Both rows recover a scene through the estimator the sensor section derives: average the
binary frames into a detection probability, then invert it with
:func:`spc_avg_to_rgb <visionsim.emulate.spc.spc_avg_to_rgb>`. The factor row samples
in-process, since ``--factor`` only sets the Bernoulli probability the emulator draws with.
"""

from __future__ import annotations

import numpy as np

from .._page import Node, cached, page_task, static

# Consecutive exposures from the quickstart capture; the largest trades a little sharpness
# for a cleaner average as the camera moves.
FRAME_COUNTS = (1, 8, 32)

# Scene flux per exposure, from photon-starved to saturated, to show the full dynamic range.
FACTORS = (0.02, 1.0, 25.0)

# Exposures averaged per factor panel; fixed so only the flux varies across the row.
FACTOR_FRAMES = 64

# Ground-truth frame the quickstart capture starts on, resampled as the factor row's flux.
GT_FRAME = cached("quickstart", "ultramodern-gt", "frames", "0000", "150.png")

# Dataset undoes the emulator's bitpacking when reading emulated captures back.
SPC_DATASET = ("quickstart", "ultramodern-spc1600fps")


def _reconstruction(count: int, index: int) -> Node:
    """Average ``count`` consecutive emulated binary frames into one still."""

    def recipe(executable: str | None, force: bool = False) -> None:
        import imageio.v3 as iio

        from visionsim.dataset import Dataset
        from visionsim.emulate.spc import spc_avg_to_rgb
        from visionsim.utils.color import linearrgb_to_srgb

        dataset = Dataset.from_path(cached(*SPC_DATASET))

        # Accumulate in float; stacking the frames first would hold them all at once.
        total: np.ndarray | None = None
        for i in range(count):
            frame, _ = dataset[i]
            assert isinstance(frame, np.ndarray)
            frame = frame.astype(np.float32)
            total = frame if total is None else total + frame
        assert total is not None

        # The mean estimates the detection probability; the quantile stretches it to [0, 1].
        intensity = spc_avg_to_rgb(total / count, factor=1.0, quantile=0.99)
        srgb = linearrgb_to_srgb(np.clip(intensity, 0.0, 1.0))

        out = static(f"sensors/spc/ultramodern-reconstruction-{index}.webp")
        out.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(out, (srgb * 255).astype(np.uint8))

    figure = f"sensors/spc/ultramodern-reconstruction-{index}.webp"
    return Node(
        name=f"spc-reconstruction-{index}",
        files=(static(figure),),
        is_figure=True,
        requires=("ultramodern-spc1600fps",),
        recipe=recipe,
    )


def _factor(factor: float, index: int) -> Node:
    """Resample one ground-truth frame at a fixed scene flux per exposure."""

    def recipe(executable: str | None, force: bool = False) -> None:
        import imageio.v3 as iio

        from visionsim.emulate.spc import emulate_spc, spc_avg_to_rgb
        from visionsim.utils.color import linearrgb_to_srgb, srgb_to_linearrgb

        # The emulator samples linear radiance, so undo the display mapping first.
        frame = iio.imread(GT_FRAME).astype(np.float32) / 255.0
        flux = srgb_to_linearrgb(frame)

        rng = np.random.default_rng(0)
        total: np.ndarray | None = None
        for _ in range(FACTOR_FRAMES):
            sample = emulate_spc(flux, factor=factor, rng=rng).astype(np.float32)
            total = sample if total is None else total + sample
        assert total is not None

        # No quantile: the panels share one display exposure, so lost signal reads as loss.
        intensity = spc_avg_to_rgb(total / FACTOR_FRAMES, factor=factor, quantile=None)
        srgb = linearrgb_to_srgb(np.clip(intensity, 0.0, 1.0))

        out = static(f"sensors/spc/ultramodern-factor-{index}.webp")
        out.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(out, (srgb * 255).astype(np.uint8))

    figure = f"sensors/spc/ultramodern-factor-{index}.webp"
    return Node(
        name=f"spc-factor-{index}",
        files=(static(figure),),
        is_figure=True,
        requires=("ultramodern-gt",),
        recipe=recipe,
    )


NODES = (
    *(_reconstruction(count, i) for i, count in enumerate(FRAME_COUNTS, 1)),
    *(_factor(factor, i) for i, factor in enumerate(FACTORS, 1)),
)

build = page_task(NODES, "passive-spc", "Regenerate the single-photon-camera emulation figures.")
