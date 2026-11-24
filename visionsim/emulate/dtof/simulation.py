"""Scene-level emulation of direct time-of-flight (dToF) measurements."""

from __future__ import annotations

import logging
import math
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.constants
import torch
from torch import Tensor

from visionsim.emulate.dtof.histogrammers import simulate_ewh

DtoFModel = Literal["mc", "forward"]

logger = logging.getLogger(__name__)


def make_irf(pulse_width_bins: float = 1.0) -> Tensor:
    """Build a normalized Gaussian laser pulse (instrument response function) kernel.

    The Gaussian is sampled at bin centers, so it is accurate for pulses of at least about
    one bin; narrower pulses collapse towards a single tap.

    Args:
        pulse_width_bins (float, optional): Pulse full width at half maximum, in time bins.
            Defaults to 1.0.

    Returns:
        Normalized kernel of odd length, symmetric about its center bin.

    Raises:
        ValueError: If ``pulse_width_bins`` is not positive.
    """
    if pulse_width_bins <= 0:
        raise ValueError(f"pulse_width_bins must be positive, got {pulse_width_bins!r}")

    # the FWHM-to-sigma conversion of a Gaussian.
    sigma = pulse_width_bins / (2 * math.sqrt(2 * math.log(2)))
    half = max(math.ceil(3 * sigma), 1)
    x = torch.arange(-half, half + 1, dtype=torch.float32)
    irf = torch.exp(-0.5 * (x / sigma) ** 2)
    return irf / irf.sum()


def apply_irf(transients: Tensor, pulse_width_bins: float) -> Tensor:
    """Convolve transients with a Gaussian laser pulse along the last (time) axis.

    The transient is periodic in the laser period (late returns alias to early bins), so the
    convolution is circular: pulse energy spilling past one end of the window re-enters at
    the other, and the total flux of every transient is conserved.

    Args:
        transients (torch.Tensor): Photon rates of shape ``(..., n_tbins)``.
        pulse_width_bins (float): Pulse full width at half maximum, in time bins. Zero skips
            pulse shaping.

    Returns:
        Shaped transients, same shape as ``transients``.

    Raises:
        ValueError: If ``pulse_width_bins`` is negative (raised by :func:`make_irf`).
    """
    if pulse_width_bins == 0:
        return transients
    irf = make_irf(pulse_width_bins).to(transients.dtype)
    half = irf.numel() // 2
    # Sum of shifted copies handles kernels longer than the window, unlike circular padding.
    return sum(w * torch.roll(transients, shifts=k - half, dims=-1) for k, w in enumerate(irf))


def _resolve_tof_geometry(
    *,
    n_tbins: int | None = None,
    bin_width: float | None = None,
    min_depth: float = 0.0,
    max_depth: float | None = None,
    period: float | None = None,
) -> tuple[int, float, float, float, float]:
    """Resolve the transient grid from any sufficient combination of geometry inputs.

    The span is set by ``max_depth - min_depth``, by one full laser round trip (``period``
    spans ``c * period / 2``), or by ``n_tbins * bin_width``; it must agree across every
    given source. Any sufficient subset resolves the rest. Redundant but consistent inputs
    log a warning; contradictory inputs raise.

    Args:
        n_tbins (int, optional): Number of transient bins over the span. Defaults to none
            (derived from ``span / bin_width``, which must divide evenly).
        bin_width (float, optional): Bin width in metres. Defaults to none (``span / n_tbins``).
        min_depth (float, optional): Depth of the first bin; earlier returns wrap to the
            window's end. Defaults to 0.0.
        max_depth (float, optional): Depth past the last bin, in metres. Defaults to none
            (``min_depth + c * period / 2``).
        period (float, optional): Laser period in seconds, spanning ``c * period / 2`` of
            range. Defaults to none (``2 * span / c``).

    Returns:
        Tuple of ``(n_tbins, bin_width, min_depth, max_depth, period)`` with every value resolved.

    Raises:
        ValueError: If no span or bin count can be derived, or given sources contradict.
    """
    tol = 1e-6
    if n_tbins is not None and (int(n_tbins) != n_tbins or n_tbins <= 0):
        raise ValueError(f"n_tbins must be a positive integer, got {n_tbins!r}")
    if bin_width is not None and bin_width <= 0:
        raise ValueError(f"bin_width must be > 0, got {bin_width!r}")
    if min_depth < 0:
        raise ValueError(f"min_depth must be >= 0, got {min_depth!r}")
    if period is not None and period <= 0:
        raise ValueError(f"period must be > 0, got {period!r}")
    if max_depth is not None and max_depth <= min_depth:
        raise ValueError(f"max_depth must be greater than min_depth ({min_depth!r}), got {max_depth!r}")

    span: float
    span_from_bins = False
    if max_depth is not None:
        span = max_depth - min_depth
        if period is not None:
            period_span = scipy.constants.c * period / 2
            if abs(span - period_span) > tol * max(span, period_span):
                raise ValueError(
                    f"max_depth implies a span of {span!r} m but period implies {period_span!r} m; give only one of them"
                )
            # Over-constrained but consistent: the explicit range wins.
            logger.warning("both max_depth and period given and consistent; using max_depth=%r", max_depth)
    elif period is not None:
        span = scipy.constants.c * period / 2
        max_depth = min_depth + span
    else:
        if n_tbins is None or bin_width is None:
            raise ValueError(
                "ToF geometry is under-specified: provide max_depth or period, or both n_tbins and bin_width"
            )
        span = n_tbins * bin_width
        max_depth = min_depth + span
        span_from_bins = True

    if n_tbins is None:
        if bin_width is None:
            raise ValueError("ToF geometry is under-specified: provide n_tbins or bin_width")
        n_exact = span / bin_width
        n_tbins = round(n_exact)
        if abs(n_exact - n_tbins) > tol * max(n_exact, 1.0):
            raise ValueError(f"bin_width {bin_width!r} m does not divide the span {span!r} m into an integer bin count")
    elif bin_width is None:
        bin_width = span / n_tbins
    elif not span_from_bins and abs(n_tbins * bin_width - span) > tol * max(span, n_tbins * bin_width):
        raise ValueError(
            f"n_tbins * bin_width is {n_tbins * bin_width!r} m but the span is {span!r} m; give only one of them"
        )
    elif not span_from_bins:
        logger.warning("both n_tbins and bin_width given and consistent; using the span %r m", span)

    if period is None:
        period = 2 * span / scipy.constants.c
    return n_tbins, bin_width, min_depth, max_depth, period


def calculate_transients(
    depths: Tensor | npt.NDArray,
    albedos: Tensor | npt.NDArray,
    ambient: Tensor | npt.NDArray | float | None = None,
    *,
    n_tbins: int | None = None,
    bin_width: float | None = None,
    min_depth: float = 0.0,
    max_depth: float | None = None,
    period: float | None = None,
    signal_photons: float = 1.0,
    ambient_photons: float = 4.0,
    reference_depth: float | None = 1.0,
    pulse_width: float = 0.0,
) -> Tensor:
    """Compute per-pixel photon arrival rates from a depth map and reflectance.

    Signal and ambient flux are defined at scene level through a calibration setup
    assumption: ``signal_photons`` is the mean signal photons returned per laser cycle by
    a unit-albedo surface, and ``ambient_photons`` is the scene-average ambient photons per
    cycle, spatially weighted by ``ambient`` (e.g. the mean of a linear RGB render's
    channels) when provided, else uniform. Laser power conversion and ambient light
    source physics are intentionally left out.

    The signal falls off with the inverse square of the true surface depth,
    ``signal_photons * albedo * (reference_depth / depth) ** 2``, so ``signal_photons`` is the return
    of a unit-albedo surface at ``reference_depth``. The falloff uses the depth before range wrapping
    and is floored at one bin width to stay finite near zero. Ambient does not fall off.

    The signal impulse is then shaped by a Gaussian laser pulse of the given FWHM (see
    :func:`make_irf`), which conserves flux: the total rate per pixel is the signal mass plus ``ambient_photons``.

    Non-finite or non-positive depths model no surface and contribute ambient only;
    depths outside ``[min_depth, max_depth)`` alias (range wrapping).

    Args:
        depths (torch.Tensor | npt.NDArray): Depths in metres, arbitrary shape ``(*spatial,)``.
        albedos (torch.Tensor | npt.NDArray): Per-pixel reflectance in ``[0, 1]``, same shape
            as ``depths``.
        ambient (torch.Tensor | npt.NDArray | float, optional): Per-pixel ambient weights of
            positive mean, same shape as ``depths``, or a single positive scalar for the whole scene.
            Defaults to none (uniform ambient).
        n_tbins (int, optional): Number of transient bins over ``max_depth - min_depth``.
            Defaults to none (derived from ``span / bin_width``, which must divide evenly).
        bin_width (float, optional): Bin width in metres. Defaults to none (``span / n_tbins``).
        min_depth (float, optional): Depth of the first bin; earlier returns wrap to the
            window's end. Defaults to 0.0.
        max_depth (float, optional): Depth past the last bin, in metres; ``min_depth +
            c * period / 2`` when only ``period`` is given. Defaults to none (derived).
        period (float, optional): Laser period in seconds, spanning ``c * period / 2`` of
            range. Defaults to none (derived as ``2 * span / c``).
        signal_photons (float, optional): Scene-level signal photons per cycle at unit albedo.
            Defaults to 1.0.
        ambient_photons (float, optional): Scene-average ambient photons per cycle.
            Defaults to 4.0.
        reference_depth (float, optional): Depth in metres at which ``signal_photons`` is defined;
            signal scales as ``(reference_depth / depth) ** 2``. None disables the falloff.
            Defaults to 1.0.
        pulse_width (float, optional): Laser pulse full width at half maximum, in seconds
            (``1e-9`` is 1 ns); zero skips pulse shaping, otherwise it must span at least one
            time bin (``period / n_tbins``). Defaults to 0.0.

    Returns:
        Per-pixel mean photon rate per bin of shape ``(n_pixels, n_tbins)``, one row per flattened pixel.

    Raises:
        ValueError: If ``depths`` is not at least 1-D, ``albedos`` or ``ambient`` does not
            match its shape, ``pulse_width`` is negative or shorter than one time bin, ``ambient`` has non-positive mean, the photon rates are negative,
            or the geometry is under-specified or contradictory.
    """
    depths = torch.as_tensor(depths, dtype=torch.float32)
    albedos = torch.as_tensor(albedos, dtype=torch.float32)
    if depths.ndim < 1:
        raise ValueError(f"depths must be at least 1-D, got shape {tuple(depths.shape)}")
    if albedos.shape != depths.shape:
        raise ValueError(
            f"albedos and depths must have the same shape, got {tuple(albedos.shape)} and {tuple(depths.shape)}"
        )
    if signal_photons < 0 or ambient_photons < 0:
        raise ValueError(
            f"signal_photons and ambient_photons must be non-negative, got {signal_photons!r} and {ambient_photons!r}"
        )

    if reference_depth is not None and reference_depth <= 0:
        raise ValueError(f"reference_depth must be > 0 or None, got {reference_depth!r}")

    if pulse_width < 0:
        raise ValueError(f"pulse_width must be non-negative, got {pulse_width!r}")

    n_tbins, bin_width, min_depth, max_depth, period = _resolve_tof_geometry(
        n_tbins=n_tbins, bin_width=bin_width, min_depth=min_depth, max_depth=max_depth, period=period
    )

    # One bin lasts period / n_tbins; narrower pulses are not resolved by the sampled kernel.
    bin_time = period / n_tbins
    if 0 < pulse_width < bin_time * (1 - 1e-9):
        raise ValueError(
            f"pulse_width must be 0 (no shaping) or at least one time bin ({bin_time!r} s for this geometry), "
            f"got {pulse_width!r}"
        )

    if ambient is None:
        weights = torch.ones_like(depths)
    else:
        ambient_map: Tensor = torch.as_tensor(ambient, dtype=torch.float32)
        if ambient_map.ndim == 0:
            # A scalar weights every pixel equally; mean-normalization cancels its value.
            ambient_map = torch.full_like(depths, float(ambient_map))
        if ambient_map.shape != depths.shape:
            raise ValueError(
                f"ambient and depths must have the same shape, got {tuple(ambient_map.shape)} and {tuple(depths.shape)}"
            )
        if float(ambient_map.mean()) <= 0:
            raise ValueError("ambient must have a positive mean")
        weights = ambient_map / ambient_map.mean()

    # Flatten to one row per pixel so valid rows and bins index 1-D against each other.
    depths = depths.reshape(-1)
    albedos = albedos.reshape(-1)
    weights = weights.reshape(-1)
    valid = torch.isfinite(depths) & (depths > 0)

    # Returns outside [min_depth, max_depth) alias: earlier arrivals wrap to the window's end.
    span = max_depth - min_depth
    transient_idx = (torch.remainder(depths - min_depth, span) * (n_tbins / span)).floor().to(torch.int64)
    transient_idx = transient_idx.clamp(max=n_tbins - 1)

    # Signal: one rate bin at the depth bin, with inverse-square falloff relative to reference_depth.
    transients = torch.zeros(depths.shape[0], n_tbins, dtype=torch.float32)
    rows = torch.nonzero(valid).squeeze(-1)
    signal = signal_photons * albedos[valid]
    if reference_depth is not None:
        # True depth, not the wrapped one: an aliased return is still as faint as its real distance.
        signal = signal * (reference_depth / depths[valid].clamp(min=bin_width)) ** 2
    transients[rows, transient_idx[valid]] = signal

    # Ambient: ambient_photons per cycle spread flat across the period.
    transients += (ambient_photons * weights / n_tbins).unsqueeze(1)
    # Convert the pulse FWHM from seconds to bins: one bin lasts period / n_tbins.
    return apply_irf(transients, pulse_width * n_tbins / period)


def emulate_dtof(
    depths: Tensor | npt.NDArray,
    albedos: Tensor | npt.NDArray,
    ambient: Tensor | npt.NDArray | float | None = None,
    *,
    n_tbins: int | None = None,
    bin_width: float | None = None,
    min_depth: float = 0.0,
    max_depth: float | None = None,
    period: float | None = None,
    signal_photons: float = 1.0,
    ambient_photons: float = 4.0,
    reference_depth: float | None = 1.0,
    n_pulses: int = 1000,
    n_bins: int | None = None,
    dead_time_bins: int = 10,
    free_running: bool = True,
    model: DtoFModel = "mc",
    pulse_width: float = 0.0,
    generator: torch.Generator | None = None,
) -> npt.NDArray[np.floating]:
    """Emulate dToF equi-width histogram (EWH) measurements for one frame.

    Signal and ambient flux follow the scene-level calibration of
    :func:`calculate_transients`; passing an ambient map (e.g. the mean of a linear
    RGB render's channels) weights the ambient flux per pixel, while the albedo map
    scales the signal return.

    Two readout models are supported: ``"mc"`` simulates every laser cycle with
    Poisson photon arrivals and non-paralyzable dead time (iterative), while
    ``"forward"`` applies the single-pass analytical model of the chosen operation
    mode. Synchronous operation (``free_running=False``) records at most one photon
    per cycle and ignores ``dead_time_bins``.

    Args:
        depths (torch.Tensor | npt.NDArray): Depths in metres, arbitrary shape ``(*spatial,)``.
        albedos (torch.Tensor | npt.NDArray): Per-pixel reflectance in ``[0, 1]``, same shape
            as ``depths``.
        ambient (torch.Tensor | npt.NDArray | float, optional): Per-pixel ambient weights of
            positive mean, same shape as ``depths``, or a single positive scalar for the whole scene.
            Defaults to none (uniform ambient).
        n_tbins (int, optional): Number of transient bins over ``max_depth - min_depth``.
            Defaults to none (derived from ``span / bin_width``, which must divide evenly).
        bin_width (float, optional): Bin width in metres. Defaults to none (``span / n_tbins``).
        min_depth (float, optional): Depth of the first bin; earlier returns wrap to the
            window's end. Defaults to 0.0.
        max_depth (float, optional): Depth past the last bin, in metres; ``min_depth +
            c * period / 2`` when only ``period`` is given. Defaults to none (derived).
        period (float, optional): Laser period in seconds, spanning ``c * period / 2`` of
            range. Defaults to none (derived as ``2 * span / c``).
        signal_photons (float, optional): Scene-level signal photons per cycle at unit albedo.
            Defaults to 1.0.
        ambient_photons (float, optional): Scene-average ambient photons per cycle. Defaults to 4.0.
        reference_depth (float, optional): Depth in metres at which ``signal_photons`` is defined;
            signal falls off as the inverse square of depth. None disables it. Defaults to 1.0.
        n_pulses (int, optional): Number of laser cycles per frame. Defaults to 1000.
        n_bins (int, optional): Number of EWH bins; must divide ``n_tbins``. Defaults to none
            (use ``n_tbins``).
        dead_time_bins (int, optional): Non-paralyzable dead time in time bins, free-running
            only. Defaults to 10.
        free_running (bool, optional): Free-running (True) or synchronous single-hit (False)
            operation. Defaults to True.
        model (DtoFModel, optional): Readout model, per-cycle Monte-Carlo (``"mc"``) or
            single-pass analytical (``"forward"``). Defaults to ``"mc"``.
        pulse_width (float, optional): Laser pulse full width at half maximum, in seconds
            (``1e-9`` is 1 ns); zero skips pulse shaping. Defaults to 0.0.
        generator (torch.Generator, optional): Seeded CPU generator for ``"mc"``. Defaults to
            none (fresh entropy).

    Returns:
        EWH photon counts of shape ``(*spatial, n_bins)``: realized counts for ``"mc"``, expected
        counts for ``"forward"``.

    Raises:
        ValueError: If ``model`` is unknown, ``n_pulses`` is not positive, ``dead_time_bins``
            is not a non-negative integer, ``n_tbins`` is not an exact multiple of ``n_bins``,
            or the geometry is under-specified or contradictory.
    """
    if model not in ("mc", "forward"):
        raise ValueError(f"model must be one of 'mc' or 'forward', got {model!r}")
    if model == "forward":
        # TODO: forward models land with the single-pixel forward models and tests.
        raise NotImplementedError("model='forward' is not available yet; use model='mc'")
    if n_pulses <= 0:
        raise ValueError(f"n_pulses must be > 0, got {n_pulses}")
    if int(dead_time_bins) != dead_time_bins or dead_time_bins < 0:
        raise ValueError(f"dead_time_bins must be a non-negative integer, got {dead_time_bins!r}")
    dead_time_bins = int(dead_time_bins)
    n_tbins, _bin_width, min_depth, max_depth, _period = _resolve_tof_geometry(
        n_tbins=n_tbins, bin_width=bin_width, min_depth=min_depth, max_depth=max_depth, period=period
    )
    n_bins = n_tbins if n_bins is None else n_bins
    if n_tbins % n_bins != 0:
        raise ValueError(f"n_tbins ({n_tbins}) must be an exact multiple of n_bins ({n_bins})")

    transients = calculate_transients(
        depths,
        albedos,
        ambient,
        n_tbins=n_tbins,
        min_depth=min_depth,
        max_depth=max_depth,
        signal_photons=signal_photons,
        ambient_photons=ambient_photons,
        reference_depth=reference_depth,
        pulse_width=pulse_width,
    )
    shape = np.shape(depths)

    ewh = torch.stack(
        [
            simulate_ewh(
                row,
                n_pulses=n_pulses,
                n_bins=n_bins,
                dead_time_bins=dead_time_bins,
                free_running=free_running,
                generator=generator,
            )
            for row in transients
        ]
    )
    return ewh.reshape(*shape, n_bins).numpy()
