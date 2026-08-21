"""Ground-truth photon-detection simulator for a single SPAD pixel.

This module implements forward model #1 (the Monte-Carlo reference). Unlike the
histogrammers, it emits raw **timestamps** rather than a histogram; the histogram
is a reduction of the timestamps (see :func:`timestamps_to_histogram`), which is
also how real hardware works -- a TDC emits timestamps, binning happens later.

Photon model
------------
Arrivals are drawn per bin as ``N_b ~ Poisson(phi_bar[b])`` -- the *count* of
photons landing in bin ``b``, not merely whether one did. Sampling the count
rather than a Bernoulli indicator matters in two places:

* At ``dead_time_bins == 0`` every arriving photon is detected, so the detection
  rate is proportional to ``phi``. A Bernoulli-per-bin sampler caps detections at
  one per bin and yields ``1 - exp(-phi)`` instead, which is a flux-dependent
  undercount (4.8% at ``phi=0.1`` per bin, 21% at ``phi=0.5``, 37% at ``phi=1``).
* The one-detection-per-bin behaviour that real detectors *do* exhibit then
  emerges from the dead time rather than being assumed a priori.

Photons within a bin are treated as arriving at the **bin centre**. Since every
photon shares that offset the half-bin cancels out of the dead-time comparison,
so the walk below is exact in bin units. The cost of this model is that dead
times shorter than one bin cannot be represented -- see ``dead_time_bins``.

Modes
-----
free-running (asynchronous)
    The detector re-arms ``dead_time_bins`` after each detection, independent of
    cycle boundaries. Dead time wraps across cycles. Multi-hit is inherent and
    always enabled. Because the walk tracks arm time in *absolute* bins, a dead
    time longer than one cycle is handled correctly rather than saturating at
    one cycle.

synchronous (gated)
    The detector is re-armed at the start of every cycle, so dead time never
    crosses a cycle boundary. Defaults to one detection per cycle, the
    conventional single-photon lidar setup for which the Coates estimator is
    derived. Pass ``max_detections_per_cycle=None`` for gated multi-hit.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "simulate_photon_timestamps",
    "timestamps_to_histogram",
    "sample_photon_arrivals",
]


def sample_photon_arrivals(
    phi_bar: Tensor,
    n_pulses: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw per-bin photon arrival *counts* for ``n_pulses`` laser cycles.

    Args:
        phi_bar: Photon arrival rate per bin for one pixel, shape ``(n_tbins,)``.
            Negative entries (which convolution and offset arithmetic can
            produce) are clamped to zero.
        n_pulses: Number of laser cycles to simulate.
        generator: Optional RNG for reproducibility.

    Returns:
        Integer tensor of shape ``(n_pulses, n_tbins)`` holding photon counts.
    """
    if phi_bar.ndim != 1:
        raise ValueError(f"phi_bar must be 1-D (single pixel), got shape {tuple(phi_bar.shape)}")
    if n_pulses <= 0:
        raise ValueError(f"n_pulses must be > 0, got {n_pulses}")

    rates = torch.clamp(phi_bar.to(torch.float32), min=0.0)
    rates = rates.expand(n_pulses, -1)
    counts = torch.poisson(rates, generator=generator)
    return counts.to(torch.int64)


def _sample_occupied_cells(
    rates: Tensor,
    n_pulses: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sparse equivalent of :func:`sample_photon_arrivals`, without the dense grid.

    Exploits the superposition property of the Poisson distribution: the total
    photon count over all ``n_pulses x n_tbins`` cells is
    ``Poisson(n_pulses * sum(phi))``, and conditional on that total each photon
    lands in a cell drawn independently -- cycle uniform, bin categorical with
    weights ``phi / sum(phi)``. That is *exact*, not an approximation, and costs
    O(number of photons) instead of O(cycles x bins).

    Returns sorted global cell indices ``cycle * n_tbins + bin`` and the photon
    count in each, covering only cells that received at least one photon.
    """
    empty = torch.zeros(0, dtype=torch.int64)
    total_rate = float(rates.sum())
    if total_rate <= 0.0:
        return empty, empty

    lam = torch.tensor(total_rate * n_pulses, dtype=torch.float64)
    n_photons = int(torch.poisson(lam, generator=generator).item())
    if n_photons == 0:
        return empty, empty

    n_tbins = int(rates.shape[0])
    bins = torch.multinomial(
        rates.to(torch.float64), n_photons, replacement=True, generator=generator
    )
    cycles = torch.randint(0, n_pulses, (n_photons,), generator=generator, dtype=torch.int64)

    g, _ = torch.sort(cycles * n_tbins + bins)
    occupied, counts = torch.unique_consecutive(g, return_counts=True)
    return occupied.to(torch.int64), counts.to(torch.int64)


def _deadtime_walk(
    counts: np.ndarray,
    n_tbins: int,
    dead_time_bins: int,
    free_running: bool,
    paralyzable: bool,
    max_det_per_cycle: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense-input wrapper around :func:`_walk_occupied` (used by the tests)."""
    flat = counts.reshape(-1)
    occupied = np.nonzero(flat)[0]
    return _walk_occupied(
        occupied, flat[occupied], n_tbins, dead_time_bins, free_running, paralyzable,
        max_det_per_cycle,
    )


def _walk_occupied(
    occupied: np.ndarray,
    occ_counts: np.ndarray,
    n_tbins: int,
    dead_time_bins: int,
    free_running: bool,
    paralyzable: bool,
    max_det_per_cycle: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential dead-time scan over occupied bins. Returns ``(cycles, bins)``.

    Args:
        occupied: Sorted global bin indices ``cycle * n_tbins + bin`` that
            received at least one photon.
        occ_counts: Photon count in each of those cells.

    Kept as plain numpy with scalar-only arithmetic so it stays ``numba.njit``
    compatible if this scan ever becomes the bottleneck. It visits only occupied
    cells, so its cost is O(number of photons) rather than O(cycles x bins).

    ``t_armed`` is the earliest absolute bin index at which the detector is
    sensitive again; the detector is armed at global bin ``g`` iff
    ``g >= t_armed``. Tracking it in absolute bins is what lets a dead time
    exceeding one cycle work in free-running mode.
    """
    out_cycle = np.empty(int(occ_counts.sum()), dtype=np.int64)
    out_bin = np.empty(out_cycle.shape[0], dtype=np.int64)
    n_out = 0

    t_armed = -1  # armed from the start
    cur_cycle = -1
    n_det_this_cycle = 0

    for i in range(occupied.shape[0]):
        g = int(occupied[i])
        cycle = g // n_tbins

        if cycle != cur_cycle:
            cur_cycle = cycle
            n_det_this_cycle = 0
            if not free_running:
                # Gated: the detector is re-armed at every cycle boundary, so no
                # dead time is carried in from the previous cycle. Cycles with no
                # arrivals are skipped entirely, which is harmless because
                # nothing can happen in them.
                t_armed = cycle * n_tbins

        if n_det_this_cycle >= max_det_per_cycle:
            continue

        if g >= t_armed:
            # With a non-zero dead time only the first photon in the bin can be
            # detected: the rest arrive at the same bin centre, inside the window
            # that the first one just opened. With zero dead time they all count.
            n_emit = int(occ_counts[i]) if dead_time_bins == 0 else 1
            headroom = max_det_per_cycle - n_det_this_cycle
            if n_emit > headroom:
                n_emit = headroom

            for _ in range(n_emit):
                out_cycle[n_out] = cycle
                out_bin[n_out] = g - cycle * n_tbins
                n_out += 1
            n_det_this_cycle += n_emit
            t_armed = g + dead_time_bins
        elif paralyzable:
            # Paralyzable: an arrival that is *not* detected still retriggers the
            # quench, extending the dead window. This is the only line in which
            # the two dead-time models differ.
            t_armed = g + dead_time_bins

    return out_cycle[:n_out], out_bin[:n_out]


def simulate_photon_timestamps(
    phi_bar: Tensor,
    n_pulses: int,
    *,
    dead_time_bins: int = 0,
    free_running: bool = True,
    paralyzable: bool = False,
    max_detections_per_cycle: int | None = -1,
    generator: torch.Generator | None = None,
    sparse: bool | None = None,
) -> Tensor:
    """Simulate photon detection timestamps for a single pixel.

    Args:
        phi_bar: Photon arrival rate per bin, shape ``(n_tbins,)``.
        n_pulses: Number of laser cycles to simulate.
        dead_time_bins: Dead time in whole time bins. Must be a non-negative
            integer; sub-bin dead times are not representable in this model. A
            dead time longer than one cycle is supported in free-running mode.
        free_running: True for asynchronous mode, False for gated/synchronous.
        paralyzable: If True, every *arrival* retriggers the dead window whether
            or not it was detected (passive quenching). If False (the default),
            only detections do (active quenching).
        max_detections_per_cycle: ``-1`` selects the per-mode default: one
            detection per cycle when gated, unlimited when free-running. Pass
            ``None`` for unlimited or a positive integer for an explicit cap.
            Free-running requires unlimited.
        generator: Optional RNG for reproducibility.
        sparse: Arrival-sampling strategy. ``None`` (the default) picks
            automatically: the sparse sampler is used whenever the expected
            photon count is small relative to the ``n_pulses x n_tbins`` grid,
            which is the normal low-flux regime and avoids drawing millions of
            zeros. Both strategies are exact and statistically equivalent, but
            they consume the RNG differently, so a fixed seed only reproduces a
            given stream for a fixed choice. Pass an explicit bool to pin it.

    Returns:
        Int64 tensor of shape ``(n_detections, 2)``; column 0 is the cycle index
        and column 1 is the time bin within that cycle. Rows are sorted by
        ``(cycle, bin)``. Entries are **not** guaranteed unique: with
        ``dead_time_bins == 0`` several photons in one bin each produce a row.
    """
    if int(dead_time_bins) != dead_time_bins or dead_time_bins < 0:
        raise ValueError(f"dead_time_bins must be a non-negative integer, got {dead_time_bins!r}")
    dead_time_bins = int(dead_time_bins)

    if max_detections_per_cycle == -1:
        max_detections_per_cycle = None if free_running else 1
    if free_running and max_detections_per_cycle is not None:
        raise ValueError(
            "free-running mode is inherently multi-hit; max_detections_per_cycle "
            f"must be None, got {max_detections_per_cycle}"
        )
    if max_detections_per_cycle is not None and max_detections_per_cycle < 1:
        raise ValueError(
            f"max_detections_per_cycle must be >= 1 or None, got {max_detections_per_cycle}"
        )

    if phi_bar.ndim != 1:
        raise ValueError(f"phi_bar must be 1-D (single pixel), got shape {tuple(phi_bar.shape)}")
    n_tbins = int(phi_bar.shape[-1])
    rates = torch.clamp(phi_bar.to(torch.float32), min=0.0)

    if sparse is None:
        # The sparse sampler costs O(photons) and the dense one O(cycles x bins),
        # with a constant-factor penalty for the sort. Prefer sparse whenever the
        # grid is expected to be mostly empty.
        sparse = float(rates.sum()) * n_pulses < 0.25 * n_pulses * n_tbins

    if sparse:
        occupied, occ_counts = _sample_occupied_cells(rates, n_pulses, generator=generator)
        occupied_np, occ_counts_np = occupied.numpy(), occ_counts.numpy()
    else:
        flat = sample_photon_arrivals(rates, n_pulses, generator=generator).reshape(-1).numpy()
        occupied_np = np.nonzero(flat)[0]
        occ_counts_np = flat[occupied_np]

    if not free_running and max_detections_per_cycle == 1:
        # Conventional gated single-hit. The detector is re-armed at every cycle
        # start and stops after the first detection, so dead time can never block
        # anything -- the answer is just the first occupied cell of each cycle,
        # which vectorises with no scan at all.
        if occupied_np.shape[0] == 0:
            return torch.zeros((0, 2), dtype=torch.int64)
        cycle_of = occupied_np // n_tbins
        first = np.empty(cycle_of.shape[0], dtype=bool)
        first[0] = True
        np.not_equal(cycle_of[1:], cycle_of[:-1], out=first[1:])
        g = occupied_np[first]
        return torch.stack(
            [torch.from_numpy(cycle_of[first]), torch.from_numpy(g % n_tbins)], dim=1
        ).to(torch.int64)

    cap = (1 << 62) if max_detections_per_cycle is None else int(max_detections_per_cycle)
    cyc, bns = _walk_occupied(
        occupied_np,
        occ_counts_np,
        n_tbins,
        dead_time_bins,
        bool(free_running),
        bool(paralyzable),
        cap,
    )
    return torch.stack(
        [torch.from_numpy(cyc), torch.from_numpy(bns)], dim=1
    ).to(torch.int64)


def timestamps_to_histogram(
    timestamps: Tensor,
    n_tbins: int,
    n_hist_bins: int | None = None,
) -> Tensor:
    """Reduce ``(cycle, bin)`` timestamps to an equi-width photon histogram.

    Args:
        timestamps: ``(n_detections, 2)`` tensor from
            :func:`simulate_photon_timestamps`.
        n_tbins: Number of time bins per cycle used to produce the timestamps.
        n_hist_bins: Histogram resolution. Defaults to ``n_tbins``. A coarser
            histogram is produced by merging adjacent bins, which requires
            ``n_tbins`` to be an exact multiple of ``n_hist_bins``.

    Returns:
        Float32 tensor of shape ``(n_hist_bins,)`` holding detection counts.
    """
    if n_hist_bins is None:
        n_hist_bins = n_tbins
    if n_hist_bins > n_tbins:
        raise ValueError(
            f"n_hist_bins ({n_hist_bins}) cannot exceed the TDC resolution n_tbins ({n_tbins})"
        )
    if n_tbins % n_hist_bins != 0:
        raise ValueError(
            f"n_tbins ({n_tbins}) must be an exact multiple of n_hist_bins ({n_hist_bins}) "
            "so that time bins merge evenly"
        )

    if timestamps.numel() == 0:
        return torch.zeros(n_hist_bins, dtype=torch.float32)

    idx = timestamps[:, 1] * n_hist_bins // n_tbins
    return torch.bincount(idx, minlength=n_hist_bins).to(torch.float32)
