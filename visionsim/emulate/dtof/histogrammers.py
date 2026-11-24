"""Single-pixel dToF kernels: Monte-Carlo photon sampling with dead time, analytical forward models and histogramming."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


def _sample_occupied_cells(
    rates: Tensor,
    n_pulses: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Draw photon arrivals of a single pixel over ``n_pulses`` laser cycles as occupied cells.

    Samples the total photon count, then each photon's bin and cycle, instead of a dense
    ``(n_pulses, n_tbins)`` Poisson table: the cost and memory scale with the number of photons,
    which suits the low flux (well under one photon per bin) of realistic dToF scenes.

    Args:
        rates (torch.Tensor): 1-D non-negative mean photon arrival rate per time bin.
        n_pulses (int): Number of laser cycles to sample.
        generator (torch.Generator, optional): Seeded CPU generator. Defaults to none (fresh entropy).

    Returns:
        Tuple of the sorted flat indices ``cycle * n_tbins + bin`` of occupied cells and the
        photon count of each occupied cell.
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
    bins = torch.multinomial(rates.to(torch.float64), n_photons, replacement=True, generator=generator)
    cycles = torch.randint(0, n_pulses, (n_photons,), generator=generator, dtype=torch.int64)

    g, _ = torch.sort(cycles * n_tbins + bins)
    occupied, counts = torch.unique_consecutive(g, return_counts=True)
    return occupied.to(torch.int64), counts.to(torch.int64)


def _walk_occupied(
    occupied: npt.NDArray[np.int64],
    occ_counts: npt.NDArray[np.int64],
    n_tbins: int,
    dead_time_bins: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Sequentially scan occupied cells under non-paralyzable dead time. Returns (cycles, bins)."""
    out_cycle: npt.NDArray[np.int64] = np.empty(int(occ_counts.sum()), dtype=np.int64)
    out_bin: npt.NDArray[np.int64] = np.empty(out_cycle.shape[0], dtype=np.int64)
    n_out = 0

    t_armed = -1
    for i in range(occupied.shape[0]):
        g = int(occupied[i])
        if g < t_armed:
            continue

        # A cell inside dead time yields one detection; without dead time every photon is detected.
        n_emit = int(occ_counts[i]) if dead_time_bins == 0 else 1
        cycle, bin_ = divmod(g, n_tbins)
        for _ in range(n_emit):
            out_cycle[n_out] = cycle
            out_bin[n_out] = bin_
            n_out += 1
        t_armed = g + dead_time_bins

    return out_cycle[:n_out], out_bin[:n_out]


def simulate_photon_timestamps(
    phi_bar: Tensor,
    n_pulses: int,
    *,
    dead_time_bins: int = 0,
    free_running: bool = True,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Simulate photon detection timestamps of a single pixel over ``n_pulses`` laser cycles.

    Synchronous operation (``free_running=False``) arms the receiver at every cycle start and
    records at most one photon per cycle, so ``dead_time_bins`` has no effect. Free-running
    operation keeps the receiver armed across cycles and applies a non-paralyzable dead time
    of ``dead_time_bins`` bins after every detection.

    Dead-time convention: a detection in bin ``g`` re-arms the receiver at bin ``g + dead_time_bins``,
    so consecutive detections are at least ``dead_time_bins`` bins apart (``dead_time_bins - 1`` bins
    are blocked). Dead time carries across cycle boundaries, so values above ``n_tbins`` are supported.
    With ``dead_time_bins == 0`` every photon is detected, so a bin holding several photons yields
    several detections; with ``dead_time_bins >= 1`` a bin yields at most one detection.

    Args:
        phi_bar (torch.Tensor): 1-D mean photon arrival rate per time bin of a single pixel.
        n_pulses (int): Number of laser cycles to simulate.
        dead_time_bins (int, optional): Non-paralyzable dead time in time bins. Ignored when
            ``free_running`` is False. Defaults to 0.
        free_running (bool, optional): Free-running (True) or synchronous single-hit (False)
            operation. Defaults to True.
        generator (torch.Generator, optional): Seeded CPU generator. Defaults to none (fresh entropy).

    Returns:
        Integer ``(n, 2)`` detection timestamps as (cycle, bin) pairs in detection order.

    Raises:
        ValueError: If ``phi_bar`` is not 1-D, ``n_pulses`` is not positive, or ``dead_time_bins``
            is not a non-negative integer.
    """
    if phi_bar.ndim != 1:
        raise ValueError(f"phi_bar must be 1-D (single pixel), got shape {tuple(phi_bar.shape)}")
    if n_pulses <= 0:
        raise ValueError(f"n_pulses must be > 0, got {n_pulses}")
    if int(dead_time_bins) != dead_time_bins or dead_time_bins < 0:
        raise ValueError(f"dead_time_bins must be a non-negative integer, got {dead_time_bins!r}")
    dead_time_bins = int(dead_time_bins)

    rates = torch.clamp(phi_bar.to(torch.float32), min=0.0)
    n_tbins = int(rates.shape[-1])

    occupied, occ_counts = _sample_occupied_cells(rates, n_pulses, generator=generator)
    occupied_np, occ_counts_np = occupied.numpy(), occ_counts.numpy()

    if not free_running:
        # Synchronous single-hit: keep only the first occupied bin of every cycle.
        if occupied_np.shape[0] == 0:
            return torch.zeros((0, 2), dtype=torch.int64)
        cycle_of = occupied_np // n_tbins
        first = np.empty(cycle_of.shape[0], dtype=bool)
        first[0] = True
        np.not_equal(cycle_of[1:], cycle_of[:-1], out=first[1:])
        g = occupied_np[first]
        cyc, bns = cycle_of[first], g % n_tbins
        return torch.stack([torch.from_numpy(cyc), torch.from_numpy(bns)], dim=1).to(torch.int64)

    cyc, bns = _walk_occupied(occupied_np, occ_counts_np, n_tbins, dead_time_bins)
    return torch.stack([torch.from_numpy(cyc), torch.from_numpy(bns)], dim=1).to(torch.int64)


def timestamps_to_histogram(timestamps: Tensor, n_tbins: int, n_hist_bins: int | None = None) -> Tensor:
    """Reduce ``(cycle, bin)`` timestamps to an equi-width photon histogram (EWH).

    Args:
        timestamps (torch.Tensor): Integer ``(n, 2)`` (cycle, bin) detection timestamps.
        n_tbins (int): Number of time bins the timestamps are expressed in.
        n_hist_bins (int, optional): Number of histogram bins; must divide ``n_tbins``.
            Defaults to none (use ``n_tbins``).

    Returns:
        Photon counts of shape ``(n_hist_bins,)``.

    Raises:
        ValueError: If ``timestamps`` is not ``(n, 2)``, ``n_hist_bins`` exceeds ``n_tbins``, or
            ``n_tbins`` is not an exact multiple of ``n_hist_bins``.
    """
    if timestamps.ndim != 2 or timestamps.shape[1] != 2:
        raise ValueError(f"timestamps must be (n, 2) (cycle, bin) pairs, got shape {tuple(timestamps.shape)}")
    if n_hist_bins is None:
        n_hist_bins = n_tbins
    if n_hist_bins > n_tbins:
        raise ValueError(f"n_hist_bins ({n_hist_bins}) cannot exceed TDC resolution n_tbins ({n_tbins})")
    if n_tbins % n_hist_bins != 0:
        raise ValueError(f"n_tbins ({n_tbins}) must be an exact multiple of n_hist_bins ({n_hist_bins})")

    if timestamps.numel() == 0:
        return torch.zeros(n_hist_bins, dtype=torch.float32)

    idx = timestamps[:, 1] * n_hist_bins // n_tbins
    return torch.bincount(idx, minlength=n_hist_bins).to(torch.float32)


def forward_sync(phi_bar: Tensor) -> Tensor:
    """Analytical single-pixel forward model (Coates) for synchronous operation.

    The receiver records only the first photon of each cycle, so bin ``k`` is detected with
    probability ``exp(-sum_{j<k} phi_j) * (1 - exp(-phi_k))``: no photon arrived before ``k`` and
    at least one arrives in ``k``. This is the exact expectation of the synchronous Monte-Carlo.

    Args:
        phi_bar (torch.Tensor): 1-D mean photon arrival rate per time bin of a single pixel.

    Returns:
        Detection probability per bin per laser cycle, float64, shape ``(n_tbins,)``. It sums to
        the single-hit probability ``1 - exp(-sum(phi_bar))``.

    Raises:
        ValueError: If ``phi_bar`` is not 1-D.
    """
    if phi_bar.ndim != 1:
        raise ValueError(f"phi_bar must be 1-D (single pixel), got shape {tuple(phi_bar.shape)}")
    phi = torch.clamp(phi_bar.to(torch.float64), min=0.0)
    # Flux accumulated strictly before each bin; exp/expm1 keep precision at low flux.
    before = torch.cumsum(phi, dim=0) - phi
    return torch.exp(-before) * -torch.expm1(-phi)


def simulate_ewh(
    phi_bar: Tensor,
    *,
    n_pulses: int,
    n_bins: int | None = None,
    dead_time_bins: int = 0,
    free_running: bool = True,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Simulate a single pixel's equi-width histogram (EWH) via per-cycle Monte-Carlo.

    Args:
        phi_bar (torch.Tensor): 1-D mean photon arrival rate per time bin of a single pixel.
        n_pulses (int): Number of laser cycles to simulate.
        n_bins (int, optional): Number of histogram bins; must divide the transient width.
            Defaults to none (use the transient width).
        dead_time_bins (int, optional): Non-paralyzable dead time in time bins, free-running
            only. Defaults to 0.
        free_running (bool, optional): Free-running (True) or synchronous single-hit (False)
            operation. Defaults to True.
        generator (torch.Generator, optional): Seeded CPU generator. Defaults to none (fresh entropy).

    Returns:
        Realized photon counts of shape ``(n_bins,)`` over ``n_pulses`` cycles.
    """
    timestamps = simulate_photon_timestamps(
        phi_bar, n_pulses, dead_time_bins=dead_time_bins, free_running=free_running, generator=generator
    )
    return timestamps_to_histogram(timestamps, phi_bar.shape[-1], n_bins)
