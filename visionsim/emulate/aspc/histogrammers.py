"""Ground-truth photon-detection simulator for SPAD sensor systems."""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from visionsim.utils.progress import ElapsedProgress

# =============================================================================
# MONTE-CARLO SAMPLING & DEAD-TIME WALK KERNELS
# =============================================================================


def sample_photon_arrivals(
    phi_bar: Tensor,
    n_pulses: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw per-bin photon arrival counts for n_pulses laser cycles."""
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
    """Sparse equivalent of sample_photon_arrivals."""
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
    occupied: np.ndarray,
    occ_counts: np.ndarray,
    n_tbins: int,
    dead_time_bins: int,
    free_running: bool,
    paralyzable: bool,
    max_det_per_cycle: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential dead-time scan over occupied bins. Returns (cycles, bins)."""
    out_cycle: np.ndarray = np.empty(int(occ_counts.sum()), dtype=np.int64)
    out_bin: np.ndarray = np.empty(out_cycle.shape[0], dtype=np.int64)
    n_out = 0

    t_armed = -1
    cur_cycle = -1
    n_det_this_cycle = 0

    for i in range(occupied.shape[0]):
        g = int(occupied[i])
        cycle = g // n_tbins

        if cycle != cur_cycle:
            cur_cycle = cycle
            n_det_this_cycle = 0
            if not free_running:
                t_armed = cycle * n_tbins

        if n_det_this_cycle >= max_det_per_cycle:
            continue

        if g >= t_armed:
            n_emit = int(occ_counts[i]) if dead_time_bins == 0 else 1
            headroom = max_det_per_cycle - n_det_this_cycle
            n_emit = min(n_emit, headroom)

            for _ in range(n_emit):
                out_cycle[n_out] = cycle
                out_bin[n_out] = g - cycle * n_tbins
                n_out += 1
            n_det_this_cycle += n_emit
            t_armed = g + dead_time_bins
        elif paralyzable:
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
    """Simulate photon detection timestamps for a single pixel."""
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

    if phi_bar.ndim != 1:
        raise ValueError(f"phi_bar must be 1-D (single pixel), got shape {tuple(phi_bar.shape)}")
    n_tbins = int(phi_bar.shape[-1])
    rates = torch.clamp(phi_bar.to(torch.float32), min=0.0)

    if sparse is None:
        sparse = float(rates.sum()) * n_pulses < 0.25 * n_pulses * n_tbins

    if sparse:
        occupied, occ_counts = _sample_occupied_cells(rates, n_pulses, generator=generator)
        occupied_np, occ_counts_np = occupied.numpy(), occ_counts.numpy()
    else:
        flat = sample_photon_arrivals(rates, n_pulses, generator=generator).reshape(-1).numpy()
        occupied_np = np.nonzero(flat)[0]
        occ_counts_np = flat[occupied_np]

    if not free_running and max_detections_per_cycle == 1:
        if occupied_np.shape[0] == 0:
            return torch.zeros((0, 2), dtype=torch.int64)
        cycle_of = occupied_np // n_tbins
        first = np.empty(cycle_of.shape[0], dtype=bool)
        first[0] = True
        np.not_equal(cycle_of[1:], cycle_of[:-1], out=first[1:])
        g = occupied_np[first]
        return torch.stack([torch.from_numpy(cycle_of[first]), torch.from_numpy(g % n_tbins)], dim=1).to(torch.int64)

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
    return torch.stack([torch.from_numpy(cyc), torch.from_numpy(bns)], dim=1).to(torch.int64)


def timestamps_to_histogram(
    timestamps: Tensor,
    n_tbins: int,
    n_hist_bins: int | None = None,
) -> Tensor:
    """Reduce (cycle, bin) timestamps to an equi-width photon histogram."""
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


# =============================================================================
# PILE-UP DISTORTION FORWARD MODELS
# =============================================================================


def batch_distorted_transient_sync(
    phi_bar: torch.Tensor,
    dead_time_bins: int,
    n_bins: int,
) -> torch.Tensor:
    """Batched single-hit synchronous forward model (Coates)."""
    p_detect = 1.0 - torch.exp(-torch.clamp(phi_bar[:, :n_bins], min=0.0))
    survival = torch.ones_like(p_detect)
    survival[:, 1:] = torch.cumprod(1.0 - p_detect[:, :-1], dim=-1)
    prob_detect = p_detect * survival
    total = prob_detect.sum(dim=-1, keepdim=True)
    return prob_detect / (total + 1e-12)


def batch_distorted_transient_async(
    phi_bar: torch.Tensor,
    dead_time_bins: int,
    n_bins: int,
    n_iterations: int = 150,
) -> torch.Tensor:
    """Batched asynchronous (free-running) forward model using power iteration."""
    if phi_bar.ndim == 1:
        phi_bar = phi_bar.unsqueeze(0)

    phi = torch.clamp(phi_bar[:, :n_bins], min=0.0)
    total = phi.sum(dim=-1, keepdim=True)

    if dead_time_bins == 0:
        zero_mask = total.squeeze(-1) == 0
        result = phi / (total + 1e-12)
        result[zero_mask] = 0.0
        return result

    N = phi_bar.shape[0]
    B = n_bins
    device = phi_bar.device
    dtype = phi_bar.dtype

    dt = int(dead_time_bins) % B

    phi_ext = torch.cat([phi, phi[:, :-1]], dim=-1)
    cum = torch.zeros(N, 2 * B, device=device, dtype=dtype)
    cum[:, 1:] = torch.cumsum(phi_ext, dim=-1)

    k_idx = torch.arange(B, device=device)
    i_idx = torch.arange(B, device=device)

    s = (i_idx.unsqueeze(0) - k_idx.unsqueeze(1)) % B
    k_exp = k_idx.unsqueeze(1).expand(B, B)
    end_exp = k_exp + s

    cum_k = cum[:, k_exp]
    cum_end = cum[:, end_exp]
    cyclic_sum = cum_end - cum_k

    p_i = (1.0 - torch.exp(-phi)).unsqueeze(1)
    denom = (1.0 - torch.exp(-total.squeeze(-1))).view(N, 1, 1) + 1e-15
    M = p_i * torch.exp(-cyclic_sum) / denom
    M = M / (M.sum(dim=-1, keepdim=True) + 1e-15)

    j_idx = torch.arange(B, device=device)
    rearm_k = (j_idx + dt) % B
    T = M[:, rearm_k, :]

    v = torch.ones(N, 1, B, device=device, dtype=dtype) / B
    for _ in range(n_iterations):
        v = torch.bmm(v, T)
        v = v / (v.sum(dim=-1, keepdim=True) + 1e-15)

    distorted = v.squeeze(1)
    zero_mask = total.squeeze(-1) == 0
    distorted[zero_mask] = 0.0

    return distorted


def calculate_distorted_transient(
    phi_bar: torch.Tensor,
    dead_time_bins: int,
    n_bins: int,
) -> torch.Tensor:
    is_1d = phi_bar.ndim == 1
    phi = phi_bar.unsqueeze(0) if is_1d else phi_bar
    out = batch_distorted_transient_async(phi, dead_time_bins, n_bins)
    return out.squeeze(0) if is_1d else out


def calculate_distorted_transient_sync(
    phi_bar: torch.Tensor,
    dead_time_bins: int,
    n_bins: int,
) -> torch.Tensor:
    is_1d = phi_bar.ndim == 1
    phi = phi_bar.unsqueeze(0) if is_1d else phi_bar
    out = batch_distorted_transient_sync(phi, dead_time_bins, n_bins)
    return out.squeeze(0) if is_1d else out


# =============================================================================
# HISTOGRAMMER BASE CLASS
# =============================================================================


class HistogrammerBase:
    """Base class for SPAD readout and TDC histogramming operations."""

    def __init__(self):
        pass

    def calculate_transients(
        self,
        irradiance_frames: torch.Tensor,
        depth_frames: torch.Tensor,
        offsets: torch.Tensor,
        fov_masks: torch.Tensor,
        gt_ntime_bins: int,
        max_depth: float,
        sensor_fov: list | None = None,
        pixel_fov_list: list | None = None,
        w: int | None = None,
        h: int | None = None,
        omega: float | None = None,
    ) -> tuple[torch.Tensor, list]:
        if hasattr(irradiance_frames, "magnitude"):
            irradiance_frames = irradiance_frames.magnitude
        if hasattr(depth_frames, "magnitude"):
            depth_frames = depth_frames.magnitude
        if hasattr(offsets, "magnitude"):
            offsets = offsets.magnitude

        device = irradiance_frames.device
        if fov_masks.device != device:
            fov_masks = fov_masks.to(device)
        if depth_frames.device != device:
            depth_frames = depth_frames.to(device)
        if offsets.device != device:
            offsets = offsets.to(device)

        transient_list = []
        ambient_offsets = []

        if len(irradiance_frames.shape) == 3:
            for frame_idx in range(irradiance_frames.shape[0]):
                with ElapsedProgress(disable=True) as progress:
                    task = progress.add_task("Processing FOV masks", total=len(fov_masks))
                    for mask_idx, fov_mask in enumerate(fov_masks):
                        index_mask = fov_mask > 0
                        current_irradiance_vals = irradiance_frames[frame_idx][index_mask]

                        n_fov_pixels = int(index_mask.sum())
                        if n_fov_pixels == 0:
                            transient_list.append(
                                torch.zeros((1, gt_ntime_bins), dtype=irradiance_frames.dtype, device=device)
                            )
                            ambient_offsets.append(torch.zeros((), device=device))
                            continue

                        if (
                            sensor_fov is not None
                            and pixel_fov_list is not None
                            and w is not None
                            and h is not None
                            and omega is not None
                        ):
                            fov_irradiance_vals = get_irradiance_with_fov(
                                current_irradiance_vals, sensor_fov, pixel_fov_list[mask_idx], omega, w, h
                            )
                        else:
                            fov_irradiance_vals = current_irradiance_vals

                        if hasattr(fov_irradiance_vals, "magnitude"):
                            fov_irradiance_vals = fov_irradiance_vals.magnitude

                        current_depth_vals = depth_frames[frame_idx][index_mask]
                        masked_offsets = offsets[frame_idx][index_mask]

                        fov_irradiance_vals = fov_irradiance_vals * fov_mask[index_mask]

                        valid = torch.isfinite(current_depth_vals) & (current_depth_vals > 0)
                        current_depth_vals = current_depth_vals[valid]
                        fov_irradiance_vals = fov_irradiance_vals[valid]

                        ambient_offsets.append(masked_offsets.sum() / (n_fov_pixels * gt_ntime_bins))

                        if current_depth_vals.numel() == 0:
                            transient_list.append(
                                torch.zeros((1, gt_ntime_bins), dtype=irradiance_frames.dtype, device=device)
                            )
                            continue

                        transient_idx = torch.floor(current_depth_vals * gt_ntime_bins / max_depth).to(torch.long)
                        transient_idx = torch.remainder(transient_idx, gt_ntime_bins)

                        row = torch.zeros(gt_ntime_bins, dtype=fov_irradiance_vals.dtype, device=device)
                        row = row.scatter_add(0, transient_idx, fov_irradiance_vals / n_fov_pixels)
                        transient_list.append(row.unsqueeze(0))
                        progress.advance(task)
            return torch.concat(transient_list), ambient_offsets
        else:
            with ElapsedProgress(disable=True) as progress:
                task = progress.add_task("Processing FOV masks", total=len(fov_masks))
                for mask_idx, fov_mask in enumerate(fov_masks):
                    index_mask = fov_mask > 0
                    current_irradiance_vals = irradiance_frames[index_mask]

                    n_fov_pixels = int(index_mask.sum())
                    if n_fov_pixels == 0:
                        transient_list.append(
                            torch.zeros((1, gt_ntime_bins), dtype=irradiance_frames.dtype, device=device)
                        )
                        ambient_offsets.append(torch.zeros((), device=device))
                        continue

                    if (
                        sensor_fov is not None
                        and pixel_fov_list is not None
                        and w is not None
                        and h is not None
                        and omega is not None
                    ):
                        fov_irradiance_vals = get_irradiance_with_fov(
                            current_irradiance_vals, sensor_fov, pixel_fov_list[mask_idx], omega, w, h
                        )
                    else:
                        fov_irradiance_vals = current_irradiance_vals

                    if hasattr(fov_irradiance_vals, "magnitude"):
                        fov_irradiance_vals = fov_irradiance_vals.magnitude

                    current_depth_vals = depth_frames[index_mask]
                    masked_offsets = offsets[index_mask]

                    fov_irradiance_vals = fov_irradiance_vals * fov_mask[index_mask]

                    valid = torch.isfinite(current_depth_vals) & (current_depth_vals > 0)
                    current_depth_vals = current_depth_vals[valid]
                    fov_irradiance_vals = fov_irradiance_vals[valid]

                    ambient_offsets.append(masked_offsets.sum() / (n_fov_pixels * gt_ntime_bins))

                    if current_depth_vals.numel() == 0:
                        transient_list.append(
                            torch.zeros((1, gt_ntime_bins), dtype=irradiance_frames.dtype, device=device)
                        )
                        continue

                    transient_idx = torch.floor(current_depth_vals * gt_ntime_bins / max_depth).to(torch.long)
                    transient_idx = torch.remainder(transient_idx, gt_ntime_bins)

                    row = torch.zeros(gt_ntime_bins, dtype=fov_irradiance_vals.dtype, device=device)
                    row = row.scatter_add(0, transient_idx, fov_irradiance_vals / n_fov_pixels)
                    transient_list.append(row.unsqueeze(0))
                    progress.advance(task)
            return torch.concat(transient_list), ambient_offsets

    def calculate_arrival_rates(
        self, irf: torch.Tensor, transients: torch.Tensor, offset, gt_ntime_bins: int
    ) -> torch.Tensor:
        arrival_rates = torch.zeros_like(transients, dtype=torch.float32, device=transients.device)
        for i in range(transients.shape[0]):
            convolved_signal = F.conv1d(transients[i].view(1, 1, -1), irf.flip(-1).view(1, 1, -1), padding="same").view(
                -1
            )

            if isinstance(offset, (list, tuple)):
                background = offset[i] if i < len(offset) else (offset[0] if len(offset) > 0 else 0.0)
            elif isinstance(offset, torch.Tensor):
                background = offset[i] if offset.dim() > 0 and i < offset.shape[0] else offset
            else:
                background = offset

            if not isinstance(background, torch.Tensor):
                background = torch.tensor(background, dtype=torch.float32, device=transients.device)

            background_extended = background.expand_as(convolved_signal) if background.dim() == 0 else background
            arrival_rates[i, :] = convolved_signal + background_extended
        return arrival_rates

    def simulate_pixel_photon_hist(
        self,
        phi_bar: torch.Tensor,
        n_pulses: int,
        n_bins: int,
        free_running: bool,
        dead_time_bins: int,
        paralyzable: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        n_tbins = phi_bar.shape[-1]
        timestamps = simulate_photon_timestamps(
            phi_bar,
            n_pulses,
            dead_time_bins=int(dead_time_bins),
            free_running=bool(free_running),
            paralyzable=bool(paralyzable),
            max_detections_per_cycle=None,
            generator=generator,
        )
        photon_hist = timestamps_to_histogram(timestamps, n_tbins, n_bins)
        return photon_hist.to(phi_bar.device)

    def simulate_pixel_ewh(
        self,
        phi_bar: torch.Tensor,
        n_pulses: int,
        n_bins: int,
        free_running: bool,
        dead_time_bins: int,
        paralyzable: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.simulate_pixel_photon_hist(
            phi_bar,
            n_pulses,
            n_bins,
            free_running,
            dead_time_bins,
            paralyzable=paralyzable,
            generator=generator,
        )



