from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Union

import torch
import torch.nn.functional as F
from pint import Quantity
from torch import Tensor
from tqdm import tqdm

from visionsim.emulate.aspc.detector import (
    simulate_photon_timestamps,
    timestamps_to_histogram,
)
from visionsim.emulate.aspc.units import validate_units
from visionsim.emulate.aspc.utils import get_irradiance_with_fov, ureg


def calculate_distorted_transient(phi_bar, dead_time_bins, n_hist_bins):
    r"""Free-running (asynchronous) single-pixel pile-up forward model.

    Exact discrete-time renewal model. After a detection in bin ``i`` the detector
    re-arms **at** bin ``i + dead_time_bins`` (so bins ``i+1 … i+dt-1`` are blocked),
    and the next detection is the first armed bin that receives a photon. The
    transition kernel is therefore

    .. math::
        T[i, j] \;\propto\; p_j \prod_{m=r}^{j-1} (1 - p_m),
        \qquad p_k = 1 - e^{-\varphi_k},\; r = i + dt

    walking forward from ``r`` with wraparound. The measured histogram shape is the
    stationary distribution of ``T``.

    Two things this gets right that the previous formulation did not (verified
    against the Monte-Carlo reference in ``detector.py`` to within binomial 3-sigma
    on flat, ramped, random and spike transients):

    * The survival product runs over ``[r, j)`` and **excludes bin j itself** — you
      do not have to survive a bin in order to be detected in it. The old inclusive
      ``cumsum`` slice required exactly that, which cancels out for flat phi (a
      constant factor absorbed by row normalisation) but biases every non-flat
      transient. That was the residual behind finding **M1**.
    * ``dead_time_bins`` counts *bins of dead time*, matching
      ``camera.py``'s ``int(dead_time_s * n_bins * frequency)``. The old code
      re-armed one bin late.

    Dead times longer than one cycle are handled by walking the full cycle from
    ``r``; there is no ``% n_hist_bins`` wrap that could silently reduce the dead
    time to zero (finding **M3**).

    Args:
        phi_bar: Per-bin photon arrival rates for one pixel, shape ``(n_tbins,)``.
        dead_time_bins: Dead time in whole bins. ``0`` means every arriving photon
            is detected, so the shape is simply ``phi / sum(phi)``.
        n_hist_bins: Number of bins per cycle.

    Returns:
        Normalized distorted transient, shape ``(n_hist_bins,)``, summing to 1.

    Citation reference (original formulation):
    J. Rapp, Y. Ma, R. M. A. Dawson and V. K. Goyal, "Dead Time Compensation for
    High-Flux Ranging," IEEE TSP, vol. 67, no. 13, pp. 3471-3486, 2019.
    """
    n_bins = int(n_hist_bins)
    out_dtype = phi_bar.dtype
    phi = torch.clamp(phi_bar[..., :n_bins].to(torch.float64), min=0.0)
    dead_time_bins = int(dead_time_bins)

    total = phi.sum()
    if total <= 0:
        return phi.to(out_dtype)
    if dead_time_bins == 0:
        # No dead time: every arriving photon is detected, so the expected count
        # per bin is phi itself. Sampling occupancy (1 - e^-phi) here instead is
        # the dt=0 finding.
        return (phi / total).to(out_dtype)

    p_detect = 1.0 - torch.exp(-phi)
    order = torch.arange(n_bins, device=phi.device)
    transition = torch.zeros((n_bins, n_bins), dtype=torch.float64, device=phi.device)

    for bin_idx in range(n_bins):
        # Candidate bins in the order the detector visits them after re-arming.
        visit = (bin_idx + dead_time_bins + order) % n_bins
        run = phi[visit]
        # Exclusive prefix sum => survive [r, j), not [r, j].
        survived = torch.cat([torch.zeros(1, dtype=torch.float64, device=phi.device),
                              torch.cumsum(run, dim=0)[:-1]])
        row = p_detect[visit] * torch.exp(-survived)
        transition[bin_idx, visit] = row / (row.sum() + 1e-300)

    eigenvalues, eigenvectors = torch.linalg.eig(transition.T)
    principal = eigenvectors[:, torch.argmax(torch.abs(eigenvalues))].real
    if principal[0] < 0:
        principal = -principal
    return (principal / principal.sum()).to(out_dtype)


def calculate_distorted_transient_sync(
    phi_bar: torch.Tensor,
    dead_time_bins: int,
    n_hist_bins: int,
) -> torch.Tensor:
    """Synchronous (gated) single-pixel pile-up forward model.

    The detector is re-armed at the start of every cycle, so dead time never
    crosses a cycle boundary and there is no carry-in state. Exact discrete-time
    recursion on ``F[j] = P(a detection occurs in bin j)``:

    * a *first* detection at ``j`` contributes ``p_j * prod_{m<j} (1 - p_m)``;
    * a later one contributes ``F[i] * p_j * prod_{m=i+dt}^{j-1} (1 - p_m)`` for
      every earlier detection bin ``i`` with ``i + dt <= j``.

    **Single-hit** (one detection per cycle — the conventional lidar setup that
    Coates inverts) is ``dead_time_bins >= n_hist_bins``, and reduces exactly to
    first-photon-wins ``p_j * prod_{m<j}(1 - p_m)``. Note that under a one-per-cycle
    cap the dead time cannot bind at all, so the single-hit answer is *independent*
    of its value — verified against the Monte-Carlo reference across
    ``dt in {0, 1, 3, 8, n_hist_bins}``.

    The previous implementation used the already-gated ``prob_detect`` inside the
    survival product instead of the raw per-bin ``p_detect``, double-discounting
    liveness; in single-hit mode it missed the Coates result by ~19 sigma
    (finding **H1**).

    Args:
        phi_bar: Per-bin photon arrival rates for one pixel, shape ``(n_tbins,)``.
        dead_time_bins: Dead time in whole bins; ``>= n_hist_bins`` selects
            single-hit. ``0`` means every arriving photon is detected.
        n_hist_bins: Number of bins per cycle.

    Returns:
        Normalized distorted transient, shape ``(n_hist_bins,)``, summing to 1.
    """
    n_bins = int(n_hist_bins)
    out_dtype = phi_bar.dtype
    phi = torch.clamp(phi_bar[..., :n_bins].to(torch.float64), min=0.0)
    dead_time_bins = int(dead_time_bins)

    total = phi.sum()
    if total <= 0:
        return phi.to(out_dtype)
    if dead_time_bins == 0:
        # As in the free-running model: with no dead time every photon is detected.
        return (phi / total).to(out_dtype)

    p_detect = 1.0 - torch.exp(-phi)
    # log-domain survival so that long runs of bright bins do not underflow
    log_survive = torch.log(torch.clamp(1.0 - p_detect, min=1e-300))
    cum_log = torch.cat([torch.zeros(1, dtype=torch.float64, device=phi.device),
                         torch.cumsum(log_survive, dim=0)])  # cum_log[k] = sum_{m<k}

    first_detection = p_detect * torch.exp(cum_log[:n_bins])

    if dead_time_bins >= n_bins:
        prob_detect = first_detection  # single-hit: first photon wins
    else:
        # Accumulate into a list and stack, rather than writing in place into a
        # preallocated tensor: the recursion reads earlier entries of the same
        # buffer it writes to, which bumps the autograd version counter and makes
        # backward() fail with "a variable needed for gradient computation has
        # been modified by an inplace operation". Building it functionally keeps
        # the multi-hit branch differentiable like the rest of the module.
        detections: list[torch.Tensor] = []
        for j in range(n_bins):
            acc = first_detection[j]
            for i in range(0, j - dead_time_bins + 1):
                acc = acc + detections[i] * p_detect[j] * torch.exp(cum_log[j] - cum_log[i + dead_time_bins])
            detections.append(acc)
        prob_detect = torch.stack(detections)

    return (prob_detect / prob_detect.sum()).to(out_dtype)


def batch_distorted_transient_sync(
    phi_bar: torch.Tensor,
    dead_time_bins: int,  # kept for API compatibility, ignored
    n_hist_bins: int,
) -> torch.Tensor:
    """
    Batched single-hit synchronous forward model (Coates).
    First photon wins — rest of cycle is dead.
    """
    p_detect = 1.0 - torch.exp(-torch.clamp(phi_bar[:, :n_hist_bins], min=0.0))

    survival = torch.ones_like(p_detect)
    survival[:, 1:] = torch.cumprod(1.0 - p_detect[:, :-1], dim=-1)

    prob_detect = p_detect * survival

    total = prob_detect.sum(dim=-1, keepdim=True)
    distorted = prob_detect / (total + 1e-12)

    return distorted

def batch_distorted_transient_async(
    phi_bar: torch.Tensor,
    dead_time_bins: int,
    n_hist_bins: int,
    n_iterations: int = 100,
) -> torch.Tensor:
    """
    Batched asynchronous (free-running) forward model using power iteration.

    Replaces the per-pixel eigendecomposition with batched power iteration
    to find the stationary distribution of the Markov chain.

    Args:
        phi_bar: shape (N, n_hist_bins) — arrival rates for N pixels.
        dead_time_bins: dead time in bins (wraps around cycle boundary).
        n_hist_bins: number of bins per cycle.
        n_iterations: number of power iteration steps.

    Returns:
        shape (N, n_hist_bins) — normalized distorted transients.
    """
    # dead_time=0 → every photon is detected; no pile-up distortion.
    # Consistent with the slow-path simulate_pixel_ewh and simulate_ewh_diff.
    if dead_time_bins == 0:
        phi = torch.clamp(phi_bar[:, :n_hist_bins], min=0.0)
        total = phi.sum(dim=-1, keepdim=True)
        zero_mask = total.squeeze(-1) == 0
        result = phi / (total + 1e-12)
        result[zero_mask] = 0.0
        return result

    N = phi_bar.shape[0]
    device = phi_bar.device
    dtype = phi_bar.dtype

    dt = int(dead_time_bins) % int(n_hist_bins)

    # (N, B)
    phi = torch.clamp(phi_bar[:, :n_hist_bins], min=0.0)
    cum_phi = torch.cumsum(phi, dim=-1)                # (N, B)
    total = phi.sum(dim=-1, keepdim=True)               # (N, 1)

    # Handle zero-flux pixels
    zero_mask = (total.squeeze(-1) == 0)
    if zero_mask.all():
        return phi

    # Build transition matrices: T[n, i, j] for all pixels
    # T[i, j] = phi[j] * exp(-survival_exponent[i,j]) / (1 - exp(-total))
    # survival_exponent depends on whether i + dt wraps around

    # Precompute denominator: 1 - exp(-total), shape (N, 1, 1)
    denom = (1.0 - torch.exp(-total)).unsqueeze(-1)     # (N, 1, 1)

    # Build survival exponents for all (i, j) pairs
    # indices: i = row (detection bin), j = column (next detection bin)
    i_idx = torch.arange(n_hist_bins, device=device)     # (B,)
    j_idx = torch.arange(n_hist_bins, device=device)     # (B,)

    # Wrap point for each row: (i + dt) mod B
    wrap_point = (i_idx + dt) % n_hist_bins               # (B,)

    # For each row i, the survival exponent at column j is:
    #   cum_phi[j] + total - cum_phi[wrap_point[i]]    (base)
    #   then subtract total for bins after wrap_point[i] (if no wrap)
    #   or before wrap_point[i] (if wrap)

    # cum_phi at wrap points: (N, B) -> gather -> (N, B, 1)
    cum_at_wrap = cum_phi[:, wrap_point].unsqueeze(-1)    # (N, B, 1)

    # cum_phi broadcast: (N, 1, B)
    cum_j = cum_phi.unsqueeze(1)                          # (N, 1, B)

    # Base survival exponent: (N, B, B)
    surv_exp = cum_j + total.unsqueeze(-1) - cum_at_wrap

    # Correction: subtract total for bins past the wrap point
    # For row i, bins j where j > wrap_point[i] (no-wrap case: i+dt < B)
    # or j > wrap_point[i] (wrap case: i+dt >= B, subtract for j > wrap)
    wrap_expanded = wrap_point.unsqueeze(-1)               # (B, 1)
    j_expanded = j_idx.unsqueeze(0)                        # (1, B)

    if dt > 0:
        # Mask where we need to subtract total
        # When i + dt < n_hist_bins: subtract for j in (wrap_point[i], ...]
        # When i + dt >= n_hist_bins: wrap_point < i, subtract for j > wrap AND j <= i
        no_wrap_mask = (i_idx + dt) < n_hist_bins          # (B,)

        # For no-wrap rows: subtract total where j > wrap_point[i]
        subtract_mask_no_wrap = (j_expanded > wrap_expanded) & no_wrap_mask.unsqueeze(-1)

        # For wrap rows: subtract total where j > wrap_point[i] AND we haven't wrapped past
        subtract_mask_wrap = (j_expanded > wrap_expanded) & (~no_wrap_mask.unsqueeze(-1))

        subtract_mask = subtract_mask_no_wrap | subtract_mask_wrap  # (B, B)
        surv_exp = surv_exp - total.unsqueeze(-1) * subtract_mask.unsqueeze(0).float()

    # Transition probs: phi[j] * exp(-surv_exp) / denom
    # phi broadcast: (N, 1, B)
    phi_j = phi.unsqueeze(1)                               # (N, 1, B)
    T = phi_j * torch.exp(-surv_exp) / (denom + 1e-15)    # (N, B, B)

    # Normalize rows to sum to 1 (proper stochastic matrix)
    T = T / (T.sum(dim=-1, keepdim=True) + 1e-15)

    # Power iteration to find stationary distribution
    # v = v @ T repeatedly; start with uniform
    v = torch.ones(N, 1, n_hist_bins, device=device, dtype=dtype) / n_hist_bins

    for _ in range(n_iterations):
        v = torch.bmm(v, T)                               # (N, 1, B)
        v = v / (v.sum(dim=-1, keepdim=True) + 1e-15)

    distorted = v.squeeze(1)                               # (N, B)

    # Zero-flux pixels get uniform or zero
    distorted[zero_mask] = 0.0

    return distorted

class HistogrammerBase:
    """
    Base class for histogramming operations in SPAD sensors.
    Handles transient calculation, arrival rate computation, and histogram simulation.
    """

    def __init__(self):
        """Initialize the histogrammer base class."""
        pass

    def get_pixel_fov_mask(
        self, empty_mask: Tensor, row1: float, row2: float, col1: float, col2: float, vignette: bool = True
    ) -> Tensor:
        """
        Generates a rectangular FOV mask with optional smooth vignette (weights in [0, 1]).
        If vignette=True: 1.0 at the rectangle center, smoothly decreasing to 0.0 at the rectangle edges.
        If vignette=False: 1.0 inside the rectangle, 0.0 outside.
        Values represent how unmasked a pixel is.

        Args:
            empty_mask (Tensor): Empty array used to define output shape (H, W). Values are ignored.
            row1 (float): Normalized start row (0.0 to 1.0).
            row2 (float): Normalized end row (0.0 to 1.0).
            col1 (float): Normalized start column (0.0 to 1.0).
            col2 (float): Normalized end column (0.0 to 1.0).
            vignette (bool): If True, apply smooth vignette; if False, use rectangular FOV. Default is True.

        Returns:
            Tensor: A float mask in [0, 1] with optional vignette inside the specified rectangle.
        """
        img_rows, img_cols = empty_mask.shape

        # Ensure bounds are in proper order and clipped
        r1 = max(0, min(img_rows, int(round(row1 * img_rows))))
        r2 = max(0, min(img_rows, int(round(row2 * img_rows))))
        c1 = max(0, min(img_cols, int(round(col1 * img_cols))))
        c2 = max(0, min(img_cols, int(round(col2 * img_cols))))

        mask = torch.zeros((img_rows, img_cols), dtype=torch.float32, device=empty_mask.device)

        if r1 == r2 or c1 == c2:
            # Degenerate rectangle -> return zeros
            return mask

        if vignette:
            # Build coordinate grid for the rectangle region
            rr = torch.arange(r1, r2, device=empty_mask.device)
            cc = torch.arange(c1, c2, device=empty_mask.device)
            yy, xx = torch.meshgrid(rr, cc, indexing="ij")

            # Center and half-sizes
            cy = (r1 + r2 - 1) / 2.0
            cx = (c1 + c2 - 1) / 2.0
            half_h = max((r2 - r1) / 2.0, 1e-6)
            half_w = max((c2 - c1) / 2.0, 1e-6)

            # Normalized radial distance from center within the rectangle
            dy = (yy - cy) / half_h
            dx = (xx - cx) / half_w
            radial = torch.sqrt(dx * dx + dy * dy)

            # Vignette profile: 1 at center, 0 at edges (clamped)
            # Use a smooth falloff; adjust exponent for steeper/softer edges
            exponent = 2.0
            vignette_weights = torch.clamp(1.0 - torch.pow(radial, exponent), 0.0, 1.0)

            mask[r1:r2, c1:c2] = vignette_weights
        else:
            # Rectangular FOV: set all pixels inside rectangle to 1.0
            mask[r1:r2, c1:c2] = 1.0

        return mask

    def get_perpixel_fov_masks(
        self, empty_mask: Tensor, pixel_fov_list: list, device: torch.device = torch.device("cpu"), vignette: bool = True
    ) -> Tensor:
        """
        Generates a list of FOV masks based on `pixel_fov_list`.

        Args:
            empty_mask (Tensor): An array to define the shape of the masks.
            pixel_fov_list (list): List of FOV coordinates [row1, row2, col1, col2].
            device (torch.device): The device to load tensors onto ('cpu' or 'cuda').

        Returns:
            torch.Tensor: A tensor of float masks with values in [0, 1].
        """
        # Ensure empty_mask is on the correct device
        if empty_mask.device != device:
            empty_mask = empty_mask.to(device)

        mask_list = []
        for r1, r2, c1, c2 in pixel_fov_list:
            mask = self.get_pixel_fov_mask(empty_mask, r1, r2, c1, c2, vignette=vignette)
            # Ensure mask is on the correct device
            if mask.device != device:
                mask = mask.to(device)
            mask_list.append(mask.unsqueeze(0))

        return torch.concat(mask_list)

    def calculate_transients(
        self,
        irradiance_frames: torch.Tensor,
        depth_frames: torch.Tensor,
        offsets: torch.Tensor,
        fov_masks: torch.Tensor,
        gt_ntime_bins: int,
        max_depth: float,
        sensor_fov: list = None,
        pixel_fov_list: list = None,
        w: int = None,
        h: int = None,
        omega: float = None,
    ) -> tuple[torch.Tensor, list]:
        """
        Calculates the transient signal for each defined pixel FOV.

        Args:
            irradiance_frames (torch.Tensor): Tensor of irradiance images.
            depth_frames (torch.Tensor): Tensor of depth images.
            offsets (torch.Tensor): Tensor of offset values.
            fov_masks (torch.Tensor): Tensor of FOV masks.
            gt_ntime_bins (int): Total number of time bins for the transient.
            max_depth (float): Maximum depth corresponding to the last time bin.
            sensor_fov (list): Sensor field of view [fov_x, fov_y]. Optional.
            pixel_fov_list (list): List of pixel FOV coordinates. Optional.
            w (int): Image width. Optional.
            h (int): Image height. Optional.
            omega (float): Solid angle per pixel. Optional.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: A tensor containing the calculated transients.
                - list: List of ambient offsets.
        """
        # Ensure all tensors are on the same device
        device = irradiance_frames.device
        if fov_masks.device != device:
            fov_masks = fov_masks.to(device)
        if depth_frames.device != device:
            depth_frames = depth_frames.to(device)
        if offsets.device != device:
            offsets = offsets.to(device)

        transient_list = []
        ambient_offsets = []
        for frame_idx in range(irradiance_frames.shape[0]):
            for mask_idx, fov_mask in enumerate(tqdm(fov_masks, desc="Processing FOV masks", disable=True)):
                index_mask = fov_mask > 0
                current_irradiance_vals = irradiance_frames[frame_idx][index_mask]

                # Number of render pixels this FOV covers. The SPAD sees the *average*
                # radiance over its FOV, and each render pixel subtends 1/n_fov_pixels of
                # it, so contributions are weighted by 1/n_fov_pixels rather than summed
                # raw. Without this the collected photon count scales with render
                # resolution, coupling the render grid to the physical sensor (F3).
                n_fov_pixels = int(index_mask.sum())
                if n_fov_pixels == 0:
                    transient_list.append(
                        torch.zeros(
                            (1, gt_ntime_bins), dtype=irradiance_frames.dtype, device=irradiance_frames.device
                        )
                    )
                    ambient_offsets.append(torch.zeros((), device=irradiance_frames.device))
                    continue

                # Apply FOV correction if parameters are provided. NOTE: this is a
                # wide-angle arcsin correction (factor ~1.00-1.08), NOT a per-FOV
                # normalisation -- that is handled by n_fov_pixels above.
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

                current_depth_vals = depth_frames[frame_idx][index_mask]
                masked_offsets = offsets[frame_idx][index_mask]

                # Extract magnitude only if these are Pint Quantity objects, otherwise use as-is
                if hasattr(current_depth_vals, "magnitude"):
                    current_depth_vals = current_depth_vals.magnitude
                if hasattr(fov_irradiance_vals, "magnitude"):
                    fov_irradiance_vals = fov_irradiance_vals.magnitude
                if hasattr(masked_offsets, "magnitude"):
                    masked_offsets = masked_offsets.magnitude

                # Apply vignette weights (values in [0, 1]) to the irradiance that is
                # actually binned. Vignetting attenuates real signal, so it legitimately
                # reduces the total -- it is not a normalisation.
                fov_irradiance_vals = fov_irradiance_vals * fov_mask[index_mask]

                # Reject physically invalid depths (non-finite, or <= 0 meaning "no
                # surface"). Out-of-range depths are NOT invalid -- see aliasing below.
                valid = torch.isfinite(current_depth_vals) & (current_depth_vals > 0)
                current_depth_vals = current_depth_vals[valid]
                fov_irradiance_vals = fov_irradiance_vals[valid]

                ambient_offsets.append(masked_offsets.sum() / (n_fov_pixels * gt_ntime_bins))

                if current_depth_vals.numel() == 0:
                    transient_list.append(
                        torch.zeros(
                            (1, gt_ntime_bins), dtype=irradiance_frames.dtype, device=irradiance_frames.device
                        )
                    )
                    continue

                # Convert depth values to time bin locations. Returns beyond max_depth
                # arrive during a later laser cycle, so they *alias* back into the window
                # at (2d/c) mod (1/f) rather than being clamped into the last bin. This
                # is a property of the arrival process and is therefore independent of
                # gated vs free-running operation.
                transient_idx = torch.floor(current_depth_vals * gt_ntime_bins / max_depth).to(torch.long)
                transient_idx = torch.remainder(transient_idx, gt_ntime_bins)

                row = torch.zeros(
                    gt_ntime_bins, dtype=fov_irradiance_vals.dtype, device=fov_irradiance_vals.device
                )
                row = row.scatter_add(0, transient_idx, fov_irradiance_vals / n_fov_pixels)
                transient_list.append(row.unsqueeze(0))

        final_transients = torch.concat(transient_list)
        return final_transients, ambient_offsets

    def calculate_arrival_rates(
        self, irf: torch.Tensor, transients: torch.Tensor, offset, gt_ntime_bins: int
    ) -> torch.Tensor:
        """
        Calculates the photon arrival rates by convolving transients with the IRF
        and adding background noise.

        Args:
            irf (torch.Tensor): The instrumental response function.
            transients (torch.Tensor): The calculated transients.
            offset: Background offset values (can be a list, tensor, or scalar).
            gt_ntime_bins (int): Total number of time bins.

        Returns:
            torch.Tensor: A tensor of photon arrival rates.
        """
        arrival_rates = torch.zeros_like(transients, dtype=torch.float32, device=transients.device)
        for i in tqdm(range(transients.shape[0]), desc="Convolving transients", disable=True):
            # Reshape for conv1d: (batch_size, in_channels, signal_length)
            # F.conv1d implements cross-correlation, not convolution: it does NOT flip
            # the kernel. Flipping here makes this a true convolution, which matters for
            # asymmetric pulse shapes (a custom IRF was otherwise time-reversed, biasing
            # the recovered depth). Symmetric kernels are unaffected.
            convolved_signal = F.conv1d(
                transients[i].view(1, 1, -1), irf.flip(-1).view(1, 1, -1), padding="same"
            ).view(-1)
            # Add signal and background components
            # Handle offset as list, tensor, or scalar
            if isinstance(offset, (list, tuple)):
                background = offset[i] if i < len(offset) else offset[0] if len(offset) > 0 else 0.0
            elif isinstance(offset, torch.Tensor):
                background = offset[i] if offset.dim() > 0 and i < offset.shape[0] else offset
            else:
                background = offset

            # Convert to tensor if needed
            if not isinstance(background, torch.Tensor):
                background = torch.tensor(background, dtype=torch.float32, device=transients.device)

            background_extended = background.expand_as(convolved_signal) if background.dim() == 0 else background
            arrival_rates[i, :] = convolved_signal + background_extended
        return arrival_rates

    def simulate_pixel_photon_hist(
        self,
        phi_bar: torch.Tensor,
        n_pulses: int,
        n_hist_bins: int,
        free_running: bool,
        dead_time_bins: int,
        paralyzable: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Sample a photon-detection histogram for a single pixel.

        Thin wrapper over the ground-truth timestamp simulator: detections are
        sampled as ``(cycle, bin)`` timestamps and then reduced to a histogram,
        which is also the order real hardware works in -- a TDC emits
        timestamps, binning happens afterwards.

        Shared by the EWH and EDH histogrammers, which differ only in what they
        do with the resulting histogram.

        Gated mode is multi-hit here (no per-cycle cap). For conventional
        single-hit lidar call
        :func:`~visionsim.emulate.aspc.detector.simulate_photon_timestamps`
        directly with ``max_detections_per_cycle=1``.

        Args:
            phi_bar (torch.Tensor): Expected photon arrival rates for one pixel across time bins.
            n_pulses (int): Number of laser pulses to simulate.
            n_hist_bins (int): Histogram resolution. May be coarser than the
                number of time bins in ``phi_bar``, which must then be an exact
                multiple of it.
            free_running (bool): True for free-running mode, False for gated mode.
            dead_time_bins (int): Dead time in whole time bins. The detector
                re-arms *at* ``t_detect + dead_time_bins``, so a detection blocks
                the ``dead_time_bins - 1`` bins that follow it.
            paralyzable (bool): If True, every arrival re-opens the dead window
                whether or not it was detected (passive quenching).
            generator (torch.Generator | None): Optional RNG for reproducibility.

        Returns:
            torch.Tensor: Accumulated photon histogram, shape ``(n_hist_bins,)``.
        """
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
        photon_hist = timestamps_to_histogram(timestamps, n_tbins, n_hist_bins)
        return photon_hist.to(phi_bar.device)

@dataclass
class HistConfig:
    type: Union[str, None] = None
    min_depth: Quantity = 0 * ureg.meter
    max_depth: Quantity = 20 * ureg.meter
    n_bins: int = 1000
    n_hist_bins: Union[int, None] = None
    bin_width: Quantity = 0.03 * ureg.meter
    pixel_fov_list: List[List[float]] = field(
        default_factory=lambda: [
            [0, 0.4, 0.3, 0.6],
            [0.7, 0.95, 0.6, 0.9],
            # Full scene. Must be exactly 1.0, not 0.999: bounds are rounded to integer
            # pixels, so 0.999 is resolution-dependent (exact at 300x400, but drops the
            # last row and column at 1000x1000).
            [0, 1.0, 0, 1.0],
        ]
    )
    vignette: bool = True
    n_pulses: int = 100
    dead_time_s: Quantity = 10e-9 * ureg.second
    free_running: bool = True
    fast_sim: bool = True

    def validate(self):
        if self.min_depth < 0 * ureg.meter:
            raise ValueError("min_depth must be >= 0m")
        if self.max_depth <= self.min_depth:
            raise ValueError("max_depth must be > min_depth")
        if self.n_bins <= 0:
            raise ValueError("n_bins must be > 0")
        if self.n_hist_bins is not None and self.n_hist_bins <= 0:
            raise ValueError("n_hist_bins must be > 0")
        if self.bin_width <= 0:
            raise ValueError("bin_width must be > 0")
        if self.n_pulses <= 0:
            raise ValueError("n_pulses must be > 0")
        if self.dead_time_s <= 0:
            raise ValueError("dead_time_s must be > 0")


class Histogrammer(HistogrammerBase):
    """
    Equi-Width Histogrammer class
    """

    @validate_units()
    def __init__(self, config: HistConfig = HistConfig()):
        super().__init__()
        # config.validate()
        for k, v in config.__dict__.items():
            setattr(self, k, v)

    def simulate_pixel_ewh(
        self,
        phi_bar: torch.Tensor,
        n_pulses: int,
        n_hist_bins: int,
        free_running: bool,
        dead_time_bins: int,
        fast_sim: bool = True,
        paralyzable: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Simulates the Equi-Width Histogram (EWH) for a single pixel.

        The EWH *is* the sampled photon histogram, so this delegates straight to
        :meth:`HistogrammerBase.simulate_pixel_photon_hist`; see there for the
        detection model and the dead-time convention.

        Args:
            phi_bar (torch.Tensor): Expected photon arrival rates for one pixel across time bins.
            n_pulses (int): Number of laser pulses to simulate.
            n_hist_bins (int): Number of histogram bins.
            free_running (bool): True for free-running mode, False for gated mode.
            dead_time_bins (int): Number of time bins for dead time.
            fast_sim (bool): Unused; kept for call-site compatibility.
            paralyzable (bool): If True, every arrival re-opens the dead window.
            generator (torch.Generator | None): Optional RNG for reproducibility.

        Returns:
            torch.Tensor: A tensor representing the accumulated photon histogram for the pixel.
        """
        return self.simulate_pixel_photon_hist(
            phi_bar, n_pulses, n_hist_bins, free_running, dead_time_bins,
            paralyzable=paralyzable, generator=generator,
        )

    def simulate_ewh(
        self,
        arrival_rates: torch.Tensor,
        n_pulses: int,
        n_hist_bins: int,
        free_running: bool = True,
        dead_time_bins: int = 0,
        fast_sim: bool = True,
    ) -> list[torch.Tensor]:
        """
        Simulates the Equi-Width Histogram (EWH) for all pixels/FOVs.

        Args:
            arrival_rates (torch.Tensor): Tensor of photon arrival rates for all FOVs.
            n_pulses (int): Number of laser pulses to simulate.
            n_hist_bins (int): Number of histogram bins.
            free_running (bool): True for free-running mode, False for gated mode.
            dead_time_bins (int): Number of time bins for dead time.

        Returns:
            list[torch.Tensor]: A list of tensors, where each tensor is the EWH for a pixel.
        """

        if fast_sim:
            if free_running:
                photon_hist_probs = batch_distorted_transient_async(
                    arrival_rates, dead_time_bins, n_hist_bins
                )
            else:
                photon_hist_probs = batch_distorted_transient_sync(
                    arrival_rates, dead_time_bins, n_hist_bins
                )
            photon_hists = photon_hist_probs * n_pulses
            return [photon_hists[i] for i in range(photon_hists.shape[0])]

        ewh_pixel_list = []
        for p_idx in tqdm(range(arrival_rates.shape[0]), desc="Simulating EWH"):
            ewh_pixel_list.append(
                self.simulate_pixel_ewh(arrival_rates[p_idx], n_pulses, n_hist_bins, free_running, dead_time_bins, fast_sim=fast_sim)
            )
        return ewh_pixel_list

    def simulate_ewh_diff(
        self,
        arrival_rates: torch.Tensor,
        n_pulses: int,
        n_hist_bins: int,
        free_running: bool = True,
        dead_time_bins: int = 0,
    ) -> torch.Tensor:
        """
        Simulates the Equi-Width Histogram (EWH) for all pixels/FOVs (differentiable version).

        Args:
            arrival_rates (torch.Tensor): Tensor of photon arrival rates for all FOVs.
            n_pulses (int): Number of laser pulses to simulate.
            n_hist_bins (int): Number of histogram bins.
            free_running (bool): True for free-running mode, False for gated mode.
            dead_time_bins (int): Number of time bins for dead time.

        Returns:
            torch.Tensor: A tensor of EWH histograms for all pixels.
        """
        assert dead_time_bins == 0, "Current differentiable EWH does not support non-zero dead time"

        # Use expected value instead of sampling for differentiability
        # For Poisson distribution, E[X] = lambda = arrival_rates * n_pulses
        ewh_pixel_list = arrival_rates * n_pulses

        return ewh_pixel_list


class HistogrammerEDH(HistogrammerBase):
    """
    Equi-Depth Histogrammer class
    """

    def __init__(self, config: HistConfig = HistConfig()):
        super().__init__()
        config.validate()
        for k, v in config.__dict__.items():
            setattr(self, k, v)

    def photon_hist2edh(self, photon_hist, n_edh_bins):
        """
        Converts a photon histogram to Equi-Depth Histogram (EDH) bins.

        Args:
            photon_hist (torch.Tensor): Photon histogram.
            n_edh_bins (int): Number of EDH bins.

        Returns:
            torch.Tensor: EDH bin boundaries.
        """
        gt_ntime_bins = photon_hist.shape[-1]

        tr_img = photon_hist + torch.max(photon_hist) * 0.0000000001 / gt_ntime_bins
        tr_cs = torch.cumsum(tr_img, axis=-1)
        tr_sum = torch.sum(tr_img, axis=-1)
        edh_bins = torch.zeros(n_edh_bins + 1, device=photon_hist.device)
        edh_bins[-1] = gt_ntime_bins

        for idx in range(edh_bins.shape[-1] - 2):
            e1_ori = tr_cs - tr_sum * (idx + 1.0) / n_edh_bins
            e1 = e1_ori * 1.0
            e2 = e1_ori * 1.0
            e1[e1_ori > 0] = -1000000000000.0
            e2[e1_ori < 0] = 1000000000000.0

            neg_max_val_, neg_max_idx_ = torch.max(e1, axis=-1)
            pos_min_val_, pos_min_idx_ = torch.min(e2, axis=-1)
            k1 = 1  # pos_min_idx_ - neg_max_idx_
            a1 = torch.abs(neg_max_val_)
            b1 = pos_min_val_
            edh_bins[idx + 1] = neg_max_idx_ + a1 * k1 * 1.0 / (a1 + b1 + 0.00000000000001)

        edh_bins[1:-1] += 1

        return edh_bins

    def simulate_pixel_edh(
        self,
        phi_bar: torch.Tensor,
        n_pulses: int,
        n_hist_bins: int,
        free_running: bool,
        dead_time_bins: int,
        paralyzable: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Simulates the Equi-Depth Histogram (EDH) for a single pixel.

        The EDH is a reduction of the equi-width photon histogram, not a
        separate detection model: the pixel is simulated once at full TDC
        resolution and the equi-depth boundaries are then read off the result.

        Note that ``n_hist_bins`` means something different here than it does
        for the EWH. It is the number of *equi-depth* bins, i.e. the number of
        boundaries to place; the underlying photon histogram is always kept at
        the full ``phi_bar`` time-bin resolution, since rebinning it first would
        quantise the boundaries.

        Args:
            phi_bar (torch.Tensor): Expected photon arrival rates for one pixel across time bins.
            n_pulses (int): Number of laser pulses to simulate.
            n_hist_bins (int): Number of equi-depth bins.
            free_running (bool): True for free-running mode, False for gated mode.
            dead_time_bins (int): Number of time bins for dead time.
            paralyzable (bool): If True, every arrival re-opens the dead window.
            generator (torch.Generator | None): Optional RNG for reproducibility.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Photon histogram, at full time-bin resolution.
                - torch.Tensor: EDH bin boundaries.
        """
        n_tbins = phi_bar.shape[-1]
        photon_hist = self.simulate_pixel_photon_hist(
            phi_bar, n_pulses, n_tbins, free_running, dead_time_bins,
            paralyzable=paralyzable, generator=generator,
        )
        photon_edh = self.photon_hist2edh(photon_hist, n_hist_bins)
        return photon_hist, photon_edh

    def simulate_edh(
        self,
        arrival_rates: torch.Tensor,
        n_pulses: int,
        n_hist_bins: int,
        free_running: bool = True,
        dead_time_bins: int = 0,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Simulates the Equi-Depth Histogram (EDH) for all pixels/FOVs.

        Args:
            arrival_rates (torch.Tensor): Tensor of photon arrival rates for all FOVs.
            n_pulses (int): Number of laser pulses to simulate.
            n_hist_bins (int): Number of histogram bins.
            free_running (bool): True for free-running mode, False for gated mode.
            dead_time_bins (int): Number of time bins for dead time.

        Returns:
            tuple: A tuple containing:
                - list[torch.Tensor]: List of photon histograms for each pixel.
                - list[torch.Tensor]: List of EDH bin boundaries for each pixel.
        """
        edh_pixel_list = []
        photon_hist_pixel_list = []

        for p_idx in tqdm(range(arrival_rates.shape[0]), desc="Simulating EDH"):
            pixel_photon_hist, pixel_edh = self.simulate_pixel_edh(
                arrival_rates[p_idx], n_pulses, n_hist_bins, free_running, dead_time_bins
            )
            photon_hist_pixel_list.append(pixel_photon_hist)
            edh_pixel_list.append(pixel_edh)
        return photon_hist_pixel_list, edh_pixel_list
