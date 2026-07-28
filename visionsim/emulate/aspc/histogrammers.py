from dataclasses import dataclass, field
from typing import List, Union

import torch
import torch.nn.functional as F
from pint import Quantity
from torch import Tensor
from tqdm import tqdm
import numpy as np
from itertools import product

from visionsim.emulate.aspc.units import validate_units
from visionsim.emulate.aspc.utils import get_irradiance_with_fov, ureg


def calculate_distorted_transient(phi_bar, dead_time_bins, n_hist_bins):
  r""" PyTorch function translated from MATLAB/NumPy code from https://github.com/Goyal-STIR-Group/HighFluxSPL.
  Function generates the distorted transient given input as the true transient and deadtime.

  Citation reference:
  J. Rapp, Y. Ma, R. M. A. Dawson and V. K. Goyal, "Dead Time Compensation for High-Flux Ranging," 
  in IEEE Transactions on Signal Processing, vol. 67, no. 13, pp. 3471-3486, 1 July1, 2019, doi: 10.1109/TSP.2019.2914891.
  """
  num_bins = phi_bar.shape[-1]
  
  # FIX: Wrap the dead time to ensure it doesn't exceed the histogram bounds
  # We use int() to ensure we don't get tensor-type mismatch issues with the modulo
  dt_bins_wrapped = int(dead_time_bins) % int(n_hist_bins)

  cumulative_transient = torch.cumsum(phi_bar, dim=-1)
  total_photons = torch.sum(phi_bar, dim=-1, keepdim=True)

  transition_matrix = torch.zeros((num_bins, num_bins), dtype=phi_bar.dtype, device=phi_bar.device)

  for bin_idx in range(num_bins):
    if bin_idx + dt_bins_wrapped < n_hist_bins:
      # Use .clone() to prevent autograd issues with in-place modifications if training a model
      survival_exponent = (cumulative_transient + total_photons - cumulative_transient[bin_idx + dt_bins_wrapped : bin_idx + dt_bins_wrapped + 1]).clone()
      survival_exponent[bin_idx + dt_bins_wrapped + 1 :] = survival_exponent[bin_idx + dt_bins_wrapped + 1 :] - total_photons
      
      transition_matrix[bin_idx, :] = phi_bar * torch.exp(-survival_exponent) / (1 - torch.exp(-total_photons)) / (torch.sum(phi_bar * torch.exp(-survival_exponent) / (1 - torch.exp(-total_photons))) + 0.0000000000001)
    else:
      survival_exponent = (cumulative_transient + total_photons - cumulative_transient[bin_idx + dt_bins_wrapped - n_hist_bins : bin_idx + dt_bins_wrapped - n_hist_bins + 1]).clone()
      survival_exponent[bin_idx + dt_bins_wrapped - n_hist_bins + 1 :] = survival_exponent[bin_idx + dt_bins_wrapped - n_hist_bins + 1 :] - total_photons
      
      transition_matrix[bin_idx, :] = phi_bar * torch.exp(-survival_exponent) / (1 - torch.exp(-total_photons)) / (torch.sum(phi_bar * torch.exp(-survival_exponent) / (1 - torch.exp(-total_photons))) + 0.0000000000001)

  if not total_photons.item():
     return phi_bar

  eigenvalues, eigenvectors = torch.linalg.eig(transition_matrix.T)

  principal_eigen_idx = torch.argmax(torch.abs(eigenvalues))
  # No need to extract val, just the vectors
  principal_eigen_vec = eigenvectors[:, principal_eigen_idx].real

  if principal_eigen_vec[0] < 0:
    principal_eigen_vec = -1 * principal_eigen_vec

  distorted_transient = principal_eigen_vec * 1.0
  distorted_transient = distorted_transient / (torch.sum(distorted_transient))

  return distorted_transient



def calculate_distorted_transient_sync(
    phi_bar: torch.Tensor,
    dead_time_bins: int,
    n_hist_bins: int,
) -> torch.Tensor:
    """
    Synchronous forward model for pile-up distortion (single-hit and multi-hit).
 
    Models the detection probability distribution for a synchronous (gated) SPAD
    where the detector resets at each cycle boundary. Dead time never wraps across
    cycles.
 
    - Single-hit mode: set dead_time_bins >= n_hist_bins
    - Multi-hit mode:  set dead_time_bins to the actual detector dead time
 
    Based on the classical Coates pile-up model generalized to multi-hit operation.
 
    Args:
        phi_bar:         Arrival rates per bin (shape: [n_hist_bins]).
        dead_time_bins:  Dead time in number of bins. For single-hit (first-photon-wins),
                         pass dead_time_bins >= n_hist_bins.
        n_hist_bins:     Number of histogram bins per cycle.
 
    Returns:
        Normalized distorted transient (probability distribution, sums to 1).
    """
    device = phi_bar.device
    dtype = phi_bar.dtype
 
    # Per-bin detection probability: p_i = 1 - exp(-r_i)
    p_detect = 1.0 - torch.exp(-torch.clamp(phi_bar[:n_hist_bins], min=0.0))
 
    # prob_live[i]   = probability the detector is not in dead time at bin i
    # prob_detect[i] = probability of actually registering a photon at bin i
    prob_detect = torch.zeros(n_hist_bins, device=device, dtype=dtype)
 
    # Cycle start: detector is always live (synchronous reset)
    prob_detect[0] = p_detect[0]
 
    for i in range(1, n_hist_bins):
        # Detector is live at bin i iff no detection occurred in
        # bins [max(0, i - dead_time_bins) .. i-1]
        start = max(0, i - dead_time_bins)
        prob_live_i = torch.prod(1.0 - prob_detect[start:i])
        prob_detect[i] = prob_live_i * p_detect[i]
 
    # Normalize to a proper distribution
    total = prob_detect.sum()
    if total > 0:
        distorted = prob_detect / total
    else:
        distorted = prob_detect
 
    return distorted
 
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

        num_transients = fov_masks.shape[0]  # Number of FOVs
        transients = torch.zeros(
            (num_transients, gt_ntime_bins),
            dtype=irradiance_frames.dtype,
            device=irradiance_frames.device,
            requires_grad=irradiance_frames.requires_grad,
        )

        transient_list = []
        ambient_offsets = []
        for frame_idx in range(irradiance_frames.shape[0]):
            for mask_idx, fov_mask in enumerate(tqdm(fov_masks, desc="Processing FOV masks", disable=True)):
                index_mask = fov_mask > 0
                current_irradiance_vals = irradiance_frames[frame_idx][torch.nonzero(fov_mask, as_tuple=True)]

                # Apply FOV correction if parameters are provided
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

                # Apply vignette weights to irradiance
                current_irradiance_vals = current_irradiance_vals * fov_mask[index_mask]
                current_depth_vals = depth_frames[frame_idx][torch.nonzero(index_mask, as_tuple=True)]
                masked_offsets = offsets[frame_idx][torch.nonzero(index_mask, as_tuple=True)]
                ambient_offsets.append(masked_offsets.sum() / gt_ntime_bins)

                # Extract magnitude only if these are Pint Quantity objects, otherwise use as-is
                if hasattr(current_depth_vals, "magnitude"):
                    current_depth_vals = current_depth_vals.magnitude
                if hasattr(fov_irradiance_vals, "magnitude"):
                    fov_irradiance_vals = fov_irradiance_vals.magnitude

     
                # Convert depth values to time bin locations
                transient_idx = torch.floor(current_depth_vals * gt_ntime_bins / max_depth).to(torch.long)
                transient_idx = torch.clamp(transient_idx, 0, gt_ntime_bins - 1)  # Ensure indices are within bounds
           
                # Use torch.scatter_add for efficient accumulation into transients
                t1 = transients[mask_idx].scatter_add(0, transient_idx, fov_irradiance_vals)

                transient_list.append(t1.unsqueeze(0))

        final_transients = torch.concat(transient_list)
        return final_transients, ambient_offsets

    def calculate_arrival_rates(
        self, 
        irf: torch.Tensor, 
        transients: torch.Tensor, 
        offset, 
        gt_ntime_bins: int, 
        glare_mask: torch.Tensor, 
        bloom_scale: float, 
        bloom_falloff: float, 
        scan_mode: str,
        scan_width: int,
        bloom_sim: bool, 
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

        irf = torch.tensor([25,34,52,113,255,207,0,0,0],dtype=torch.float32, device=transients.device)

        for i in tqdm(range(transients.shape[0]), desc="Convolving transients", disable=False):


            # Reshape for conv1d: (batch_size, in_channels, signal_length)
            convolved_signal = F.conv1d(transients[i].view(1, 1, -1), irf.view(1, 1, -1), padding="same").view(-1)
            
            arrival_rates[i, :] = convolved_signal

        if bloom_sim:
            # # Create a new array to hold the results so we don't corrupt our read data
            arrival_rates = arrival_rates.reshape((192, 256, 672))

            # 1. Unpack raw PyTorch tensor and units
            units = arrival_rates.units if hasattr(arrival_rates, 'units') else 1
            raw_arrival = arrival_rates.magnitude if hasattr(arrival_rates, 'magnitude') else arrival_rates
            raw_mask = glare_mask.magnitude if hasattr(glare_mask, 'magnitude') else glare_mask

            raw_arrival = raw_arrival.reshape((192, 256, 672))
            new_arrival_rates = torch.zeros_like(raw_arrival)

            device = raw_arrival.device
            dtype = raw_arrival.dtype

            # 2. Run loop purely on raw PyTorch tensors
            for i, j in tqdm(np.ndindex((192, 256)), total=192*256, desc="Adding Glare"):
            # Restrict window strictly to the active 6-row LiDAR chunk
                if scan_mode == "horizontal":
                    start_idx = (j // scan_width) * scan_width  # e.g., 0, 6, 12, ...
                    stop_idx = start_idx + scan_width            # e.g., 6, 12, 18, ...
                    local_j = j - start_idx                     # Local row index inside the 6-row slice (0 to 5)
                    
                    glare_contributers = raw_arrival[:, start_idx:stop_idx, :] * raw_mask[0, :, start_idx:stop_idx, None]
                    
                    i_coords = torch.arange(192, device=device, dtype=dtype)
                    j_coords = torch.arange(start_idx, stop_idx, device=device, dtype=dtype)
                    
                    distance = torch.sqrt((i_coords - i).unsqueeze(1)**2 + (j_coords - j).unsqueeze(0)**2)
                    decay_mask = bloom_scale * torch.exp(-bloom_falloff * distance)
                    decay_mask[i, local_j] = 0.0

                    weighted_glare = glare_contributers * decay_mask.unsqueeze(-1)

                    # Pure PyTorch addition (no Pint unit checking overhead inside loop)
                    new_arrival_rates[i, j, :] = raw_arrival[i, j, :] + weighted_glare.sum(dim=(0, 1))
                elif scan_mode == "vertical":
                    start_idx = (i // scan_width) * scan_width  # e.g., 0, 6, 12, ...
                    stop_idx = start_idx + scan_width            # e.g., 6, 12, 18, ...
                    local_i = i - start_idx                     # Local row index inside the 6-row slice (0 to 5)
                    
                    glare_contributers = raw_arrival[start_idx:stop_idx, :, :] * raw_mask[0, start_idx:stop_idx, :, None]
                    
                    i_coords = torch.arange(start_idx, stop_idx, device=device, dtype=dtype)
                    j_coords = torch.arange(256, device=device, dtype=dtype)
                    
                    distance = torch.sqrt((i_coords - i).unsqueeze(1)**2 + (j_coords - j).unsqueeze(0)**2)
                    decay_mask = bloom_scale * torch.exp(-bloom_falloff * distance)
                    decay_mask[local_i, j] = 0.0

                    weighted_glare = glare_contributers * decay_mask.unsqueeze(-1)

                    # Pure PyTorch addition (no Pint unit checking overhead inside loop)
                    new_arrival_rates[i, j, :] = raw_arrival[i, j, :] + weighted_glare.sum(dim=(0, 1))
                else:
                    ## TODO FLASH SYSTEM
                    pass


            # 3. Flatten and re-wrap Pint units
            arrival_rates = new_arrival_rates.flatten(start_dim=0, end_dim=1) * units

        ## Background add
        for i in tqdm(range(arrival_rates.shape[0]), desc="Adding Background", disable=False):

            if isinstance(offset, (list, tuple)):
                background = offset[i] if i < len(offset) else offset[0] if len(offset) > 0 else 0.0
            elif isinstance(offset, torch.Tensor):
                background = offset[i] if offset.dim() > 0 and i < offset.shape[0] else offset
            else:
                background = offset

            # Convert to tensor if needed
            if not isinstance(background, torch.Tensor):
                background = torch.tensor(background, dtype=torch.float32, device=transients.device)

            background_extended = background.expand_as(arrival_rates[i,:]) if background.dim() == 0 else background
            arrival_rates[i, :] = arrival_rates[i,:] + background_extended
        return arrival_rates

    def _apply_non_pr_deadtime(self, buffer: torch.Tensor, dead_time_bins: int, n_tbins: int):
        """
        Applies non-paralyzable dead time to a photon arrival buffer (helper function).

        Args:
            buffer (torch.Tensor): A boolean tensor representing photon arrivals over time.
                            The second half `buffer[n_tbins:]` contains current arrivals.
            dead_time_bins (int): Number of time bins for dead time.
            n_tbins (int): Number of time bins for a single pulse period.
        """
        # Identify indices where current arrivals occurred
        current_arrivals_indices = torch.nonzero(buffer[n_tbins:], as_tuple=True)[0] + n_tbins

        # print("current_arrivals_indices: ", current_arrivals_indices.shape)

        for idx in current_arrivals_indices:
            # Check for previous photon detection within the dead time window
            start_check = int(max(idx - dead_time_bins, 0))
            end_check = idx  # Up to (but not including) the current photon

            # If any photon was detected in the previous dead_time_bins, current photon is "missed"
            if torch.any(buffer[start_check:end_check]):
                buffer[idx] = False  # Set current photon detection to False (missed)


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
            [0, 0.999, 0, 0.999],
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
        self, phi_bar: torch.Tensor, n_pulses: int, n_hist_bins: int, free_running: bool, dead_time_bins: int, fast_sim: bool = True,
    ) -> torch.Tensor:
        """
        Simulates the Equi-Width Histogram (EWH) for a single pixel.

        Args:
            phi_bar (torch.Tensor): Expected photon arrival rates for one pixel across time bins.
            n_pulses (int): Number of laser pulses to simulate.
            n_hist_bins (int): Number of histogram bins.
            free_running (bool): True for free-running mode, False for gated mode.
            dead_time_bins (int): Number of time bins for dead time.

        Returns:
            torch.Tensor: A tensor representing the accumulated photon histogram for the pixel.
        """
        photon_hist = torch.zeros(n_hist_bins, dtype=torch.float32, device=phi_bar.device)
        n_tbins = phi_bar.shape[-1]

        # if fast_sim:

        #     if free_running:
        #         photon_hist_prob = calculate_distorted_transient(phi_bar, dead_time_bins, n_hist_bins)
        #     else:
        #         photon_hist_prob = calculate_distorted_transient_sync(phi_bar, dead_time_bins, n_hist_bins)

        #     photon_hist = photon_hist_prob * n_pulses

        #     return photon_hist

        # Buffer to store arrivals for dead-time checking
        # First half for previous pulse arrivals, second half for current pulse arrivals
        buffer = torch.zeros((n_tbins * 2), dtype=torch.bool, device=phi_bar.device)

        # Pre-calculate probability once (outside the loop) as we are assuming phi_bar is static per pulse.
        # Clamp negative arrival rates that can occur from convolution/offset math.
        # p(detection) = 1 - e^(-rate), with p in [0, 1]
        phi_bar = torch.clamp(phi_bar, min=0.0)
        prob_detection = 1.0 - torch.exp(-phi_bar)
        prob_detection = torch.clamp(prob_detection, 0.0, 1.0)

        # print("prob_detection", prob_detection.min(), prob_detection.max(), prob_detection.mean())

        for n_ in tqdm(range(n_pulses), desc="Simulating Pixel EWH",disable=True):
            detections = torch.bernoulli(prob_detection)

            # Update buffer
            buffer[n_tbins:] = detections.bool()

            # Apply non-paralyzable dead time
            if dead_time_bins > 0:
                self._apply_non_pr_deadtime(buffer, dead_time_bins, n_tbins)

            # Accumulate detected photons into the histogram
            photon_hist += buffer[n_tbins:].float()

            # Update buffer for next pulse based on free-running or gated mode
            if free_running:
                buffer[:n_tbins] = buffer[n_tbins:]  # Carry over current arrivals to previous for next iteration
            else:
                buffer[:n_tbins] = 0  # Clear previous buffer (gated mode)
        return photon_hist

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

    def gumbel_poisson(self, rate, K=50, tau=0.5):
        """
        Differentiable relaxation of Poisson sampling using Gumbel-softmax.

        Args:
            rate: (B, ...) Poisson rate λ
            K: Maximum value in support
            tau: Temperature parameter for Gumbel-softmax

        Returns:
            relaxed sample with same shape as rate
        """
        # Build support [0..K]
        ks = torch.arange(K + 1, device=rate.device).view(1, -1)

        # Compute unnormalized logits = log p(k | rate)
        # log p = -rate + k log rate - log(k!)
        log_probs = -rate.unsqueeze(-1) + ks * torch.log(rate.unsqueeze(-1) + 1e-9) - torch.lgamma(ks + 1)

        # Sample Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(log_probs)))

        # Relaxed Gumbel-Max --> Gumbel-Softmax
        y = F.softmax((log_probs + gumbel_noise) / tau, dim=-1)

        # Relaxed expected sample
        relaxed_sample = (y * ks).sum(dim=-1)

        return relaxed_sample

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
        # Alternative: ewh_pixel_list = self.gumbel_poisson(arrival_rates * n_pulses, K=40, tau=0.3)

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
        self, phi_bar: torch.Tensor, n_pulses: int, n_hist_bins: int, free_running: bool, dead_time_bins: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Simulates the Equi-Depth Histogram (EDH) for a single pixel.

        Args:
            phi_bar (torch.Tensor): Expected photon arrival rates for one pixel across time bins.
            n_pulses (int): Number of laser pulses to simulate.
            n_hist_bins (int): Number of histogram bins.
            free_running (bool): True for free-running mode, False for gated mode.
            dead_time_bins (int): Number of time bins for dead time.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Photon histogram.
                - torch.Tensor: EDH bin boundaries.
        """
        n_tbins = phi_bar.shape[-1]
        photon_hist = torch.zeros(n_tbins, dtype=torch.float32, device=phi_bar.device)

        # Buffer to store arrivals for dead-time checking
        # First half for previous pulse arrivals, second half for current pulse arrivals
        buffer = torch.zeros((n_tbins * 2), dtype=torch.bool, device=phi_bar.device)

        for n_ in range(n_pulses):
            # Generate photon arrivals using Poisson distribution
            photon_vec = torch.poisson(phi_bar)
            buffer[n_tbins:] = photon_vec > 0  # Mark where photons arrived in current pulse

            # Apply non-paralyzable dead time
            if dead_time_bins > 0:
                self._apply_non_pr_deadtime(buffer, dead_time_bins, n_tbins)

            # Accumulate detected photons into the histogram
            photon_hist += buffer[n_tbins:].float()

            # Update buffer for next pulse based on free-running or gated mode
            if free_running:
                buffer[:n_tbins] = buffer[n_tbins:]  # Carry over current arrivals to previous for next iteration
            else:
                buffer[:n_tbins] = 0  # Clear previous buffer (gated mode)

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