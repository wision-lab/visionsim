import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.constants import c 
import torch
import time
import torch.nn.functional as F
from tqdm import tqdm
import os # Import os for path checking
from torch import Tensor

def get_pixel_fov_mask(empty_mask: Tensor, row1: float, row2: float, col1: float, col2: float) -> np.ndarray:
    """
    Generates a rectangular FOV mask for each pixel based on row,column parameters.
    (Placeholder function which can be added as a builder method to sensor class)

    Args:
        empty_mask (np.ndarray): Passing empty array which can be reused to create fov masks.
        row1 (float): Normalized start row (0.0 to 1.0).
        row2 (float): Normalized end row (0.0 to 1.0).
        col1 (float): Normalized start column (0.0 to 1.0).
        col2 (float): Normalized end column (0.0 to 1.0).

    Returns:
        np.ndarray: A boolean mask where the specified region is True.
        
    """
    img_rows, img_cols = empty_mask.shape
    empty_mask = empty_mask*0

    empty_mask[int(row1 * img_rows): int(row2 * img_rows),
         int(col1 * img_cols): int(col2 * img_cols)] = True

    return empty_mask

def get_perpixel_fov_masks(empty_mask: Tensor, pixel_fov_list: list, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Generates a list of FOV masks based on `pixel_fov_list`.
    (Placeholder function which can be added as a builder method to sensor class)

    Args:
        empty_mask (Tensor): An array to define the shape of the masks.
        pixel_fov_list (list): List of FOV coordinates [row1, row2, col1, col2].
        device (torch.device): The device to load tensors onto ('cpu' or 'cuda').

    Returns:
        torch.Tensor: A tensor of boolean masks.
    """
    mask_list = []
    for r1, r2, c1, c2 in pixel_fov_list:
        mask_list.append(get_pixel_fov_mask(empty_mask, r1, r2, c1, c2).unsqueeze(0))
    
    return torch.concat(mask_list)

def calculate_transients(irradiance_frames: torch.Tensor, 
                         depth_frames: torch.Tensor,
                         fov_masks: torch.Tensor, 
                         gt_ntime_bins: int, max_depth: float) -> torch.Tensor:
    """
    Calculates the transient signal for each defined pixel FOV.

    Args:
        irradiance_frames (torch.Tensor): Tensor of irradiance images.
        depth_frames (torch.Tensor): Tensor of depth images.
        fov_masks (torch.Tensor): Tensor of boolean FOV masks.
        gt_ntime_bins (int): Total number of time bins for the transient.
        max_depth (float): Maximum depth corresponding to the last time bin.

    Returns:
        torch.Tensor: A tensor containing the calculated transients.
    """
    num_transients = fov_masks.shape[0] # Number of FOVs
    transients = torch.zeros((num_transients, gt_ntime_bins), 
                             dtype=irradiance_frames.dtype, 
                             device=irradiance_frames.device,
                             requires_grad=irradiance_frames.requires_grad)

    # print("Calculating transients...")
    transient_list = []
    for mask_idx, fov_mask in enumerate(fov_masks):
        # Get values only within the current FOV mask for the first frame
        # Assuming albedo/depth don't change much across frames for transient calculation within an FOV
        # If transients should be frame-specific, this loop needs to be nested over frames.
        # For simplicity, using albedo_frames[0] and depth_frames[0]

        for frame_idx in range(irradiance_frames.shape[0]):
            current_irradiance_vals = irradiance_frames[frame_idx][torch.nonzero(fov_mask,as_tuple=True)]
            current_depth_vals = depth_frames[frame_idx][torch.nonzero(fov_mask,as_tuple=True)]
            
            # if not frame_idx:
            #     print("current_irradiance_vals",current_irradiance_vals.min(),current_irradiance_vals.max())
            #     print("current_depth_vals",current_depth_vals.min(),current_depth_vals.max())

            # Convert depth values to time bin locations
            transient_idx1 = torch.floor(current_depth_vals * gt_ntime_bins / max_depth).to(torch.long)
            transient_idx = torch.clamp(transient_idx1, 0, gt_ntime_bins - 1) # Ensure indices are within bounds
            
            # Use torch.scatter_add for efficient accumulation into transients
            t1 = transients[mask_idx].scatter_add(0, transient_idx, current_irradiance_vals)
        transient_list.append(t1.unsqueeze(0))
    
    final_transients = torch.concat(transient_list)
    return final_transients

def calculate_arrival_rates(irf: torch.Tensor, transients: torch.Tensor, offset: torch.Tensor, gt_ntime_bins: int) -> torch.Tensor:
    """
    Calculates the photon arrival rates by convolving transients with the IRF
    and adding background noise.

    Args:
        irf (torch.Tensor): The instrumental response function.
        transients (torch.Tensor): The calculated transients.
        phi_sig (float): Signal photon rate scaling factor.
        phi_bkg (float): Background photon rate scaling factor.
        gt_ntime_bins (int): Total number of time bins.

    Returns:
        torch.Tensor: A tensor of photon arrival rates.
    """
    arrival_rates = torch.zeros_like(transients, dtype=torch.float32, device=transients.device)
    # print("Calculating arrival rates...")
    for i in range(transients.shape[0]):
        # Reshape for conv1d: (batch_size, in_channels, signal_length)
        convolved_signal = F.conv1d(transients[i].view(1, 1, -1),
                                    irf.view(1, 1, -1),
                                    padding='same').view(-1)
        # Add signal and background components
        offset_normalized = (offset - offset.min()) / (offset.max() - offset.min() + 1e-9)
        background = offset_normalized.mean() * 0.01
        background_extended = background.expand_as(convolved_signal)
        arrival_rates[i, :] = convolved_signal + background_extended
        # arrival_rates[i, :] = (convolved_signal)
    return arrival_rates

def _apply_non_pr_deadtime(buffer: torch.Tensor, dead_time_bins: int, n_tbins: int):
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

    for idx in current_arrivals_indices:
        # Check for previous photon detection within the dead time window
        start_check = int(max(idx - dead_time_bins, 0))
        end_check = idx # Up to (but not including) the current photon
        
        # If any photon was detected in the previous dead_time_bins, current photon is "missed"
        if torch.any(buffer[start_check:end_check]):
            buffer[idx] = False # Set current photon detection to False (missed)

def simulate_pixel_ewh(phi_bar: torch.Tensor, n_pulses: int, n_hist_bins: int,
                       free_running: bool, dead_time_bins: int) -> torch.Tensor:
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
    
    # Buffer to store arrivals for dead-time checking
    # First half for previous pulse arrivals, second half for current pulse arrivals
    buffer = torch.zeros((n_tbins * 2), dtype=torch.bool, device=phi_bar.device)

    for n_ in range(n_pulses):
        # Generate photon arrivals using Poisson distribution
        photon_vec = torch.poisson(phi_bar)
        buffer[n_tbins:] = photon_vec > 0 # Mark where photons arrived in current pulse

        # Apply non-paralyzable dead time
        if dead_time_bins > 0:
            _apply_non_pr_deadtime(buffer, dead_time_bins, n_tbins)

        # Accumulate detected photons into the histogram
        photon_hist += buffer[n_tbins:].float()

        # Update buffer for next pulse based on free-running or gated mode
        if free_running:
            buffer[:n_tbins] = buffer[n_tbins:] # Carry over current arrivals to previous for next iteration
        else:
            buffer[:n_tbins] = 0 # Clear previous buffer (gated mode)
    return photon_hist

def simulate_ewh(arrival_rates: torch.Tensor, n_pulses: int, n_hist_bins: int,
                 free_running: bool = False, dead_time_bins: int = 0) -> list[torch.Tensor]:
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
    ewh_pixel_list = []
    print("Simulating EWH for pixels...")
    for p_idx in tqdm(range(arrival_rates.shape[0]), desc="Simulating EWH"):
        ewh_pixel_list.append(simulate_pixel_ewh(arrival_rates[p_idx],
                                                 n_pulses,
                                                 n_hist_bins,
                                                 free_running,
                                                 dead_time_bins))
    return ewh_pixel_list

def simulate_ewh_diff(arrival_rates: torch.Tensor, n_pulses: int, n_hist_bins: int,
                 free_running: bool = False, dead_time_bins: int = 0) -> list[torch.Tensor]:
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
    
    assert dead_time_bins == 0, "Current differentiable EWH does not support non-zero dead time"

    ewh_pixel_list = torch.poisson(arrival_rates*n_pulses)

    return ewh_pixel_list

def photon_hist2edh(photon_hist, n_edh_bins):

    gt_ntime_bins = photon_hist.shape[-1]

    tr_img = photon_hist + torch.max(photon_hist)*0.0000000001/gt_ntime_bins
    bins = tr_img.shape
    tr_cs = torch.cumsum(tr_img, axis=-1)
    tr_sum = torch.sum(tr_img, axis=-1)
    edh_bins = torch.zeros(n_edh_bins+1)
    edh_bins[-1] = gt_ntime_bins

    for idx in range(edh_bins.shape[-1]-2):
      e1_ori = tr_cs - tr_sum*(idx+1.0)/n_edh_bins
      e1 = e1_ori*1.0
      e2 = e1_ori*1.0
      e1[e1_ori > 0] = -1000000000000.0
      e2[e1_ori < 0] = 1000000000000.0

      neg_max_val_, neg_max_idx_ = torch.max(e1, axis=-1)
      pos_min_val_, pos_min_idx_ = torch.min(e2, axis=-1)
      k1 = 1# pos_min_idx_ - neg_max_idx_
      a1 = torch.abs(neg_max_val_)
      b1 = pos_min_val_
      edh_bins[idx+1] = (neg_max_idx_ + a1*k1*1.0/(a1+b1+0.00000000000001))

    edh_bins[1:-1]+=1

    return edh_bins

def simulate_pixel_edh(phi_bar: torch.Tensor, n_pulses: int, n_hist_bins: int,
                       free_running: bool, dead_time_bins: int) -> torch.Tensor:
    """
    Simulates the Equi-Depth Histogram (EDH) for a single pixel.

    Args:
        phi_bar (torch.Tensor): Expected photon arrival rates for one pixel across time bins.
        n_pulses (int): Number of laser pulses to simulate.
        n_hist_bins (int): Number of histogram bins.
        free_running (bool): True for free-running mode, False for gated mode.
        dead_time_bins (int): Number of time bins for dead time.

    Returns:
        torch.Tensor: A tensor representing the accumulated photon histogram for the pixel.
    """
    n_tbins = phi_bar.shape[-1]
    photon_hist = torch.zeros(n_tbins, dtype=torch.float32, device=phi_bar.device)
    
    # Buffer to store arrivals for dead-time checking
    # First half for previous pulse arrivals, second half for current pulse arrivals
    buffer = torch.zeros((n_tbins * 2), dtype=torch.bool, device=phi_bar.device)

    for n_ in range(n_pulses):
        # Generate photon arrivals using Poisson distribution
        photon_vec = torch.poisson(phi_bar)
        buffer[n_tbins:] = photon_vec > 0 # Mark where photons arrived in current pulse

        # Apply non-paralyzable dead time
        if dead_time_bins > 0:
            _apply_non_pr_deadtime(buffer, dead_time_bins, n_tbins)

        # Accumulate detected photons into the histogram
        photon_hist += buffer[n_tbins:].float()

        # Update buffer for next pulse based on free-running or gated mode
        if free_running:
            buffer[:n_tbins] = buffer[n_tbins:] # Carry over current arrivals to previous for next iteration
        else:
            buffer[:n_tbins] = 0 # Clear previous buffer (gated mode)
    
    photon_edh = photon_hist2edh(photon_hist, n_hist_bins)
    
    return photon_hist, photon_edh

def simulate_edh(arrival_rates: torch.Tensor, n_pulses: int, n_hist_bins: int,
                 free_running: bool = False, dead_time_bins: int = 0) -> list[torch.Tensor]:
    """
    Simulates the Equi-Depth Histogram (EDH) for all pixels/FOVs.

    Args:
        arrival_rates (torch.Tensor): Tensor of photon arrival rates for all FOVs.
        n_pulses (int): Number of laser pulses to simulate.
        n_hist_bins (int): Number of histogram bins.
        free_running (bool): True for free-running mode, False for gated mode.
        dead_time_bins (int): Number of time bins for dead time.

    Returns:
        list[torch.Tensor]: A list of tensors, where each tensor is the EDH for a pixel.
    """
    edh_pixel_list = []
    photon_hist_pixel_list = []

    print("Simulating EDH for pixels...")
    for p_idx in tqdm(range(arrival_rates.shape[0]), desc="Simulating EDH"):
        pixel_photon_hist, pixel_edh = simulate_pixel_edh(arrival_rates[p_idx],
                                                 n_pulses,
                                                 n_hist_bins,
                                                 free_running,
                                                 dead_time_bins)
        photon_hist_pixel_list.append(pixel_photon_hist)
        edh_pixel_list.append(pixel_edh)
    return photon_hist_pixel_list, edh_pixel_list
