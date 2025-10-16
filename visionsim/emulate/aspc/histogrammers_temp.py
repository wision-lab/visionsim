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
from utils import get_irradiance_with_fov, ureg

def get_albedo_intensity_depth_frames(data_dir: str, Nr: int = 0, Nc: int = 0, device: torch.device = torch.device("cpu")):
    """
    Loads RGB and depth frames, processes them, and returns them as tensors.
    (Placeholder function which will be replaced by the RGBD dataloader)

    Args:
        data_dir (str): Path to the directory containing image files.
        num_frames (int): Number of frames to load.
        Nr (int): Target number of rows for resizing. If 0, no resizing.
        Nc (int): Target number of columns for resizing. If 0, no resizing.
        device (torch.device): The device to load tensors onto ('cpu' or 'cuda').

    Returns:
        tuple: Tensors of albedo frames, intensity frames, and depth frames.
    """
    albedo_frames, intensity_frames, depth_frames = [], [], []

    print(f"Loading frames from {data_dir}...")
    rgb_dir = os.path.join(data_dir, "frames")
    depth_dir = os.path.join(data_dir, "depths")
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.lower().endswith(('.png','.hdr'))])
    print("rgb_files.shape",len(rgb_files))
    print("depth_files.shape",len(depth_files))

    # Use the minimum number of frames available in both folders
    # num_frames = min(len(rgb_files), len(depth_files))
    num_frames = 1
    for idx in tqdm(range(num_frames), desc="Loading frames"):
        # i = idx + 1  # Frame index (1-based, to match original naming)
        rgb_file = rgb_files[idx]
        depth_file = depth_files[idx]
        rgb_img_pth = os.path.join(data_dir, "frames", rgb_file)
        # depth_img_pth = os.path.join(data_dir, "depths", f"{i}.png")
        depth_img_pth = os.path.join(data_dir, "depths", depth_file)

        if not os.path.exists(rgb_img_pth) or not os.path.exists(depth_img_pth):
            print(f"Warning: Missing files for frame {idx}. Skipping.")
            continue

        rgb_img = cv2.imread(rgb_img_pth, 1)[50:-50, 50:-50, ::-1] # Remove border, BGR to RGB
        depth_img = cv2.imread(depth_img_pth, -1).astype(float)[50:-50, 50:-50] # Remove border, unchanged

        # display depth image
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title(f"Depth Image {idx}")
        plt.imshow(depth_img, cmap="viridis")
        plt.colorbar(label="Depth (raw units)")
        plt.show()

        print("Idx: ",idx, "depth: ", depth_img.shape, depth_img.min(), depth_img.max(), depth_img.mean(), depth_img.std())
        print("Idx: ",idx, "albedo: ", rgb_img.shape, rgb_img.min(), rgb_img.max(), rgb_img.mean(), rgb_img.std())

        if Nr and Nc:
            rgb_img = cv2.resize(rgb_img, (Nc, Nr))
            depth_img = cv2.resize(depth_img, (Nc, Nr))

        # Normalize depth to meters, assuming 255 max value maps to 10.0 meters
        depth_img = depth_img * 10.0 / 255.0
        print("depth_img.shape",depth_img.shape)

        # Assuming laser wavelength is close to infrared (red channel for albedo)
        albedo_frames.append(rgb_img[:, :, 0] / 255.0)
        # Convert RGB to grayscale for intensity
        intensity_frames.append(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY) / 255.0)
        # depth_frames.append(depth_img)
        # INSERT_YOUR_CODE
        # If depth_img has 3 dimensions, squeeze the last dimension
        if depth_img.ndim == 3:
            depth_frames.append(depth_img[:,:,0])
        else:
            depth_frames.append(depth_img)

        # display depth image
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title(f"Depth Image {idx}")
        plt.imshow(depth_img, cmap="viridis")
        plt.colorbar(label="Depth (raw units)")
        plt.show()

        print("Idx: ",idx, "depth: ", depth_img.shape, depth_img.min(), depth_img.max(), depth_img.mean(), depth_img.std())
        print("Idx: ",idx, "albedo: ", albedo_frames[-1].shape, albedo_frames[-1].min(), albedo_frames[-1].max(), albedo_frames[-1].mean())
        print("Idx: ",idx, "intensity: ", intensity_frames[-1].shape, intensity_frames[-1].min(), intensity_frames[-1].max(), intensity_frames[-1].mean())

    # Convert lists of numpy arrays to torch tensors
    albedo_frames_tensor = torch.tensor(np.array(albedo_frames), dtype=torch.float32, device=device)
    intensity_frames_tensor = torch.tensor(np.array(intensity_frames), dtype=torch.float32, device=device)
    depth_frames_tensor = torch.tensor(np.array(depth_frames), dtype=torch.float32, device=device) * ureg.meter

    return albedo_frames_tensor, intensity_frames_tensor, depth_frames_tensor

def get_pixel_fov_mask(empty_mask: Tensor, row1: float, row2: float, col1: float, col2: float) -> Tensor:
    """
    Generates a rectangular FOV mask with a smooth vignette (weights in [0, 1]).
    1.0 at the rectangle center, smoothly decreasing to 0.0 at the rectangle edges,
    and 0.0 outside the rectangle. Values represent how unmasked a pixel is.

    Args:
        empty_mask (Tensor): Empty array used to define output shape (H, W). Values are ignored.
        row1 (float): Normalized start row (0.0 to 1.0).
        row2 (float): Normalized end row (0.0 to 1.0).
        col1 (float): Normalized start column (0.0 to 1.0).
        col2 (float): Normalized end column (0.0 to 1.0).

    Returns:
        Tensor: A float mask in [0, 1] with vignette inside the specified rectangle.
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

    # Build coordinate grid for the rectangle region
    rr = torch.arange(r1, r2)
    cc = torch.arange(c1, c2)
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
    vignette = torch.clip(1.0 - torch.pow(radial, exponent), 0.0, 1.0)

    mask[r1:r2, c1:c2] = vignette
    return mask


def get_perpixel_fov_masks(empty_mask: Tensor, pixel_fov_list: list, device: torch.device = torch.device("cpu")) -> Tensor:
    """
    Generates a list of FOV masks based on `pixel_fov_list`.
    (Placeholder function which can be added as a builder method to sensor class)

    Args:
        empty_mask (Tensor): An array to define the shape of the masks.
        pixel_fov_list (list): List of FOV coordinates [row1, row2, col1, col2].
        device (torch.device): The device to load tensors onto ('cpu' or 'cuda').

    Returns:
        torch.Tensor: A tensor of float masks with values in [0, 1].
    """
    mask_list = []
    for r1, r2, c1, c2 in pixel_fov_list:
        mask_list.append(get_pixel_fov_mask(empty_mask, r1, r2, c1, c2).unsqueeze(0))

    return torch.concat(mask_list)

def calculate_transients(irradiance_frames: torch.Tensor, 
                         depth_frames: torch.Tensor,
                         offsets: torch.Tensor,
                         fov_masks: torch.Tensor, 
                         gt_ntime_bins: int, 
                         max_depth: float,
                         sensor_fov: list,
                         pixel_fov_list: list,
                         w: int,
                         h: int,
                         omega: float) -> torch.Tensor:
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

    print("Calculating transients...")
    transient_list = []
    ambient_offsets = []
    for frame_idx in range(irradiance_frames.shape[0]):
        # Get values only within the current FOV mask for the first frame
        # Assuming albedo/depth don't change much across frames for transient calculation within an FOV
        # If transients should be frame-specific, this loop needs to be nested over frames.
        # For simplicity, using albedo_frames[0] and depth_frames[0]

        for mask_idx, fov_mask in enumerate(tqdm(fov_masks, desc="Processing FOV masks")):
            index_mask = fov_mask > 0
            current_irradiance_vals = irradiance_frames[frame_idx][torch.nonzero(fov_mask,as_tuple=True)]
            # print(f"current_irradiance_vals: {current_irradiance_vals}")
            fov_irradiance_vals = get_irradiance_with_fov(current_irradiance_vals, sensor_fov, pixel_fov_list[mask_idx], omega, w, h)
            # Apply vignette weights to irradiance
            current_irradiance_vals = current_irradiance_vals * fov_mask[index_mask]
            current_depth_vals = depth_frames[frame_idx][torch.nonzero(fov_mask,as_tuple=True)]
            masked_offsets = offsets[frame_idx][torch.nonzero(fov_mask,as_tuple=True)]
            ambient_offsets.append(masked_offsets.sum() / gt_ntime_bins)
            # print(f"offset: {offsets}")
            # print(f"ambient_offsets: {ambient_offsets}")

            # Convert depth values to time bin locations
            current_depth_vals = current_depth_vals.magnitude
            fov_irradiance_vals = fov_irradiance_vals.magnitude
            # print(f"fov_irradiance_vals: {fov_irradiance_vals}")
            # print("current_depth_vals",current_depth_vals.min(),current_depth_vals.max())
            # print("max_depth",max_depth)
            transient_idx = torch.floor(current_depth_vals * gt_ntime_bins / max_depth).to(torch.long)   
            transient_idx = torch.clamp(transient_idx, 0, gt_ntime_bins - 1) # Ensure indices are within bounds
            
            # Use torch.scatter_add for efficient accumulation into transients
            t1 = transients[mask_idx].scatter_add(0, transient_idx, fov_irradiance_vals)
            transient_list.append(t1.unsqueeze(0))
    
    final_transients = torch.concat(transient_list)
    return final_transients, ambient_offsets


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
    print("Calculating arrival rates...")
    for i in tqdm(range(transients.shape[0]), desc="Convolving transients"):
        # Reshape for conv1d: (batch_size, in_channels, signal_length)
        convolved_signal = F.conv1d(transients[i].view(1, 1, -1),
                                    irf.view(1, 1, -1),
                                    padding='same').view(-1)
        # Add signal and background components
        background = offset[i]
        background_extended = background.expand_as(convolved_signal)
        arrival_rates[i, :] = convolved_signal + background_extended
        # print(f"transients[i]: {transients[i]}")
        # print(f"convolved_signal: {convolved_signal}")
        # print(f"background: {background}")
        # print(f"background_extended: {background_extended}")
        # print(f"arrival_rates[i, :]: {arrival_rates[i, :]}")
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
