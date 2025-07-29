import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.constants import c 
import torch
import time
import torch.nn.functional as F
from tqdm.notebook import tqdm
import os # Import os for path checking

def get_albedo_intensity_depth_frames(data_dir: str, num_frames: int, Nr: int = 0, Nc: int = 0, device: torch.device = torch.device("cpu")):
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

    print(f"Loading {num_frames} frames from {data_dir}...")
    for i in tqdm(range(1, num_frames + 1), desc="Loading frames"):
        rgb_img_pth = os.path.join(data_dir, f"{i}.jpg")
        depth_img_pth = os.path.join(data_dir, f"{i}.png")

        if not os.path.exists(rgb_img_pth) or not os.path.exists(depth_img_pth):
            print(f"Warning: Missing files for frame {i}. Skipping.")
            continue

        rgb_img = cv2.imread(rgb_img_pth, 1)[50:-50, 50:-50, ::-1] # Remove border, BGR to RGB
        depth_img = cv2.imread(depth_img_pth, -1).astype(float)[50:-50, 50:-50] # Remove border, unchanged

        if Nr and Nc:
            rgb_img = cv2.resize(rgb_img, (Nc, Nr))
            depth_img = cv2.resize(depth_img, (Nc, Nr))

        # Normalize depth to meters, assuming 255 max value maps to 10.0 meters
        depth_img = depth_img * 10.0 / 255.0

        # Assuming laser wavelength is close to infrared (red channel for albedo)
        albedo_frames.append(rgb_img[:, :, 0] / 255.0)
        # Convert RGB to grayscale for intensity
        intensity_frames.append(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY) / 255.0)
        depth_frames.append(depth_img)

    # Convert lists of numpy arrays to torch tensors
    albedo_frames_tensor = torch.tensor(np.array(albedo_frames), dtype=torch.float32, device=device)
    intensity_frames_tensor = torch.tensor(np.array(intensity_frames), dtype=torch.float32, device=device)
    depth_frames_tensor = torch.tensor(np.array(depth_frames), dtype=torch.float32, device=device)

    return albedo_frames_tensor, intensity_frames_tensor, depth_frames_tensor

def get_pixel_fov_mask(empty_mask: np.ndarray, row1: float, row2: float, col1: float, col2: float) -> np.ndarray:
    """
    Generates a rectangular FOV mask for each pixel based on row,column parameters.
    (Placeholder function which can be added as a builder method to the sensor class)

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


def get_perpixel_fov_masks(empty_mask: np.ndarray, pixel_fov_list: list, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Generates a list of FOV masks based on `pixel_fov_list`.
    (Placeholder function which can be added as a builder method to the sensor class)

    Args:
        empty_mask (np.ndarray): An array to define the shape of the masks.
        pixel_fov_list (list): List of FOV coordinates [row1, row2, col1, col2].
        device (torch.device): The device to load tensors onto ('cpu' or 'cuda').

    Returns:
        torch.Tensor: A tensor of boolean masks.
    """
    mask_list = []
    for r1, r2, c1, c2 in pixel_fov_list:
        mask_list.append(get_pixel_fov_mask(empty_mask, r1, r2, c1, c2))
    return torch.tensor(np.array(mask_list), dtype=torch.bool, device=device)

def calculate_transients(albedo_frames: torch.Tensor, depth_frames: torch.Tensor,
                         fov_masks: torch.Tensor, gt_ntime_bins: int, max_depth: float) -> torch.Tensor:
    """
    Calculates the transient signal for each defined pixel FOV.

    Args:
        albedo_frames (torch.Tensor): Tensor of albedo images.
        depth_frames (torch.Tensor): Tensor of depth images.
        fov_masks (torch.Tensor): Tensor of boolean FOV masks.
        gt_ntime_bins (int): Total number of time bins for the transient.
        max_depth (float): Maximum depth corresponding to the last time bin.

    Returns:
        torch.Tensor: A tensor containing the calculated transients.
    """
    num_transients = fov_masks.shape[0] # Number of FOVs
    transients = torch.zeros((num_transients, gt_ntime_bins), dtype=torch.float32, device=fov_masks.device)

    print("Calculating transients...")
    for mask_idx, fov_mask in enumerate(tqdm(fov_masks, desc="Processing FOV masks")):
        # Get values only within the current FOV mask for the first frame
        # Assuming albedo/depth don't change much across frames for transient calculation within an FOV
        # If transients should be frame-specific, this loop needs to be nested over frames.
        # For simplicity, using albedo_frames[0] and depth_frames[0]
        current_albedo_vals = albedo_frames[0][fov_mask]
        current_depth_vals = depth_frames[0][fov_mask]

        # Convert depth values to time bin locations
        transient_idx = torch.floor(current_depth_vals * gt_ntime_bins / max_depth).to(torch.long)
        transient_idx = torch.clamp(transient_idx, 0, gt_ntime_bins - 1) # Ensure indices are within bounds

        # Calculate scale factor for each unique time bin
        scale_factor = current_albedo_vals / (current_depth_vals**2 + 1e-6) # Add epsilon to avoid division by zero
        
        # Use torch.scatter_add for efficient accumulation into transients
        transients[mask_idx].scatter_add_(0, transient_idx, scale_factor)

        # Normalize the transient to sum to 1
        transients[mask_idx] = transients[mask_idx] / (torch.sum(transients[mask_idx]) + 1e-9)

    return transients

def get_laser_irf(sigma_bins: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Generates a Gaussian instrumental response function (IRF) for the laser.

    Args:
        sigma_bins (int): Standard deviation of the Gaussian in time bins.
        device (torch.device): The device to create the tensor on.

    Returns:
        torch.Tensor: A 1D tensor representing the Gaussian IRF.
    """
    kernel_size = 10 * sigma_bins + 1
    center = kernel_size // 2
    x = torch.arange(0, kernel_size, dtype=torch.float32, device=device) - center
    gaussian = torch.exp(-(x**2) / (2 * sigma_bins**2))
    gaussian = gaussian / (gaussian.sum() + 1e-9) # Normalize to sum to 1
    return gaussian

def calculate_arrival_rates(irf: torch.Tensor, transients: torch.Tensor, phi_sig: float, phi_bkg: float, gt_ntime_bins: int) -> torch.Tensor:
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
        arrival_rates[i, :] = (convolved_signal * phi_sig +
                               phi_bkg / gt_ntime_bins)
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
        start_check = max(idx - dead_time_bins, 0)
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

def plot_results(albedo_frames: torch.Tensor, depth_frames: torch.Tensor,
                 fov_masks: torch.Tensor, transients: torch.Tensor,
                 arrival_rates: torch.Tensor, ewh_list: list[torch.Tensor],
                 pixel_fov_list: list):
    """
    Plots the simulation results.
    """
    num_fovs = len(pixel_fov_list)
    if num_fovs == 0:
        print("No FOVs to plot.")
        return

    # FOV Masks
    fig1, ax1 = plt.subplots(1, num_fovs, figsize=(3 * num_fovs, 3))
    fig1.suptitle("FOV Masks", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax1 if num_fovs == 1 else ax1[i]
        current_ax.imshow(fov_masks[i].cpu().numpy(), cmap="gray")
        current_ax.set_title(f"FOV {i+1}")
        current_ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
    plt.show()

    # Albedo values for the first frame
    fig2, ax2 = plt.subplots(1, num_fovs, figsize=(3 * num_fovs, 3))
    fig2.suptitle("Albedo Values (First Frame)", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax2 if num_fovs == 1 else ax2[i]
        current_ax.imshow(albedo_frames[0].cpu().numpy() * fov_masks[i].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        current_ax.set_title(f"FOV {i+1}")
        current_ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Depth values for the first frame
    fig3, ax3 = plt.subplots(1, num_fovs, figsize=(3 * num_fovs, 3))
    fig3.suptitle("Depth Values (First Frame)", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax3 if num_fovs == 1 else ax3[i]
        current_ax.imshow(depth_frames[0].cpu().numpy() * fov_masks[i].cpu().numpy(), cmap="viridis", vmin=0, vmax=10) # Assuming max depth of 10m based on 10.0/255.0 scaling
        current_ax.set_title(f"FOV {i+1}")
        current_ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Transients
    fig4, ax4 = plt.subplots(num_fovs, 1, figsize=(8, 2.5 * num_fovs))
    fig4.suptitle("Transients", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax4 if num_fovs == 1 else ax4[i]
        current_ax.plot(transients[i].cpu().numpy())
        current_ax.set_title(f"FOV {i+1}")
        current_ax.set_xlabel("Time Bins")
        current_ax.set_ylabel("Normalized Amplitude")
        current_ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Arrival Rates
    fig5, ax5 = plt.subplots(num_fovs, 1, figsize=(8, 2.5 * num_fovs))
    fig5.suptitle(r'Photon Arrival Rates ($\overline{\Phi}$)', fontsize=16)
    for i in range(num_fovs):
        current_ax = ax5 if num_fovs == 1 else ax5[i]
        current_ax.plot(arrival_rates[i].cpu().numpy())
        current_ax.set_ylim(bottom=0) # Ensure y-axis starts at 0
        current_ax.set_title(f"FOV {i+1}")
        current_ax.set_xlabel("Time Bins")
        current_ax.set_ylabel("Rate (photons/bin)")
        current_ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Time Stamp Histograms (EWH)
    fig6, ax6 = plt.subplots(num_fovs, 1, figsize=(8, 2.5 * num_fovs))
    fig6.suptitle("Simulated Time Stamp Histograms (EWH)", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax6 if num_fovs == 1 else ax6[i]
        current_ax.plot(ewh_list[i].cpu().numpy())
        current_ax.set_ylim(bottom=0) # Ensure y-axis starts at 0
        current_ax.set_title(f"FOV {i+1}")
        current_ax.set_xlabel("Time Bins")
        current_ax.set_ylabel("Photon Counts")
        current_ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    # Define simulation configuration
    sim_config = {
        "num_frames": min(5, 58), # Use min to prevent issues if fewer frames exist
        "data_dir": "/content/drive/MyDrive/VisionSim2/study_0006_out/",
        "gt_ntime_bins": 1000, # Resolution of the transient
        "tmax": 100 * 1e-9, # Total time window in seconds
        "fwhm": 0.5 * 1e-9, # Laser FWHM in seconds
        "phi_sig": 0.5, # Signal photon rate scaling factor
        "phi_bkg": 5.0, # Background photon rate scaling factor
        "n_pulses": 10000, # Number of laser pulses
        "n_hist_bins": 1000, # Number of histogram bins (should match gt_ntime_bins if not resampling)
        "dead_time_s": 10 * 1e-9, # Dead time in seconds
        "free_running": False, # Free-running mode (True) or gated mode (False)

        # Assigning fov using normalized rows and columns for each pixel
        # [row1, row2, col1, col2]
        "pixel_fov_list": [[0, 0.4, 0.3, 0.6],
                           [0.7, 0.95, 0.6, 0.9]]
    }

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Initialize Parameters from Config ---
    data_dir = sim_config["data_dir"]
    num_frames = sim_config["num_frames"]
    gt_ntime_bins = sim_config["gt_ntime_bins"]
    tmax = sim_config["tmax"]
    bin_size = tmax / gt_ntime_bins
    laser_freq = 1.0 / tmax
    max_depth = 0.5 * c / laser_freq  # c from scipy.constants
    fwhm = sim_config["fwhm"]
    sigma_laser = fwhm / 2.3548
    sigma_bins = int(sigma_laser / bin_size)
    phi_sig = sim_config["phi_sig"]
    phi_bkg = sim_config["phi_bkg"]
    n_pulses = sim_config["n_pulses"]
    n_hist_bins = sim_config["n_hist_bins"]
    dead_time_s = sim_config["dead_time_s"]
    free_running = sim_config["free_running"]
    dead_time_bins = int(dead_time_s / bin_size)
    pixel_fov_list = sim_config["pixel_fov_list"]
    num_transients = len(pixel_fov_list)

    print(f"Dead time in bins: {dead_time_bins}")
    print(f"Max depth: {max_depth:.3f} meters")

    start_time = time.time()
    print("\n--- Starting SPAD Simulation (Function-based) ---")

    # 1. Load data
    albedo_frames, intensity_frames, depth_frames = get_albedo_intensity_depth_frames(data_dir, num_frames, device=device)
    
    if albedo_frames.numel() == 0:
        print("No frames loaded. Exiting simulation.")
        exit() # Exit the script if no frames are loaded

    _, img_rows, img_cols = depth_frames.shape
    empty_mask = np.zeros((img_rows, img_cols), dtype=float)

    # 2. Get FOV masks
    fov_masks = get_perpixel_fov_masks(empty_mask, pixel_fov_list, device=device)

    # 3. Calculate transients
    transients = calculate_transients(albedo_frames, depth_frames, fov_masks, gt_ntime_bins, max_depth)

    # 4. Get laser IRF
    irf = get_laser_irf(sigma_bins, device=device)

    # 5. Calculate arrival rates
    arrival_rates = calculate_arrival_rates(irf, transients, phi_sig, phi_bkg, gt_ntime_bins)

    # 6. Simulate EWH
    ewh_list = simulate_ewh(arrival_rates, n_pulses, n_hist_bins, free_running=free_running, dead_time_bins=dead_time_bins)

    end_time = time.time()
    print(f"--- Simulation Finished in {end_time - start_time:.2f} seconds ---\n")

    # Plot results
    plot_results(albedo_frames, depth_frames, fov_masks,
                 transients, arrival_rates, ewh_list,
                 pixel_fov_list)
