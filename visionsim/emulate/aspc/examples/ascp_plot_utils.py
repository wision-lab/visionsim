from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import shift
from copy import deepcopy
from scipy.ndimage import correlate1d
def cropDepth(depth_map,crop_window):
    depth_map = np.flipud(deepcopy(depth_map))
    x_min = crop_window[0]
    x_max = crop_window[1]
    y_min = crop_window[2]
    y_max = crop_window[3]
    return np.flipud(depth_map[y_min:y_max,x_min:x_max])
def get_depthMap_from_echos(statistics_frame):
    c = 3e8  # m/s
    bin_width = .758e-9
    #statistics_frame[np.isinf(statistics_frame)] = -1
    echo_max = np.argmax(statistics_frame[:,:,:,0],axis=-1)
    #print("stats_frame_depth_map",statistics_frame[97,84])
    peak_bin = np.take_along_axis(
        statistics_frame[:,:,:,1],
        echo_max[..., None],
        axis=2
    ).squeeze(-1)


    # Step 2: Convert to time
    time_delay = peak_bin * bin_width  # seconds

    # Step 3: Convert to depth
    depth_map = (c * time_delay) / 2  # meters
    depth_map[depth_map < 0] = 0
    return np.flipud(depth_map)
def get_kernels(roi_mem_path):
    with open(roi_mem_path,"r") as f:
        roi_mem = f.readlines()
        roi_mem = [x.strip() for x in roi_mem]
    
    number_of_registers_per_chunk = len(roi_mem)//32
    
    roi_chunked = []
    for i in range(32):
        
        chunk = roi_mem[i*number_of_registers_per_chunk:(i+1)*number_of_registers_per_chunk]
        roi_chunked.append(chunk)
    kernels = [x[5:13] for x in roi_chunked]
    full_kernels = []
    for kernel in kernels:
        full_kn = []
        kernel=kernel[:4]
        #print(kernel)
        for kernel_part in kernel:
            full_kn.append(int(kernel_part[2:4],16)) 
            full_kn.append(int(kernel_part[0:2],16))
        summed = sum(full_kn)
 
        full_kernels.append([x/summed for x in full_kn])

    return np.asarray(full_kernels)



def matched_filter(raw_histogram,roi_mem_path):
    fir_filter = get_kernels(roi_mem_path)
    cube = deepcopy(raw_histogram)
    cube = cube.reshape(32,1536,672)
    
    convolved_cube = np.empty((32,1536,672))
    for i in range(32):
        kernel = fir_filter[i]
        histogram_block = cube[i,:,:].astype(np.float64)
        for j in range(1536):
            histogram = histogram_block[j,:]
            convolved_cube[i,j,:] = correlate1d(histogram,kernel[::-1],mode="constant")
    return convolved_cube.reshape(192,256,672)

def get_depthMap(raw_hist,shape0,shape1):
        
        try:
            raw_hist= raw_hist.reshape((192,256,672))
        except:
           raw_hist = raw_hist.reshape((shape0,shape1,672))
        c = 3e8  # m/s
        bin_width = .758e-9 #1ns
        peak_bin = np.argmax(raw_hist, axis=2)  # shape (192, 256)

        # Step 2: Convert to time
        time_delay = peak_bin * bin_width  # seconds

        # Step 3: Convert to depth
        depth_map = (c * time_delay) / 2  # meters
        return depth_map
def fix_depth(raw_histo,zdd_fp,offset_fp):
    
    raw_histo = raw_histo.reshape((192,256,672))
    with open(zdd_fp,"r") as f:
        zdds = f.readlines()
    zdd_timing = 0.0625
    zdds = [float(x.strip().split(",")[1])*zdd_timing for x in zdds]
    
    bin_width_ns = .758
    
    zdds = [int(np.ceil(x/bin_width_ns)) for x in zdds]

    #raw_histo = np.roll(raw_histo,-25,axis=2)
    for i in range(32):
        raw_histo[i*6:(i+1)*6,:,:] = shift(raw_histo[i*6:(i+1)*6,:,:], shift=(0, 0, -(zdds[i]+2)), order=0, mode='constant', cval=0)

    with open(offset_fp, "r") as f:
        offsets= f.readlines()
        offsets = [float(x.strip()) for x in offsets]

    offsets = np.array(offsets).reshape(192,256)
    mask = offsets < 5000
    rows, cols = np.where(mask)

    # Flatten offsets for easy iteration
    shifts = -np.ceil(offsets[mask] * 6.25 / bin_width_ns).astype(np.int32)

    for r, c, s in zip(rows, cols, shifts):
        raw_histo[r, c, :] = shift(raw_histo[r, c, :], shift=s, order=0, mode='constant', cval=0)#np.roll(raw_histo[r, c, :], s)
    return raw_histo

def get_peak(convolved_histo,fitting_window,zero_extent):

    window = len(fitting_window)+zero_extent
    max_args = convolved_histo.argmax(axis=-1)
    max_vals = convolved_histo.max(axis=-1)

    bins = np.arange(convolved_histo.shape[-1])[None,:]
    max_idx = np.clip(max_args[...,None], 0, convolved_histo.shape[-1]-1)

    ## what if max_idx 0 or the max, may need more clipping here.
    mask = (bins >= max_idx - window) & (bins <= max_idx + window)
   
    convolved_histo[mask] = 0
    
    return max_args,convolved_histo

def get_stats(peak_idx, fitting_window, raw_histo):
    peak_idx_expanded = peak_idx[..., None]
    offsets = fitting_window

    idxs = peak_idx_expanded + offsets
    
    idxs = np.clip(idxs,0,raw_histo.shape[-1]-1).astype(int)
    window_counts = np.take_along_axis(raw_histo,idxs,axis=-1)

    total_energy = np.sum(window_counts,axis=-1).astype(np.float64)
    denom = np.sum(window_counts, axis=-1, keepdims=True)
    window_counts = np.divide(
        window_counts,
        denom,
        out=np.zeros_like(window_counts, dtype=np.float64),
        where=denom > 0
    )
    mean_bins = np.zeros_like(total_energy)

    nonzero = total_energy > 0
    mean_bins[nonzero] = np.sum(window_counts[nonzero] * idxs[nonzero], axis=-1).astype(np.float64) 
    variance_final = np.zeros_like(total_energy)
    denom = 1 - np.sum(window_counts[nonzero]**2,axis=-1)
    variance_final[nonzero] = np.sum(window_counts[nonzero]*(idxs[nonzero] - mean_bins[nonzero,None])**2,axis=-1)/denom
    

    return total_energy, mean_bins, variance_final, idxs[:,0]

def validate_peak():

    return np.ones((192, 256), dtype=bool)

def compute_statistics(raw_histo, zdd_fp,roi_mem_path, window_length,left_offset,right_offset, zero_extent, n):

    ## Matched filter -> get valid echos and statistics
    fitting_window = np.arange(-window_length+left_offset,window_length+right_offset)
    filtered_histogram = matched_filter(raw_histo,roi_mem_path)
    statistics_frame = np.zeros((192,256,n,4))


    
    
    for peak_number in range(n):
        found_valid = np.zeros((192, 256), dtype=bool)
        max_arg = np.zeros((192,256))
        energy_peak = np.zeros((192,256))
        mean_peak = np.zeros((192,256))
        var_peak = np.zeros((192,256))
        start_idx = np.zeros((192,256))

        while_counter = 0
        while_max = n + 10
        while (not np.all(found_valid)) and (while_counter < while_max):
            ## get a peak for all pixels
            
            max_arg[~found_valid],filtered_histogram[~found_valid,:] = get_peak(filtered_histogram[~found_valid,:],fitting_window,zero_extent)

            ## Compute Overlap

            ## Compute the statistics
            energy_peak[~found_valid],mean_peak[~found_valid],var_peak[~found_valid],start_idx[~found_valid] = get_stats(max_arg[~found_valid],fitting_window,raw_histo[~found_valid,:])

            ## test which pixels have a valid first echo
            is_legal = validate_peak()

            ## save the ones that are valid
            just_fixed = is_legal & (~found_valid)

            # Save ONLY the newly valid ones to the frame
            if np.any(just_fixed):
                statistics_frame[just_fixed, peak_number, 0] = energy_peak[just_fixed]
                statistics_frame[just_fixed, peak_number, 1] = mean_peak[just_fixed]
                statistics_frame[just_fixed, peak_number, 2] = var_peak[just_fixed]
                statistics_frame[just_fixed, peak_number, 3] = start_idx[just_fixed]
            while_counter +=1
            found_valid = found_valid | is_legal
    #statistics_frame = offset_correction(statistics_frame,zdd_fp,n)
    return statistics_frame


def plot_spad_sensor_grid(
    histogrammer, fov_masks, grid_shape, albedo_frame, depth_frame, transients, arrival_rates, ewh_list, save_path=None,raw_histo=None
):
    """
    Visualizes SPAD data and saves three distinct plots:
    1. Reconstruction: Ground Truth vs Simulator Estimate
    2. Overlay: Sensor Grid over Ground Truth Albedo/Depth
    3. Waveforms: Transient Response vs Generated Histogram (EWH)
    """
    rows, cols = grid_shape

    # --- HELPER: ROBUST CONVERSION TO NUMPY ---
    def ensure_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        # Handle Pint quantities (including those wrapping CUDA tensors)
        if hasattr(x, "magnitude"):
            return ensure_numpy(x.magnitude)
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        if isinstance(x, (list, tuple)):
            if len(x) > 0 and torch.is_tensor(x[0]):
                return torch.stack(x).detach().cpu().numpy()
            else:
                return np.array([ensure_numpy(v) for v in x])
        return np.asarray(x)

    # Convert inputs
    albedo_img = ensure_numpy(albedo_frame)
    depth_img = ensure_numpy(depth_frame)
    ewh_data = ensure_numpy(ewh_list)
    transient_data = ensure_numpy(transients)

    # ---------------------------------------------------------
    # PRE-CALCULATE METRICS
    # ---------------------------------------------------------

    # depth_map_est = np.zeros((rows, cols))
    # depth_map_gt = np.zeros((rows, cols))

    # # Calculate bin width
    # max_depth_val = histogrammer.max_depth
    # if hasattr(max_depth_val, "magnitude"):
    #     max_depth_val = max_depth_val.magnitude
    # bin_dist_m = max_depth_val / histogrammer.n_bins

    # # Fill maps
    # for r in range(rows):
    #     for c in range(cols):
    #         idx = r * cols + c

    #         # Estimate
    #         if idx < len(ewh_data):
    #             peak_bin = np.argmax(ewh_data[r,c])
    #             # Filter: Only estimate depth if there are photons
    #             if np.sum(ewh_data[r,c]) > 0:
    #                 depth_map_est[r, c] = peak_bin * bin_dist_m

    #         # Ground Truth
    #         if idx < len(fov_masks):
    #             mask = ensure_numpy(fov_masks[idx])
    #             valid_depths = depth_img[mask > 0]
    #             if len(valid_depths) > 0:
    #                 depth_map_gt[r, c] = np.mean(valid_depths)

    # --- GLOBAL SCALING ---

    zdd_fp = "/u/g/u/gump/vsim/zddIndex_19_358296692.txt" 
    #raw_histo = fix_depth(np.flipud(raw_histo),zdd_fp,"/nobackup2/gump/gpuLaptopFiles/testing_new_glare_method/offset.txt")

    roi_mem_path = "/u/g/u/gump/vsim/roi_mem_B34_1ms.txt"

    n = 3
    distance_thresh = 0
    window = 4
    left_offset = -1
    right_offset = 2
    
    crop_window = (45,220,5,137)
    gt_stats =compute_statistics(raw_histo,zdd_fp,roi_mem_path,window,left_offset,right_offset,distance_thresh,n)
    depth_map_gt = cropDepth(np.flipud(get_depthMap_from_echos(gt_stats)),crop_window)

    #ewh_data[:,:,:] = shift(ewh_data[:,:,:], shift=(0, 0, -10), order=0, mode='constant', cval=0)
    est_stats =compute_statistics(np.flipud(ewh_data.copy()),zdd_fp,roi_mem_path,window,left_offset,right_offset,distance_thresh,n)
    depth_map_est = cropDepth(get_depthMap_from_echos(est_stats),crop_window)
  

    ewh_data = ewh_data.reshape(transient_data.shape[0],transient_data.shape[1],672)
    vmin_depth = .5
    vmax_depth = 5.5

    # Handle Filenames
    scenario_name = ""
    if save_path:
        path_obj = Path(save_path)
        base_name = path_obj.stem.replace("_reconstruction", "")
        parent = path_obj.parent
        scenario_name = f": {base_name}"

        save_path_recon = parent / f"{base_name}_reconstruction.png"
        save_path_overlay = parent / f"{base_name}_overlay.png"
        save_path_waveforms = parent / f"{base_name}_waveforms.png"

    # =========================================================
    # PLOT 1: RECONSTRUCTION
    # =========================================================
    fig_maps, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig_maps.suptitle(f"Reconstruction{scenario_name}", fontsize=16)

    im1 = ax1.imshow(depth_map_gt, cmap="jet", vmin=vmin_depth, vmax=vmax_depth)
    ax1.set_title("Ground Truth (Ideal)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(depth_map_est, cmap="jet", vmin=vmin_depth, vmax=vmax_depth)
    ax2.set_title("Simulator Estimate (Physics-based)")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()

    # =========================================================
    # PLOT 2: OVERLAY
    # =========================================================
    fig_ov, ax_ov = plt.subplots(1, 2, figsize=(12, 6))
    fig_ov.suptitle(f"Sensor FOV Overlay{scenario_name}", fontsize=16)

    ax_ov[0].imshow(albedo_img, cmap="gray")
    ax_ov[0].set_title("RGB/Albedo + FOV Grid")

    im_ov = ax_ov[1].imshow(depth_img, cmap="jet", vmin=vmin_depth, vmax=vmax_depth)
    ax_ov[1].set_title("High-Res Depth + FOV Grid")
    plt.colorbar(im_ov, ax=ax_ov[1], fraction=0.046, pad=0.04)

    # Draw Rectangles (Limit to 2000 to prevent crash)
    if len(fov_masks) < 2000:
        for i, mask in enumerate(fov_masks):
            mask_np = ensure_numpy(mask)
            y_indices, x_indices = np.where(mask_np > 0)
            if len(y_indices) > 0:
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                w, h = x_max - x_min, y_max - y_min
                for ax in ax_ov:
                    rect = patches.Rectangle(
                        (x_min, y_min), w, h, linewidth=1, edgecolor="r", facecolor="none", alpha=0.5
                    )
                    ax.add_patch(rect)
    plt.tight_layout()

    # =========================================================
    # PLOT 3: WAVEFORM GRID (Transient vs EWH)
    # =========================================================
    MAX_PLOTS = 16

    # If too many pixels, just plot the 4 corners
    fig_wave, ax = plt.subplots(2, 2, figsize=(10, 8))
    fig_wave.suptitle(
        f"Corner Pixels - Waveforms{scenario_name}\nBlue: Transient (Ideal) | Orange: EWH (Noisy)", fontsize=14
    )
    ewh_mid_y, ewh_mid_x = 83,132
    mid_y_raw, mid_x_raw =  96,126#np.unravel_index(np.argmax(raw_histo,axis=-1),(192,256))
    corners = [
        (0,-2,"Middle (x-2)"),
        (0, 2,"Middle x+2"),
        (-2,0,"Middle y-2"),
        (2, 0,  "Middle y+2"),

    ]
    # mid_y_raw, mid_x_raw =  77,126
    # corners = [
    #     (0,0,"Middle (x-2)"),
    #     (1, 0,"Middle x+2"),
    #     (2,0,"Middle y-2"),
    #     (-2, 0,  "Middle y+2"),

    # ]
    axes = [(0,0),(0,1),(1,0),(1,1)]
    raw_sum = (raw_histo.max(axis=-1)) + 1 

    raw_sum = raw_sum[...,None]
    raw_histo = raw_histo / raw_sum

    ewh_maxs = ewh_data.max(axis=-1) + 1
    ewh_maxs = ewh_maxs[...,None]
    ewh_data = ewh_data/ewh_maxs

    transient_maxs = transient_data.max(axis=-1) + 1
    transient_maxs = transient_maxs[...,None]
    transient_data = transient_data/transient_maxs

    for i, (s_y,s_x, name) in enumerate(corners):
        shape_y_diff, shape_x_diff = (raw_histo.shape[0]//transient_data.shape[0]),(raw_histo.shape[1]//transient_data.shape[1])
        idx_y_transient, idx_x_transient = (ewh_mid_y + s_y),(ewh_mid_x+s_x)
        idx_y_raw, idx_x_raw = (mid_y_raw + s_y),(mid_x_raw+s_x)
        t_data = transient_data[idx_y_transient, idx_x_transient]
        ax[axes[i]].plot(t_data[:100], color="tab:blue", alpha=1, linewidth=2.5, label="Transient")
        ax[axes[i]].plot(raw_histo[idx_y_raw, idx_x_raw][:100],color="red",label="Raw Histogram")

        e_data = ewh_data[idx_y_transient, idx_x_transient]
        ax[axes[i]].plot(e_data[:100], color="tab:orange", alpha=0.8, linewidth=1.5, label="EWH")

        ax[axes[i]].set_title(f"{name} (Row {s_y}, Col {s_x})")

        if i == 0:  # Legend only on first
            lines_1, labels_1 = ax[axes[i]].get_legend_handles_labels()
            
            ax[axes[i]].legend(lines_1, labels_1, loc="upper right", fontsize="small")


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # =========================================================
    # SAVE OR SHOW
    # =========================================================
    if save_path:
        print(f"Saving reconstruction to {save_path_recon}...")
        fig_maps.savefig(save_path_recon, dpi=150)

        print(f"Saving overlay to {save_path_overlay}...")
        fig_ov.savefig(save_path_overlay, dpi=150)

        print(f"Saving waveforms to {save_path_waveforms}...")
        fig_wave.savefig(save_path_waveforms, dpi=150)

        plt.close(fig_maps)
        plt.close(fig_ov)
        plt.close(fig_wave)
    else:
        plt.show()


def plot_ewh_per_pixel(histogrammer, fov_masks, albedo_frame, depth_frame, transients, arrival_rates, ewh_list):
    # print("Min max of depth_frame", depth_frame.min(), depth_frame.max())
    # print("Ewh list", ewh_list)
    # print("arrival_rates", arrival_rates)

    # Plots
    # Get number of FOVs from fov_masks
    num_fovs = fov_masks.shape[0]
    # FOV Masks
    fig1, ax1 = plt.subplots(1, num_fovs, figsize=(3 * num_fovs, 3))
    fig1.suptitle("FOV Masks", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax1 if num_fovs == 1 else ax1[i]
        current_ax.imshow(fov_masks[i].detach().cpu().numpy(), cmap="gray")
        current_ax.set_title(f"FOV {i + 1}")
        current_ax.axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent suptitle overlap
    # plt.show()

    # Albedo values for the first frame
    fig2, ax2 = plt.subplots(1, num_fovs, figsize=(3 * num_fovs, 3))
    fig2.suptitle("Albedo Values (First Frame)", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax2 if num_fovs == 1 else ax2[i]
        current_ax.imshow(
            albedo_frame.detach().cpu().numpy() * fov_masks[i].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1
        )
        current_ax.set_title(f"FOV {i + 1}")
        current_ax.axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    # Depth values for the first frame
    fig3, ax3 = plt.subplots(1, num_fovs, figsize=(3 * num_fovs, 3))
    fig3.suptitle("Depth Values (First Frame)", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax3 if num_fovs == 1 else ax3[i]
        current_ax.imshow(
            # The depth frame visualization should not be multiplied with a continuous FOV mask
            # The visualization only shows the region of the depth frame that contributes to the transient for a specific pixel
            depth_frame.detach().cpu().numpy() * (fov_masks[i].detach().cpu().numpy() > 0),
            cmap="jet",
            # depth_frame.detach().cpu().numpy() * fov_masks[i].detach().cpu().numpy(), cmap="jet"
        )  # Assuming max depth of 10m based on 10.0/255.0 scaling
        current_ax.set_title(f"FOV {i + 1}")
        current_ax.axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    # Transients
    fig4, ax4 = plt.subplots(num_fovs, 1, figsize=(8, 2.5 * num_fovs))
    fig4.suptitle("Transients", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax4 if num_fovs == 1 else ax4[i]
        current_ax.plot(transients[i].detach().cpu().numpy())
        current_ax.set_title(f"FOV {i + 1}")
        current_ax.set_xlabel("Time Bins")
        current_ax.set_ylabel("Normalized Amplitude")
        current_ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    # Arrival Rates
    fig5, ax5 = plt.subplots(num_fovs, 1, figsize=(8, 2.5 * num_fovs))
    fig5.suptitle(r"Photon Arrival Rates ($\overline{\Phi}$)", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax5 if num_fovs == 1 else ax5[i]
        current_ax.plot(arrival_rates[i].detach().cpu().numpy())
        current_ax.set_ylim(bottom=0)  # Ensure y-axis starts at 0
        current_ax.set_title(f"FOV {i + 1}")
        current_ax.set_xlabel("Time Bins")
        current_ax.set_ylabel("Rate (photons/bin)")
        current_ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    # Time Stamp Histograms (EWH)
    fig6, ax6 = plt.subplots(num_fovs, 1, figsize=(8, 2.5 * num_fovs))
    fig6.suptitle("Simulated Time Stamp Histograms (EWH)", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax6 if num_fovs == 1 else ax6[i]
        current_ax.plot(ewh_list[i].detach().cpu().numpy())
        current_ax.set_ylim(bottom=0)  # Ensure y-axis starts at 0
        current_ax.set_title(f"FOV {i + 1}")
        current_ax.set_xlabel("Time Bins")
        current_ax.set_ylabel("Photon Counts")
        current_ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_edh_per_pixel(
    config, fov_masks, albedo_frames, depth_frames, transients, arrival_rates, photon_hist_pixel_list, edh_pixel_list
):
    """
    Plots the simulation results.
    """
    num_fovs = len(fov_masks)
    if num_fovs == 0:
        print("No FOVs to plot.")
        return

    # FOV Masks
    fig1, ax1 = plt.subplots(1, num_fovs, figsize=(3 * num_fovs, 3))
    fig1.suptitle("FOV Masks", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax1 if num_fovs == 1 else ax1[i]
        current_ax.imshow(fov_masks[i].detach().cpu().numpy(), cmap="gray")
        current_ax.set_title(f"FOV {i + 1}")
        current_ax.axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent suptitle overlap
    plt.show()

    # Albedo values for the first frame
    fig2, ax2 = plt.subplots(1, num_fovs, figsize=(3 * num_fovs, 3))
    fig2.suptitle("Albedo Values (First Frame)", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax2 if num_fovs == 1 else ax2[i]
        current_ax.imshow(
            albedo_frames.detach().cpu().numpy() * fov_masks[i].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1
        )
        current_ax.set_title(f"FOV {i + 1}")
        current_ax.axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Depth values for the first frame
    fig3, ax3 = plt.subplots(1, num_fovs, figsize=(3 * num_fovs, 3))
    fig3.suptitle("Depth Values (First Frame)", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax3 if num_fovs == 1 else ax3[i]
        current_ax.imshow(
            depth_frames.detach().cpu().numpy() * fov_masks[i].detach().cpu().numpy(), cmap="viridis", vmin=0, vmax=10
        )  # Assuming max depth of 10m based on 10.0/255.0 scaling
        current_ax.set_title(f"FOV {i + 1}")
        current_ax.axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Transients
    fig4, ax4 = plt.subplots(num_fovs, 1, figsize=(8, 2.5 * num_fovs))
    fig4.suptitle("Transients", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax4 if num_fovs == 1 else ax4[i]
        current_ax.plot(transients[i].detach().cpu().numpy())
        current_ax.set_title(f"FOV {i + 1}")
        current_ax.set_xlabel("Time Bins")
        current_ax.set_ylabel("Normalized Amplitude")
        # current_ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Arrival Rates
    fig5, ax5 = plt.subplots(num_fovs, 1, figsize=(8, 2.5 * num_fovs))
    fig5.suptitle(r"Photon Arrival Rates ($\overline{\Phi}$)", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax5 if num_fovs == 1 else ax5[i]
        current_ax.plot(arrival_rates[i].detach().cpu().numpy())
        current_ax.set_ylim(bottom=0)  # Ensure y-axis starts at 0
        current_ax.set_title(f"FOV {i + 1}")
        current_ax.set_xlabel("Time Bins")
        current_ax.set_ylabel("Rate (photons/bin)")
        # current_ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Time Stamp Histograms
    fig6, ax6 = plt.subplots(num_fovs, 1, figsize=(8, 2.5 * num_fovs))
    fig6.suptitle("Simulated Photon Time Stamp Histograms \n and Equi-Depth Histograms", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax6 if num_fovs == 1 else ax6[i]
        current_ax.plot(photon_hist_pixel_list[i].detach().cpu().numpy())
        edh_bins = edh_pixel_list[i].detach().cpu().numpy()
        # counts_per_bin = np.ones(edh_bins.shape[-1])*np.sum(photon_hist_pixel_list[i].cpu().numpy())*1.0/(edh_bins.shape[-1]-1)
        bin_height = np.ones(edh_bins.shape[-1]) * np.max(photon_hist_pixel_list[i].detach().cpu().numpy())
        bin_widths = np.diff(edh_bins, 1, axis=-1)
        current_ax.bar(
            edh_bins[:-1],
            bin_height[:-1],
            width=bin_widths,
            color="none",
            align="edge",
            edgecolor="black",
            linewidth=1,
            alpha=0.2,
        )
        current_ax.set_ylim(bottom=0)  # Ensure y-axis starts at 0
        current_ax.set_title(f"FOV {i + 1}")
        current_ax.set_xlabel("Time Bins")
        current_ax.set_ylabel("Normalized Photon Counts")
        # current_ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
