import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import torch
from pathlib import Path 

def plot_spad_sensor_grid(
    histogrammer, 
    fov_masks, 
    grid_shape, 
    albedo_frame, 
    depth_frame, 
    transients, 
    arrival_rates, 
    ewh_list,
    save_path=None
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
        if isinstance(x, np.ndarray): return x
        if torch.is_tensor(x): return x.detach().cpu().numpy()
        if isinstance(x, list):
            if len(x) > 0 and torch.is_tensor(x[0]):
                return torch.stack(x).detach().cpu().numpy()
            else:
                return np.array(x)
        return np.array(x)

    # Convert inputs
    albedo_img = ensure_numpy(albedo_frame)
    depth_img = ensure_numpy(depth_frame)
    ewh_data = ensure_numpy(ewh_list)
    transient_data = ensure_numpy(transients)

    # ---------------------------------------------------------
    # PRE-CALCULATE METRICS
    # ---------------------------------------------------------
    depth_map_est = np.zeros((rows, cols))
    depth_map_gt = np.zeros((rows, cols))
    
    # Calculate bin width
    max_depth_val = histogrammer.max_depth
    if hasattr(max_depth_val, 'magnitude'):
        max_depth_val = max_depth_val.magnitude     
    bin_dist_m = max_depth_val / histogrammer.n_bins 
    
    # Fill maps
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            
            # Estimate
            if idx < len(ewh_data):
                peak_bin = np.argmax(ewh_data[idx])
                # Filter: Only estimate depth if there are photons
                if np.sum(ewh_data[idx]) > 0: 
                    depth_map_est[r, c] = peak_bin * bin_dist_m
            
            # Ground Truth
            if idx < len(fov_masks):
                mask = ensure_numpy(fov_masks[idx])
                valid_depths = depth_img[mask > 0]
                if len(valid_depths) > 0:
                    depth_map_gt[r, c] = np.mean(valid_depths)

    # --- GLOBAL SCALING ---
    vmin_depth = 0
    vmax_depth = np.max(depth_map_gt)
    if vmax_depth == 0: vmax_depth = 1.0

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
    
    im1 = ax1.imshow(depth_map_gt, cmap='jet', interpolation='nearest', vmin=vmin_depth, vmax=vmax_depth)
    ax1.set_title("Ground Truth (Ideal)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    im2 = ax2.imshow(depth_map_est, cmap='jet', interpolation='nearest', vmin=vmin_depth, vmax=vmax_depth)
    ax2.set_title("Simulator Estimate (Physics-based)")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()

    # =========================================================
    # PLOT 2: OVERLAY
    # =========================================================
    fig_ov, ax_ov = plt.subplots(1, 2, figsize=(12, 6))
    fig_ov.suptitle(f"Sensor FOV Overlay{scenario_name}", fontsize=16)
    
    ax_ov[0].imshow(albedo_img, cmap='gray')
    ax_ov[0].set_title("RGB/Albedo + FOV Grid")
    
    im_ov = ax_ov[1].imshow(depth_img, cmap='jet', vmin=vmin_depth, vmax=vmax_depth) 
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
                    rect = patches.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor='r', facecolor='none', alpha=0.5)
                    ax.add_patch(rect)
    plt.tight_layout()

    # =========================================================
    # PLOT 3: WAVEFORM GRID (Transient vs EWH)
    # =========================================================
    MAX_PLOTS = 16 
    
    if rows * cols > MAX_PLOTS:
        # If too many pixels, just plot the 4 corners
        fig_wave, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig_wave.suptitle(f"Corner Pixels - Waveforms{scenario_name}\nBlue: Transient (Ideal) | Orange: EWH (Noisy)", fontsize=14)
        corners = [
            (0, 0, "Top-Left"), (0, cols - 1, "Top-Right"),
            (rows - 1, 0, "Bottom-Left"), (rows - 1, cols - 1, "Bottom-Right")
        ]
        ax_flat = axes.flatten()
        
        for i, (r, c, name) in enumerate(corners):
            idx = r * cols + c
            ax = ax_flat[i]
            if idx < len(transient_data):
                # Normalize for plotting visibility
                t_data = transient_data[idx]
                ax.plot(t_data, color='tab:blue', alpha=0.6, linewidth=1.5, label="Transient")
                
                ax2_w = ax.twinx()
                e_data = ewh_data[idx]
                ax2_w.plot(e_data, color='tab:orange', alpha=0.8, linewidth=1.5, label="EWH")
                
                ax.set_title(f"{name} (Row {r}, Col {c})")
                ax.set_yticks([])
                ax2_w.set_yticks([])
                if i == 0: # Legend only on first
                    lines_1, labels_1 = ax.get_legend_handles_labels()
                    lines_2, labels_2 = ax2_w.get_legend_handles_labels()
                    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize='small')
    else:
        # Plot every single pixel
        fig_wave, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2), squeeze=False)
        fig_wave.suptitle(f"Pixel-wise Waveforms{scenario_name}", fontsize=16)
        
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                ax = axes[r, c]
                if idx < len(transient_data):
                    t_data = transient_data[idx]
                    ax.plot(t_data, color='tab:blue', alpha=0.6, linewidth=1)
                    
                    ax2_w = ax.twinx()
                    e_data = ewh_data[idx]
                    ax2_w.plot(e_data, color='tab:orange', alpha=0.8, linewidth=1)
                    
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax2_w.set_yticks([])
                else:
                    ax.axis('off')

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
            depth_frame.detach().cpu().numpy() * (fov_masks[i].detach().cpu().numpy()>0), cmap="jet"
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
