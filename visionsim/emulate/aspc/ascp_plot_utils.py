import matplotlib.pyplot as plt


def plot_ewh_per_pixel(config,
                    fov_masks,
                    albedo_frame,
                    depth_frame,
                    transients,
                    arrival_rates,
                    ewh_list):
    
    print("Min max of depth_frame", depth_frame.min(), depth_frame.max())
    print("Ewh list", ewh_list)
    print("arrival_rates", arrival_rates)

    # Plots
    num_fovs = len(config['histogrammer']['pixel_fov_list'])
    # FOV Masks
    fig1, ax1 = plt.subplots(1, num_fovs, figsize=(3 * num_fovs, 3))
    fig1.suptitle("FOV Masks", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax1 if num_fovs == 1 else ax1[i]
        current_ax.imshow(fov_masks[i].detach().cpu().numpy(), cmap="gray")
        current_ax.set_title(f"FOV {i+1}")
        current_ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
    # plt.show()

    # Albedo values for the first frame
    fig2, ax2 = plt.subplots(1, num_fovs, figsize=(3 * num_fovs, 3))
    fig2.suptitle("Albedo Values (First Frame)", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax2 if num_fovs == 1 else ax2[i]
        current_ax.imshow(albedo_frame.detach().cpu().numpy() * fov_masks[i].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        current_ax.set_title(f"FOV {i+1}")
        current_ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    # Depth values for the first frame
    fig3, ax3 = plt.subplots(1, num_fovs, figsize=(3 * num_fovs, 3))
    fig3.suptitle("Depth Values (First Frame)", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax3 if num_fovs == 1 else ax3[i]
        current_ax.imshow(depth_frame.detach().cpu().numpy() * fov_masks[i].detach().cpu().numpy(), cmap="jet") # Assuming max depth of 10m based on 10.0/255.0 scaling
        current_ax.set_title(f"FOV {i+1}")
        current_ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    # Transients
    fig4, ax4 = plt.subplots(num_fovs, 1, figsize=(8, 2.5 * num_fovs))
    fig4.suptitle("Transients", fontsize=16)
    for i in range(num_fovs):
        current_ax = ax4 if num_fovs == 1 else ax4[i]
        current_ax.plot(transients[i].detach().cpu().numpy())
        current_ax.set_title(f"FOV {i+1}")
        current_ax.set_xlabel("Time Bins")
        current_ax.set_ylabel("Normalized Amplitude")
        current_ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    # Arrival Rates
    fig5, ax5 = plt.subplots(num_fovs, 1, figsize=(8, 2.5 * num_fovs))
    fig5.suptitle(r'Photon Arrival Rates ($\overline{\Phi}$)', fontsize=16)
    for i in range(num_fovs):
        current_ax = ax5 if num_fovs == 1 else ax5[i]
        current_ax.plot(arrival_rates[i].detach().cpu().numpy())
        current_ax.set_ylim(bottom=0) # Ensure y-axis starts at 0
        current_ax.set_title(f"FOV {i+1}")
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
        current_ax.set_ylim(bottom=0) # Ensure y-axis starts at 0
        current_ax.set_title(f"FOV {i+1}")
        current_ax.set_xlabel("Time Bins")
        current_ax.set_ylabel("Photon Counts")
        current_ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
