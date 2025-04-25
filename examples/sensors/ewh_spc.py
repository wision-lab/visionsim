from pathlib import Path

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.constants import c
from SPCSim.data_loaders.transient_loaders import TransientGenerator
from SPCSim.postproc.ewh_postproc import PostProcEWH
from SPCSim.sensors import BaseEWHSPC
from SPCSim.utils.plot_utils import plot_ewh, plot_transient

from visionsim.dataset import Dataset

root = Path("/home/kaustubh/spsim-main_old/examples/renders/scene1/")
frames = Dataset.from_path(root / "frames")
depths = Dataset.from_path(root / "depths")
assert len(depths) == len(frames), "Different number of depth and RGB frames"

Nr, Nc = [128, 128]  # SPC sensor resolution
N_tbins = 1024  # Number of discrete time bins for "ideal" transient
tmax = 100  # Laser period in nano seconds
FWHM = 1  # Laser full wave half maximum in terms of bin-width
N_pulses = 1000  # Number of laser cycles to use
alpha_sig = 1.0  # Average signal photons per laser cycle
alpha_bkg = 4.0  # Average background photons per laser cycle
device = "cpu"  # Torch device
idx = 0  # Set frame index

# Load RGB frame and depth map
_, rgb_img, _ = frames[idx]
_, depth_img, _ = depths[idx]

# Filter out depths that might be out-of-range
depth_img = depth_img[..., -1]
max_depth = tmax * 10e-9 * N_tbins * c / 2
depth_img = cv2.inpaint(depth_img, (depth_img > max_depth).astype(np.uint8), 3, cv2.INPAINT_TELEA)

# Resize and transform to tensor, scale RGB to [0-1] range
rgb_img = cv2.resize(rgb_img, (Nr, Nc))
rgb = torch.tensor(rgb_img).to(device) / 255.0
depth_img = cv2.resize(depth_img, (Nr, Nc))
depth = torch.tensor(depth_img).to(device)

# Using the red channel as albedo and intensity
albedo = intensity = rgb[..., 0]

tr_gen = TransientGenerator(Nr=Nr, Nc=Nc, N_tbins=N_tbins, tmax=tmax, FWHM=FWHM)

# Generate the ground-truth transient for each pixel
# given distance, albedo, intensity, and illumination condition
# NOTE: The true distance is in meters and depends on tmax
phi_bar = tr_gen.get_transient(
    depth,
    albedo,
    intensity,
    torch.tensor(alpha_sig),
    torch.tensor(alpha_bkg),
)

N_bins = 64  # Number of EWH SPC bins

# Simulating 64-bin EWH SPC output
spc = BaseEWHSPC(Nr, Nc, N_pulses, device, N_tbins, int(N_bins))
captured_data = spc.capture(phi_bar)
ewh_data = captured_data["ewh"]

# Estimate distance from EWH SPC measurements
ewh_postproc = PostProcEWH(Nr, Nc, N_tbins, tmax, device)
_, estimated_depth = ewh_postproc.ewh2depth_t(ewh_data)

# --- Create Figure and GridSpec Layout ---
# Initialize the figure
# Define a GridSpec with 2 rows and 3 columns
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

# choose row, col of sample pixel
row, col = [10, 10]

plt.style.use("seaborn-v0_8-poster")
# --- First Row: Display Images ---
# Subplot for the RGB image (first column)
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(rgb_img)
ax1.scatter(col, row, c="r", marker="x")
ax1.set_title("Scene Image")
ax1.axis("off")

# Subplot for the first depth map (second column)
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(depth_img, cmap="turbo", vmin=depth_img.min(), vmax=depth_img.max())
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
ax2.set_title("True Distance")
cbar2.set_label("Depth (m)")
ax2.axis("off")

# Subplot for the second depth map (third column)
ax3 = fig.add_subplot(gs[0, 2])
im3 = ax3.imshow(estimated_depth.cpu(), cmap="turbo", vmin=depth_img.min(), vmax=depth_img.max())
cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cbar3.set_label("Depth (m)")
ax3.axis("off")

# --- Second Row: Line Plot ---
# Create a subplot that spans all three columns in the second row
ax_line = fig.add_subplot(gs[1, :])
transient = phi_bar[row, col] * N_pulses * (alpha_bkg + alpha_sig)

ax3.set_title("Depth from %d-bin EWH" % N_bins)
ax_line.set_title("Equi-width histogram (EWH)")
ewh_bins_axis = torch.linspace(0, N_tbins - N_tbins // N_bins, N_bins)
plot_transient(
    ax_line,
    transient.cpu().numpy(),
    plt_type="-r",
    label="True Transient",
)
plot_ewh(ax_line, ewh_bins_axis, ewh_data[row, col].cpu().numpy(), label="EWH histogram", color="w")

ax_line.legend(frameon=False, fontsize="12", loc="upper right")
ax_line.set_xlabel("Discretized time (a.u.)")
ax_line.set_ylabel("Photon counts")
plt.tight_layout()
fig.savefig("ewh_spc_output.svg", dpi=350)
