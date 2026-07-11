import functools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from visionsim.dataset import Dataset
from visionsim.emulate.rgb import emulate_rgb_from_sequence


def plot_images(images, title, nrows=1, ncols=None):
    """Plot images in a grid."""
    fig = plt.figure()

    if ncols is None:
        ncols = len(images)

    grid = ImageGrid(fig, (1, 1, 1), nrows_ncols=(nrows, ncols))
    fig.suptitle(title)

    for ax, img in zip(grid, images):
        ax.set_axis_off()
        ax.imshow(img)
    plt.show()


# DSLR cameras have large full well capacities, low read noise, and high bit depth
# These parameters are not from a specific camera, but are representative of DSLR sensors
emulate_dslr = functools.partial(
    emulate_rgb_from_sequence,
    readout_std=3.0,
    fwc=50000.0,
    adc_bitdepth=12,
    flux_gain=50000.0,
    sensor_gain=12.2,
)

# Smartphone cameras have small full well capacities, higher read noise, and lower bit depth
# These parameters are not from a specific camera, but are representative of smartphone sensors
# The `sensor_gain` is set such that the full output range of the ADC is used (~FWC / 2^ADC_bitdepth)
emulate_smartphone = functools.partial(
    emulate_rgb_from_sequence,
    readout_std=4.0,
    fwc=5000.0,
    adc_bitdepth=10,
    flux_gain=5000.0,
    sensor_gain=4.9,
)

# Pre-load 640 frames, normalized to [0, 1] range
# Note: this requires a lot of memory, in practice its better to load them in a streaming manner.
dataset = Dataset.from_path("cache/quickstart/lego-interp/")
imgs = np.array(dataset[:640][0]) / 255.0

# Emulate RGB camera by varying exposure, with no read noise
# Since the interpolated input is ~4kHz, these will correspond to 100fps, 25fps, and 6.25fps.
blur_dslr = [emulate_dslr(imgs[:n], readout_std=0) for n in (40, 160, 640)]
blur_sp = [emulate_smartphone(imgs[:n], readout_std=0) for n in (40, 160, 640)]
plot_images(blur_dslr + blur_sp, "Varying Exposure (Top: DSLR, Bottom: Smartphone)", nrows=2, ncols=3)

# Emulate RGB camera by varying read noise, with fixed blur/exposure
noise_dslr = [emulate_dslr(imgs[:80], readout_std=n) for n in (160, 80, 40)]
noise_sp = [emulate_smartphone(imgs[:80], readout_std=n) for n in (160, 80, 40)]
plot_images(noise_dslr + noise_sp, "Varying Noise (Top: DSLR, Bottom: Smartphone)", nrows=2, ncols=3)

# Emulate RGB camera by varying read noise, in low light
# We artificially lower the flux gain (by 16x from it's default value) to simulate low light conditions
# and compensate by decreasing the sensor gain (inversely proportional to ISO) by the same factor
low_light_dslr = [emulate_dslr(imgs[:80], flux_gain=50000/16, sensor_gain=12.2/16, readout_std=n) for n in (160, 80, 40)]
low_light_sp = [emulate_smartphone(imgs[:80], flux_gain=5000/16, sensor_gain=4.9/16, readout_std=n) for n in (160, 80, 40)]
plot_images(low_light_dslr + low_light_sp, "Varying Noise in Low Light (Top: DSLR, Bottom: Smartphone)", nrows=2, ncols=3)

# Emulate Mosaicing and Demosaicing
demosaic_methods = ("off", "bilinear", "MHC04")
mosaic_dslr = [emulate_dslr(imgs[:80], mosaic=True, demosaic=m) for m in demosaic_methods]
mosaic_sp = [emulate_smartphone(imgs[:80], mosaic=True, demosaic=m) for m in demosaic_methods]
plot_images(mosaic_dslr + mosaic_sp, "Demosaicing (Raw | Bilinear | MHC04)", nrows=2, ncols=3)

# ISP Post-processing on a noisy low-light image
isp_dslr = [
    emulate_dslr(imgs[:80], readout_std=40),
    emulate_dslr(imgs[:80], readout_std=40, denoise_sigma=1.0),
    emulate_dslr(imgs[:80], readout_std=40, denoise_sigma=1.0, sharpen_weight=1.5)
]
isp_sp = [
    emulate_smartphone(imgs[:80], readout_std=40),
    emulate_smartphone(imgs[:80], readout_std=40, denoise_sigma=1.0),
    emulate_smartphone(imgs[:80], readout_std=40, denoise_sigma=1.0, sharpen_weight=1.5)
]
plot_images(isp_dslr + isp_sp, "ISP Processing (No Filter | Denoised | Denoised+Sharpened)", nrows=2, ncols=3)

# Emulate RGB camera by varying shutter fraction (duty cycle)
# Passing the exact same 160 frames, but changing the fraction of time the shutter is open
shutter_dslr = [emulate_dslr(imgs[:160], shutter_frac=f, readout_std=0) for f in (0.1, 0.5, 1.0)]
shutter_sp = [emulate_smartphone(imgs[:160], shutter_frac=f, readout_std=0) for f in (0.1, 0.5, 1.0)]
plot_images(shutter_dslr + shutter_sp, "Varying Shutter Fraction (10% | 50% | 100%)", nrows=2, ncols=3)
