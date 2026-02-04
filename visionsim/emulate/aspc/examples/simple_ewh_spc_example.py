import math
from pathlib import Path

import numpy as np
import torch
from ruamel.yaml import YAML

from visionsim.emulate.aspc.examples.ascp_plot_utils import plot_ewh_per_pixel
from visionsim.emulate.aspc.histogrammers import HistConfig, Histogrammer
from visionsim.emulate.aspc.sensors import SPADSensor
from visionsim.emulate.aspc.sources import PulsedLaser, Sun, get_light_conditions_from_string
from visionsim.emulate.aspc.utils import (
    eval_constructor,
    irradiance_photons,
    tof2depth,
    ureg,
    ureg_constructor,
)

if __name__ == "__main__":
    ## Setting simulation parameters

    root = Path("examples/renders/scene1/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    requires_grad = True

    config_path = "visionsim/emulate/aspc/examples/config_ewh2.yaml"

    # Load config
    yaml = YAML()
    safe_builtins = {"__builtins__": {"list": list, "dict": dict, "tuple": tuple}, "np": np, "math": math}
    yaml.Constructor.add_constructor(tag="!Quantity", constructor=ureg_constructor(ureg.Quantity))
    yaml.Constructor.add_constructor(tag="!expr", constructor=eval_constructor(eval, safe_builtins))
    config = yaml.load(open(config_path))

    # Extract the rows, columns of pixels

    Nr = config["histogrammer"]["shape"][0]
    Nc = config["histogrammer"]["shape"][1]

    albedo_frames = torch.ones((1, Nr, Nc), dtype=torch.double, device=device, requires_grad=requires_grad)
    intensity_frames = torch.ones((1, Nr, Nc), dtype=torch.double, device=device, requires_grad=requires_grad)
    depth_frames = torch.ones((1, Nr, Nc), dtype=torch.double, device=device, requires_grad=requires_grad) * ureg.meter

    # Hardcoded depth for toy example with 4 pixel FOVs focusing of 1m, 2m, 3m, 4m targets
    depth_frames[0, 0, 0] *= 1
    depth_frames[0, 0, 1] *= 2
    depth_frames[0, 0, 2] *= 3
    depth_frames[0, 0, 3] *= 4

    # Active source
    active_config = config["active_source"]["pulsed_laser"]
    active_enable = active_config.pop("enabled")
    if active_enable:
        active_source = PulsedLaser(**active_config)

    # Ambient source
    ambient_config = config["ambient_source"]["sun"]
    ambient_config["light_conditions"] = get_light_conditions_from_string(ambient_config["light_conditions"])
    ambient_enable = ambient_config.pop("enabled")
    if ambient_enable:
        ambient_source = Sun(**ambient_config)

    # Histogrammer
    hist_config = config["histogrammer"]
    vignette = hist_config["vignette"]
    hist_config = HistConfig(**hist_config)
    histogrammer = Histogrammer(hist_config)

    # FOV masks
    _, img_rows, img_cols = depth_frames.shape
    empty_mask = torch.zeros((img_rows, img_cols), dtype=bool, device=device)
    fov_masks = histogrammer.get_perpixel_fov_masks(
        empty_mask, hist_config.pixel_fov_list, device=device, vignette=vignette
    )

    # Sensor
    sensor_config = config["sensor"]
    sensor = SPADSensor(**sensor_config)

    # Get transients
    sensor_config = config["sensor"]
    num_pixels = sensor_config["size"][0] * sensor_config["size"][1]

    # Convert tensors to Pint quantities for active source
    albedo_quantity = albedo_frames * ureg.dimensionless
    depth_quantity = depth_frames

    radiance = active_source.get_scene_radiance(albedo_quantity, depth_quantity, num_pixels, sensor.omega)

    # --- Verify  ---
    print("\n" + "=" * 20 + " DEBUG GEOMETRY " + "=" * 20)
    # 1. Check pixel area in meters^2
    pixel_area = sensor.pixel_pitch.to(ureg.meter).magnitude ** 2
    print(f"Pixel Area (m^2):      {pixel_area:.2e}")

    # 2. Check Geometric Loss (1/F#^2)
    f_num_loss = (1 / sensor.f_number) ** 2
    print(f"Lens Loss (1/F#^2):    {f_num_loss:.2e}")

    # 3. Check Radiance magnitude (Watts/sr/m^2)
    # Safely handle Pint Quantity wrapping a CUDA Tensor
    if hasattr(radiance, "magnitude"):
        # Extract tensor from Pint -> Move to CPU -> Convert to float
        rad_val = radiance.magnitude.mean().cpu().item()
    else:
        # It's just a raw tensor
        rad_val = radiance.mean().cpu().item()

    print(f"Radiance (Mean):       {rad_val:.2e}")
    print("=" * 56 + "\n")

    # --------------------------------------------------

    irradiance = (radiance * torch.pi / 4 * (1 / sensor.f_number) ** 2).to(irradiance_photons) * (
        sensor.pixel_pitch.to(ureg.meter)
    ) ** 2
    irradiance_tensor = irradiance.magnitude

    # Get ambient offset
    ambient_radiance = ambient_source.get_scene_radiance(sensor.omega, albedo_quantity, active_source.frequency)
    ambient_irradiance = (ambient_radiance * torch.pi / 4 * (1 / sensor.f_number) ** 2).to(irradiance_photons) * (
        sensor.pixel_pitch.to(ureg.meter)
    ) ** 2
    offset = ambient_irradiance.magnitude

    # --- VERIFICATION BLOCK ---
    print("\n" + "=" * 40)
    print("RADIOMETRIC VERIFICATION")
    print("=" * 40)

    # 1. Signal is ALREADY Per Pulse (because sources.py handled the frequency division)
    photons_per_pulse = irradiance_tensor.mean()
    print(f"Signal Strength:   {photons_per_pulse:.4f} photons/PULSE")

    # 2. Calculate Flux for reference (Signal * Frequency)
    freq_hz = active_source.frequency.to(ureg.hertz).magnitude
    photons_per_sec = photons_per_pulse * freq_hz
    print(f"Signal Flux:       {photons_per_sec:.2e} photons/sec")

    # 3. Check Linear Regime
    if photons_per_pulse > 0.1:
        print("[WARNING] SATURATION: > 0.1 photons/pulse (Pile-up dominant)")
    elif photons_per_pulse > 0.05:
        print("[CAUTION] NON-LINEAR: 0.05 - 0.1 photons/pulse (Compression)")
    else:
        print("[OK] LINEAR REGIME: < 0.05 photons/pulse")

    # 4. Ambient Check
    # Ambient is also calculated per integration window (1/freq) in sources.py
    ambient_per_pulse = np.mean(offset) if not torch.is_tensor(offset) else offset.mean().item()
    print(f"Ambient Noise:     {ambient_per_pulse:.2e} photons/PULSE")

    # 5. SBR
    sbr = photons_per_pulse / (ambient_per_pulse + 1e-15)
    print(f"SBR:               {sbr:.2f}")
    print("=" * 40 + "\n")

    print("\n" + "=" * 20 + " ROI FLUX CHECK " + "=" * 20)

    # FIX: Remove the batch dimension [1, 100, 100] -> [100, 100]
    # If using torch:
    if torch.is_tensor(irradiance_tensor):
        irradiance_2d = irradiance_tensor.squeeze()
    else:
        # If numpy
        irradiance_2d = irradiance_tensor.squeeze()

    for i, mask in enumerate(fov_masks):
        mask_bool = mask.bool()

        num_pixels_in_roi = mask_bool.sum().item()
        print(f"ROI {i}: Accumulating {num_pixels_in_roi} pixels")

        if num_pixels_in_roi > 0:
            # FIX: Apply mask to the 2D version
            region_flux = irradiance_2d[mask_bool].mean().item()
            total_histogram_counts = region_flux * num_pixels_in_roi

            print(f"   -> Flux per pixel: {region_flux:.4f}")
            print(f"   -> Histogram Height: ~{total_histogram_counts:.1f} photons/pulse")
        else:
            print("   -> ROI is empty")

    print("=" * 56 + "\n")

    # --- VERIFICATION BLOCK END ---

    transients, ambient_offsets = histogrammer.calculate_transients(
        irradiance_tensor,
        depth_frames.magnitude,
        offset,
        fov_masks,
        histogrammer.n_bins,
        active_source.max_resolvable_depth.magnitude,
    )
    print("transients.shape: ", transients.shape)

    # Calculate arrival rates
    bin_width = 2 * tof2depth(1 / active_source.frequency) / histogrammer.n_bins
    _, irf = active_source.get_kernel(bin_width)
    irf_tensor = torch.tensor(irf, dtype=float, device=device)
    arrival_rates = histogrammer.calculate_arrival_rates(irf_tensor, transients, ambient_offsets, histogrammer.n_bins)

    active_source.plot_kernel(bin_width)

    dead_time_bins = int(histogrammer.dead_time_s * histogrammer.n_bins * active_source.frequency)

    # Simulate EWH with dead time
    ewh_list = histogrammer.simulate_ewh(
        arrival_rates,
        histogrammer.n_pulses,
        histogrammer.n_bins,
        histogrammer.free_running,
        dead_time_bins,
    )

    plot_ewh_per_pixel(
        histogrammer,
        fov_masks,
        albedo_frames[0],
        depth_frames[0].magnitude,
        transients,
        arrival_rates,
        ewh_list,
    )
