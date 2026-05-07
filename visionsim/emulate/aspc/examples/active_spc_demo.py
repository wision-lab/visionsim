import os
from pathlib import Path

import torch

from visionsim.emulate.aspc.camera import Camera
from visionsim.emulate.aspc.examples.ascp_plot_utils import plot_spad_sensor_grid
from visionsim.emulate.aspc.utils import ureg


def generate_sliding_window_fovs(image_shape, kernel_size, stride):
    """
    Generates a list of FOV coordinates [y_min, y_max, x_min, x_max]
    for sliding windows over the image.

    Args:
        image_shape (tuple): (height, width) of the sensor in pixels (e.g., [100, 100])
        kernel_size (tuple): (height, width) of the window in pixels (e.g., [10, 10])
        stride (tuple): (vertical_step, horizontal_step) in pixels

    Returns:
        list: List of [y_min, y_max, x_min, x_max] normalized (0.0-1.0).
    """
    H, W = image_shape
    kh, kw = kernel_size
    sh, sw = stride

    fov_list = []

    # Iterate over rows
    for y in range(0, H - kh + 1, sh):
        # Iterate over columns
        for x in range(0, W - kw + 1, sw):
            # Calculate pixel bounds
            y_min_px, y_max_px = y, y + kh
            x_min_px, x_max_px = x, x + kw

            # Normalize to 0.0 - 1.0
            norm_box = [
                y_min_px / H,  # y_min
                y_max_px / H,  # y_max
                x_min_px / W,  # x_min
                x_max_px / W,  # x_max
            ]
            fov_list.append(norm_box)

    print(f"Generated {len(fov_list)} FOV regions.")
    return fov_list


def run_simulation_scenario(camera, new_config, scenario_name, output_dir, spad_grid_shape):
    """Runs the simulation with the specific config object and saves results."""
    print(f"\n--- Running Scenario: {scenario_name} ---")

    # Apply scenario config to rebuild sources/sensor/histogrammer
    camera.reconfigure(new_config)

    # Get data
    albedo_frames, depth_frames = camera.albedo_frames, camera.depth_frames
    transients, _ = camera.get_transients()
    arrival_rates = camera.get_arrival_rates()
    ewh_list = camera.get_ewh()

    # Save Plot
    output_path = os.path.join(output_dir, f"{scenario_name}_reconstruction.png")
    plot_spad_sensor_grid(
        camera.histogrammer,
        camera.get_fov_masks(),
        spad_grid_shape,
        albedo_frames[0],
        depth_frames[0].magnitude,
        transients,
        arrival_rates,
        ewh_list,
        save_path=output_path,  # Save instead of show
    )


if __name__ == "__main__":
    data_dir = Path("examples/renders/scene1/")
    # config_path = "visionsim/emulate/aspc/examples/active_spc_demo.yaml"
    config_path = "visionsim/emulate/aspc/examples/active_spc_demo.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera = Camera(data_dir, config_path, device)


    # ######################################################################################

    # ### High spatial resolution and low temporal resolution settings used for plots

    # ######################################################################################

    # output_dir = "results_active_spc_demo_new_temporalquantized"
    # os.makedirs(output_dir, exist_ok=True)

    # # --- AUTOMATIC FOV GENERATION ---
    # sensor_h, sensor_w = camera.sensor.h, camera.sensor.w
    # k_h, k_w = 4, 4
    # s_h, s_w = 4, 4
    # generated_fovs = generate_sliding_window_fovs((sensor_h, sensor_w), (k_h, k_w), (s_h, s_w))
    # grid_rows = len(range(0, sensor_h - k_h + 1, s_h))
    # grid_cols = len(range(0, sensor_w - k_w + 1, s_w))
    # spad_grid_shape = (grid_rows, grid_cols)
    
    # # -------------------------------------------------

    # # --- DEFINE SCENARIOS ---
    # BASE_PULSED = {"enabled": True}
    # BASE_SUN = {"enabled": True}
    # BASE_LIGHT_CONDITIONS = {"light_conditions": "BRIGHT_SUNLIGHT"}
    # BASE_HIST = {"pixel_fov_list": generated_fovs}

    # cfg1 = {
    #     "active_source": {"pulsed_laser": {**BASE_PULSED, "avg_watts": 0.0000003 * ureg.watt}},
    #     "histogrammer": {**BASE_HIST, 
    #                      "n_pulses": 10000, 
    #                      "max_depth": 15.0*ureg.meters,
    #                      "n_bins":25,
    #                      "bin_width":0.6*ureg.meters,
    #                      "vignette": True,
    #                      "dead_time_s": 75*ureg.nanoseconds,
    #                      "free_running": False,
    #                      "fast_sim": True,
    #                      },
    #     "sensor": {
    #         "size": [256, 256],
    #         "pixel_pitch": 10.0*ureg.micrometers,
    #         "f_number": 1.4,
    #         "fov": [90.5*ureg.degree, 90.5*ureg.degree]
    #     },
    #     "ambient_source": {"sun": {**BASE_SUN, **BASE_LIGHT_CONDITIONS}},
    # }
    # run_simulation_scenario(camera, cfg1, "1_High_Fidelity", output_dir, spad_grid_shape)



    ################################################################

    ### High temporal and spatial resolution settings used for plots

    ################################################################

    output_dir = "results_active_spc_demo_new_high_spatiotemporal"
    os.makedirs(output_dir, exist_ok=True)

    # --- AUTOMATIC FOV GENERATION ---
    sensor_h, sensor_w = camera.sensor.h, camera.sensor.w
    k_h, k_w = 4, 4
    s_h, s_w = 4, 4
    generated_fovs = generate_sliding_window_fovs((sensor_h, sensor_w), (k_h, k_w), (s_h, s_w))
    grid_rows = len(range(0, sensor_h - k_h + 1, s_h))
    grid_cols = len(range(0, sensor_w - k_w + 1, s_w))
    spad_grid_shape = (grid_rows, grid_cols)
    
    # -------------------------------------------------

    # --- DEFINE SCENARIOS ---
    BASE_PULSED = {"enabled": True}
    BASE_SUN = {"enabled": True}
    BASE_LIGHT_CONDITIONS = {"light_conditions": "BRIGHT_SUNLIGHT"}
    BASE_HIST = {"pixel_fov_list": generated_fovs}

    cfg1 = {
        "active_source": {"pulsed_laser": {**BASE_PULSED, "avg_watts": 0.000000007 * ureg.watt}},
        # "active_source": {"pulsed_laser": {**BASE_PULSED, "avg_watts": 0.000007 * ureg.watt}}, #To avoid pileup increase the signal flux
        "histogrammer": {**BASE_HIST, 
                         "n_pulses": 10000, 
                         "max_depth": 15.0*ureg.meters,
                         "n_bins":500,
                         "bin_width":0.03*ureg.meters,
                         "vignette": True,
                         "dead_time_s": 0*ureg.nanoseconds,
                         "free_running": True,
                         "fast_sim": True,
                         },
        "sensor": {
            "size": [256, 256],
            "pixel_pitch": 10.0*ureg.micrometers,
            "f_number": 1.4,
            "fov": [90.5*ureg.degree, 90.5*ureg.degree]
        },
        "ambient_source": {"sun": {**BASE_SUN, **BASE_LIGHT_CONDITIONS}},
    }
    run_simulation_scenario(camera, cfg1, "1_High_Fidelity", output_dir, spad_grid_shape)

    # # Scenario 2: Photon Starved (Low SNR)
    # # Reduced laser power significantly. Depth estimation should become noisy/patchy.
    # # Traditional simulators would still show perfect depth here!
    # cfg2 = {
    #     "active_source": {"pulsed_laser": {**BASE_PULSED, "avg_watts": 0.0005 * ureg.watt}},
    #     "histogrammer": {**BASE_HIST, "n_pulses": 2000},
    #     "ambient_source": {"sun": {**BASE_SUN, **BASE_LIGHT_CONDITIONS}},
    # }
    # run_simulation_scenario(camera, cfg2, "2_Photon_Starved", output_dir, spad_grid_shape)

    # # Scenario 3: High Ambient Interference
    # # Standard laser power, but very intense background light.
    # # This floods the histogram with noise, burying the signal peak.
    # cfg3 = {
    #     "active_source": {"pulsed_laser": {**BASE_PULSED, "avg_watts": 0.01 * ureg.watt}},
    #     "histogrammer": {**BASE_HIST},
    #     "ambient_source": {
    #         "sun": {**BASE_SUN, **BASE_LIGHT_CONDITIONS, "intensity": 1.0e28 * ureg.watt / ureg.meter**2}
    #     },
    # }
    # # Alternatively, keep intensity same but increase aperture or exposure if sun scaling is weird physically
    # run_simulation_scenario(camera, cfg3, "3_High_Ambient", output_dir, spad_grid_shape)

    # # Scenario 4: ADAPS Demo
    # cfg4 = {
    #     "active_source": {"pulsed_laser": {**BASE_PULSED}},
    #     "histogrammer": {**BASE_HIST},
    #     "ambient_source": {"sun": {**BASE_SUN, **BASE_LIGHT_CONDITIONS}},
    # }
    # run_simulation_scenario(camera, cfg4, "ADAPS_demo", output_dir, spad_grid_shape)

    # print("All scenarios completed. Check the 'results_comparison' folder.")
