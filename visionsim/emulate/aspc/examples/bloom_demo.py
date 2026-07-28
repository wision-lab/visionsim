import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import openexr_numpy as onp
import cv2
from visionsim.emulate.aspc.camera import Camera
from visionsim.emulate.aspc.examples.ascp_plot_utils import plot_spad_sensor_grid
from visionsim.emulate.aspc.utils import ureg
def get_kn2():
    histo = np.load("/u/g/u/gump/vsim/representative_waveform.npz")["arr"]

    m = np.argmax(histo[192//2,256//2,:])
    offset_total_kn = np.arange(-18,18) + m[...,None]

    kn_crop = histo[192//2,256//2,:]
    kn_crop = kn_crop[offset_total_kn]
    m = np.argmax(kn_crop)
    kn_crop[:m-5] = 0
    kn_crop[m+9:]=0

    return kn_crop

def cropDepth(depth_map,crop_window):
        depth_map = np.flipud(depth_map)
        x_min = crop_window[0]
        x_max = crop_window[1]
        y_min = crop_window[2]
        y_max = crop_window[3]
        return np.flipud(depth_map[y_min:y_max,x_min:x_max])

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



def load_data(scene_id):
    data_path = "/u/g/u/gump/vsim/stop_sign_v2"
    albedo = cv2.imread(os.path.join(data_path,f"albedo/albedo00000{scene_id}Image.png"), cv2.IMREAD_GRAYSCALE)
    depth = onp.read(os.path.join(data_path,f"depths/depth_00000{scene_id}Depth.exr"))["V"]


    data_path = "/u/g/u/gump/vsim/stop_sign_v2/materials"
    wall_front = onp.read(os.path.join(data_path,f"wall_front_00{scene_id}Material Index.exr"))["V"].astype(bool)
    stop_sign_base = onp.read(os.path.join(data_path,f"stop_sign_base_00{scene_id}Material Index.exr"))["V"].astype(bool)
    stop_sign_screws = onp.read(os.path.join(data_path,f"stop_sign_screws_00{scene_id}Material Index.exr"))["V"].astype(bool)
    stop_sign_pole = onp.read(os.path.join(data_path,f"stop_sign_pole_00{scene_id}Material Index.exr"))["V"].astype(bool)
    floor = onp.read(os.path.join(data_path,f"floor_00{scene_id}Material Index.exr"))["V"].astype(bool)
    stop_sign_face = onp.read(os.path.join(data_path,f"stop_sign_face_00{scene_id}Material Index.exr"))["V"].astype(bool)
    wall_back = onp.read(os.path.join(data_path,f"wall_back_00{scene_id}Material Index.exr"))["V"].astype(bool)
    mannequin = onp.read(os.path.join(data_path,f"mannequin_00{scene_id}Material Index.exr"))["V"].astype(bool)
    box = onp.read(os.path.join(data_path,f"box_00{scene_id}Material Index.exr"))["V"].astype(bool)
    stop_sign_back = onp.read(os.path.join(data_path,f"stop_sign_back_00{scene_id}Material Index.exr"))["V"].astype(bool)


    depth = np.array(depth, copy=True)
    depth = depth[None, ...]


    glare_mask = np.zeros(albedo.shape)
    albedo = np.zeros(albedo.shape)+.05
    albedo[stop_sign_back] = .05
    albedo[wall_front]=.15
    albedo[stop_sign_base] =.05
    albedo[stop_sign_screws] = .05
    albedo[stop_sign_pole] = .05
    albedo[floor] = .05
    albedo[stop_sign_face] = 10000 #10000 #1 #1500
    albedo[box] = .05
    albedo[wall_back] = .25
    albedo[mannequin] = .1
    
    albedo = albedo[None,...]
    albedo_frames = torch.from_numpy(albedo.copy()).to(device=device, dtype=torch.float64)
    depth_frames = torch.from_numpy(depth.copy()).to(device=device, dtype=torch.float64)

    depth_frames = ureg.Quantity(depth_frames, "meter")
    albedo_frames = ureg.Quantity(albedo_frames, "dimensionless")


  
    glare_mask[stop_sign_face] = 1 
    glare_mask = glare_mask[None,...]
    glare_mask_frames = torch.from_numpy(glare_mask.copy()).to(device=device, dtype=torch.bool)
    glare_mask_frames = ureg.Quantity(glare_mask_frames, "dimensionless")
   
    return albedo_frames,depth_frames,glare_mask_frames
     
    


def run_simulation_scenario(camera, new_config, output_dir, spad_grid_shape,raw_path,scene_id,glare_mask_frames):
    """Runs the simulation with the specific config object and saves results."""
    print(f"\n--- Running Scenario: {scene_id} ---")
    


    
    camera.reconfigure(new_config)
    sensor_height = camera.sensor.h
    sensor_width = camera.sensor.w
    
    albedo_frames, depth_frames = camera.albedo_frames, camera.depth_frames
    raw_histo = np.flipud(np.load(raw_path)["arr"])
    transients, ambient_offsets = camera.get_transients()
    
    arrival_rates = camera.get_arrival_rates(glare_mask_frames)
    print(arrival_rates.shape)
    
    print("Starting EWH simulation")
    ewh_list = camera.get_ewh()
    print("EWH simulation completed")

    
    # 5. Save Plot
    ewh_list = np.array([x.cpu().numpy() for x in ewh_list])
    output_path = os.path.join(output_dir, f"debug")
    os.makedirs(output_path,exist_ok=True)
    output_path = os.path.join(output_path,"image_debug.png")
    print(output_path)
    save_data = os.path.join(output_dir, f"raw_data")
    os.makedirs(save_data,exist_ok=True)
    ambient_offsets = np.array([x.cpu().numpy() for x in ambient_offsets])

    plot_spad_sensor_grid(
        camera.histogrammer,
        camera.get_fov_masks(),
        spad_grid_shape,
        albedo_frames[0],
        depth_frames[0].magnitude,
        transients.reshape((sensor_height,sensor_width,672)),
        arrival_rates,
        ewh_list.reshape((sensor_height,sensor_width,672)),
        save_path=output_path,  # Save instead of show
        raw_histo = raw_histo,

    )


if __name__ == "__main__":
    data_dir = Path(r"/u/g/u/gump/vsim/test_other_blend")
    config_path = "/u/g/u/gump/vsim/visionsim/visionsim/emulate/aspc/examples/config_blooml.yaml"
    device="cpu"
    #device = "cuda:1"

    camera = Camera(data_dir, config_path, device)
    scene_id = "0"
    
    output_dir = f"testing_bloom/{scene_id}/"
    os.makedirs(output_dir, exist_ok=True)
    
    
    raw_path = os.path.join("/u/g/u/gump/vsim/vsim_data_raw_histos",f"{scene_id}_raw.npz")
    albedo_frames,depth_frames,glare_mask_frames = load_data(scene_id)
    plt.imshow(glare_mask_frames[0,:,:])
    plt.savefig("/u/g/u/gump/vsim/testing_bloom/glare_mask.png")
    plt.close()
    camera.albedo_frames, camera.depth_frames = albedo_frames, depth_frames

    # Load npz data

    # --- AUTOMATIC FOV GENERATION ---
    sensor_h, sensor_w = camera.sensor.h, camera.sensor.w
    k_h, k_w = 1, 1
    s_h, s_w = 1, 1
    generated_fovs = generate_sliding_window_fovs((sensor_h, sensor_w), (k_h, k_w), (s_h, s_w))
    grid_rows = len(range(0, sensor_h - k_h + 1, s_h))
    grid_cols = len(range(0, sensor_w - k_w + 1, s_w))
    spad_grid_shape = (grid_rows, grid_cols)


    # -------------------------------------------------

    # --- DEFINE SCENARIOS ---
    BASE_PULSED = {"enabled": True}
    BASE_SUN = {"enabled": True}
    BASE_LIGHT_CONDITIONS = {"light_conditions": "STARLIGHT_WITHOUT_AIRGLOW"}
    BASE_HIST = {"pixel_fov_list": generated_fovs}
    cfg = {
        "active_source": {"pulsed_laser": {**BASE_PULSED}},
        "histogrammer": {**BASE_HIST},
        "ambient_source": {"sun": {**BASE_SUN, **BASE_LIGHT_CONDITIONS}},
    }
    run_simulation_scenario(camera, cfg,  output_dir, spad_grid_shape,raw_path=raw_path,scene_id=scene_id,glare_mask_frames=glare_mask_frames)

    print("All scenarios completed. Check the 'results_comparison' folder.")
