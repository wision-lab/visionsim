from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pint import UnitRegistry, set_application_registry
from ruamel.yaml.nodes import ScalarNode, SequenceNode
from scipy.constants import c, h

from visionsim.dataset import Dataset

ureg = UnitRegistry()
ureg.setup_matplotlib(True)
Q_ = ureg.Quantity
set_application_registry(ureg)

radiance = ureg.watts / ureg.steradian / ureg.meter**2
irradiance = ureg.watts / ureg.meter**2

radiance_photons = ureg.count / ureg.steradian / ureg.meter**2
irradiance_photons = ureg.count / ureg.meter**2

#############################################
#              Physical laws                #
#############################################


@ureg.wraps(ureg.meter, ureg.second)
def tof2depth(t):
    return c * t / 2


@ureg.wraps(ureg.count, (ureg.watt, ureg.second, ureg.meter))
def watts2photons(watts, t, wavelength):
    energy = watts * t
    photon_energy = h * c / wavelength
    return energy / photon_energy


#############################################
#          Optics Modeling Utils            #
#############################################


@ureg.wraps(ureg.radian, (ureg.meter, ureg.meter))
def fov_from_focal_length(fl, d):
    """Convert from the focal length to angular FOV
    See: https://en.wikipedia.org/wiki/Angle_of_view

    Args:
        fl: Focal length, either per pixel or total
        d: Image sensor dimension. If focal length is
        f_x/f_y then d should be the pixel pitch. If the
        focal length is for the whole imaging system (i.e
        it refers to the diagonal) then d should be the
        image sensor's diagonal length.

    Returns:
        Field of view in radians
    """
    return 2 * np.arctan2(d, 2 * fl)


@ureg.wraps(ureg.meter, (ureg.radian, ureg.meter))
def focal_length_from_fov(fov, d):
    """
    Convert from the angular FOV to the focal length,
    See `fov_from_focal_length` for details.
    """
    # Here FOV is twice the apex angle
    return d / (2 * np.tan(fov / 2))


@ureg.wraps(ureg.steradian, (ureg.radian, ureg.radian))
def pyramid_solid_angle(a, b):
    """
    Solid angle subtended by a four-sided right rectangular pyramid with apex angles a and b
    See: https://en.wikipedia.org/wiki/Solid_angle#Pyramid
    """
    return 4 * np.arcsin(np.sin(a / 2) * np.sin(b / 2))


def resize_like(src, target):
    return resize_to(src, target.shape)


def resize_to(img, shape):
    """Reshape image to shape, use cv2.INTER_AREA if we are shrinking, else cv2.INTER_CUBIC"""
    h, w, *_ = shape
    img_h, img_w, *_ = img.shape
    interp = cv2.INTER_AREA if img_h * img_w >= h * w else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)


#############################################
#              Data loader utils            #
#############################################


def preproc_albedo_intensity_depth_frames(
    root: str,
    device: str,
    config: dict,
    start_idx: int,
    num_frames: Optional[int] = 1,
    requires_grad: Optional[bool] = False,
):  # -> List[Tensor]:
    """Function to convert the rgb and depth frames read using data loaders to tensors"

    Args:
        root (str): Path to the root folder containing the rendered RGB images and depth maps
        device (str): Choose the compute device, cpu or cuda device
        start_idx (int): Index of the first rgb-d to be used generate active SPC frame
        num_frames (int): Number of rgb-d frames used to generate an active SPC frame
        config (dict): Dictionary containing all the simulation parameters
        requires_grad (bool): Set to True to enable gradient computations for differentiable pipelines
    Returns:
        List[Tensor]: Tensors corresponding to albedo, intensity and depth frames
    """

    frames = Dataset.from_path(root / "frames")
    depths = Dataset.from_path(root / "depths")
    assert len(depths) == len(frames), "Different number of depth and RGB frames"
    assert start_idx + num_frames <= len(frames), "start_idx + num_frames must not exceed total rendered frames"

    # Get config parameters
    Nr, Nc = config["sensor"]["size"]
    tmax = (1.0 / config["active_source"]["pulsed_laser"]["frequency"]).to(ureg.second)

    # Compute max depth
    max_depth = (tmax * ((c) * ureg.meter / ureg.second) / 2) / ureg.meter

    # print("Max depth in meters",max_depth)
    # print("c",c)
    # print("tmax", tmax)

    # tmax = 100  # Laser period in nano seconds

    rgb_img_list = list(frames[start_idx : start_idx + num_frames][1])
    depth_img_list = list(depths[start_idx : start_idx + num_frames][1])

    # print("Inside rgb_imgs.shape",len(rgb_img_list))
    # print("Inside depth_imgs.shape",len(depth_img_list))

    albedo_frames_list = []
    intensity_frames_list = []
    depth_frames_list = []

    # # Testing depth image
    import os

    import imageio.v3 as iio

    print("root", root)
    depth_dir = os.path.join(root, "depths")
    print("depth_dir", depth_dir)
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.lower().endswith((".png", ".hdr"))])
    depth_img = iio.imread(os.path.join(depth_dir, depth_files[0]))
    print(depth_img)
    print("depth_img", depth_img.shape)
    plt.figure()
    plt.imshow(depth_img)
    plt.colorbar(label="Depth")
    plt.show()

    for idx in range(len(rgb_img_list)):
        rgb_img = rgb_img_list[idx]
        if depth_img_list[idx].ndim == 3:
            # print("at ndim depth: ", depth_img_list[idx].shape, depth_img_list[idx].min(), depth_img_list[idx].max(), depth_img_list[idx].mean(),depth_img_list[idx].std())
            depth_img = depth_img_list[idx][:, :, 0]
        else:
            depth_img = depth_img_list[idx] / 255.0 * config["histogrammer"]["max_depth"]

        # Diagnose depth image
        # iio
        # depth_img = iio.imread(os.path.join(depth_dir, depth_files[idx]))
        # depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
        # cv2
        # depth_img = cv2.imread(os.path.join(depth_dir, depth_files[idx]), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        # depth_img = np.clip(depth_img, 0, max_depth)
        # depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
        # depth_img = (depth_img * 255).astype(np.uint8)

        # print("before inpaint depth: ", depth_img.shape, depth_img.min(), depth_img.max(), depth_img.mean(),depth_img.std())

        # Filter out depths that might be out-of-range
        depth_img = cv2.inpaint(
            depth_img.astype(np.float32), (depth_img > max_depth).astype(np.uint8), 3, cv2.INPAINT_TELEA
        )

        # print("after inpaint depth: ", depth_img.shape, depth_img.min(), depth_img.max(), depth_img.mean(),depth_img.std())

        # Display depth image
        plt.figure()
        plt.title(f"Depth Image {idx}")
        plt.imshow(depth_img)
        plt.colorbar(label="Depth")
        plt.show()

        # Resize and transform to tensor, scale RGB to [0-1] range
        rgb_img = cv2.resize(rgb_img, (Nc, Nr))
        rgb = torch.tensor(rgb_img.astype(float), device=device, requires_grad=requires_grad) / 255.0
        depth_img = cv2.resize(depth_img, (Nc, Nr))
        depth = torch.tensor(depth_img.astype(float), device=device, requires_grad=requires_grad).unsqueeze(0)

        # Using the red channel as albedo and intensity
        albedo = intensity = rgb[..., 0].unsqueeze(0)

        # print("Idx: ",idx, "depth: ", depth.shape, depth.min(), depth.max(), depth.mean(), depth.median(),depth.std())
        # print("Idx: ",idx, "albedo: ", albedo.shape, albedo.min(), albedo.max(), albedo.mean())
        # print("Idx: ",idx, "intensity: ", intensity.shape, intensity.min(), intensity.max(), intensity.mean())

        albedo_frames_list.append(albedo)
        intensity_frames_list.append(intensity)
        depth_frames_list.append(depth)

    albedo_frames = torch.concat(albedo_frames_list)
    intensity_frames = torch.concat(intensity_frames_list)
    depth_frames = torch.concat(depth_frames_list) * ureg.meter

    return albedo_frames, intensity_frames, depth_frames


# def preproc_albedo_intensity_depth_frames(root: str,
#                                           device: str,
#                                           config: dict,
#                                           start_idx: int,
#                                           num_frames: Optional[int] = 1,
#                                           requires_grad: Optional[bool] = False):# -> List[Tensor]:
#     """Function to convert the rgb and depth frames read using data loaders to tensors"
#     Args:
#         root (str): Path to the root folder containing the rendered RGB images and depth maps
#         device (str): Choose the compute device, cpu or cuda device
#         start_idx (int): Index of the first rgb-d to be used generate active SPC frame
#         num_frames (int): Number of rgb-d frames used to generate an active SPC frame
#         config (dict): Dictionary containing all the simulation parameters
#         requires_grad (bool): Set to True to enable gradient computations for differentiable pipelines
#     Returns:
#         List[Tensor]: Tensors corresponding to albedo, intensity and depth frames
#     """
#     frames = Dataset.from_path(root / "frames")
#     depths = Dataset.from_path(root / "depths")
#     assert len(depths) == len(frames), "Different number of depth and RGB frames"
#     assert start_idx + num_frames < len(frames), "start_idx + num_frames must not exceed total rendered frames"
#     # Get config parameters
#     Nr,Nc = config["sensor"]["size"]
#     tmax = (1.0/config['active_source']['pulsed_laser']['frequency']).to(ureg.second)
#     # tmax = float(1.0/float(config['active_source']['pulsed_laser']['frequency']))*(ureg.second)
#     # Compute max depth
#     max_depth = ((tmax*((c)*ureg.meter/ureg.second)/2)*1000)/ureg.meter
#     print("Max depth in meters",max_depth)
#     print("c",c)
#     print("tmax", tmax)
#     # tmax = 100  # Laser period in nano seconds
#     rgb_img_list = list(frames[start_idx : start_idx + num_frames][1])
#     depth_img_list = list(depths[start_idx : start_idx + num_frames][1])
#     print("Inside rgb_imgs.shape",len(rgb_img_list))
#     print("Inside depth_imgs.shape",len(depth_img_list))
#     albedo_frames_list = []
#     intensity_frames_list = []
#     depth_frames_list = []
#     for idx in range(len(rgb_img_list)):
#         rgb_img = rgb_img_list[idx]
#         depth_img = depth_img_list[idx][:,:,0]
#         print("Idx: ",idx, "depth: ", depth_img.shape, depth_img.min(), depth_img.max(), depth_img.mean())
#         depth_img = cv2.inpaint(depth_img, (depth_img > max_depth).astype(np.uint8), 3, cv2.INPAINT_TELEA)
#         print("Idx: ",idx, "depth: ", depth_img.shape, depth_img.min(), depth_img.max(), depth_img.mean())
#         # Resize and transform to tensor, scale RGB to [0-1] range
#         rgb_img = cv2.resize(rgb_img, (Nc, Nr))
#         rgb = torch.tensor(rgb_img.astype(float), device=device, requires_grad = requires_grad) / 255.0
#         depth_img = cv2.resize(depth_img, (Nc, Nr))
#         # Filter out depths that might be out-of-range
#         depth = torch.tensor(depth_img, device=device, requires_grad = requires_grad).unsqueeze(0)
#         # Using the red channel as albedo and intensity
#         albedo = intensity = rgb[..., 0].unsqueeze(0)
#         # print("Idx: ",idx, "depth: ", depth.shape, depth.min(), depth.max(), depth.mean())
#         # print("Idx: ",idx, "albedo: ", albedo.shape, albedo.min(), albedo.max(), albedo.mean())
#         # print("Idx: ",idx, "intensity: ", intensity.shape, intensity.min(), intensity.max(), intensity.mean())
#         albedo_frames_list.append(albedo)
#         intensity_frames_list.append(intensity)
#         depth_frames_list.append(depth)
#     albedo_frames = torch.concat(albedo_frames_list)
#     intensity_frames = torch.concat(intensity_frames_list)
#     depth_frames = torch.concat(depth_frames_list)
#     return albedo_frames, intensity_frames, depth_frames

#############################################
#              Helper utils                 #
#############################################


def ureg_constructor(cls):
    def ureg_yaml(loader, node):
        if isinstance(node, ScalarNode):
            value = loader.construct_scalar(node)
            return cls(value) if value else cls()
        elif isinstance(node, SequenceNode):
            value = loader.construct_sequence(node)
            return [cls(v) for v in value]
        else:
            raise NotImplementedError

    return ureg_yaml


def eval_constructor(cls, safe_builtins=None):
    def eval_yaml(loader, node):
        if isinstance(node, ScalarNode):
            value = loader.construct_scalar(node)
            return cls(value, safe_builtins)
        else:
            raise NotImplementedError

    return eval_yaml


def file_constructor(cls):
    def file_yaml(loader, node):
        if isinstance(node, ScalarNode):
            value = loader.construct_scalar(node)
            return np.loadtxt(value)
        else:
            raise NotImplementedError

    return file_yaml


def get_irradiance_with_fov(irradiance, sensor_fov, pixel_fov, omega, w, h):
    fov_x, fov_y = sensor_fov[0], sensor_fov[1]
    w_ratio, h_ratio = pixel_fov[3] - pixel_fov[2], pixel_fov[1] - pixel_fov[0]
    w_new, h_new = w * w_ratio, h * h_ratio
    fov_x, fov_y = fov_x * w_ratio, fov_y * h_ratio
    omega_new = pyramid_solid_angle(fov_x, fov_y) / (w_new * h_new)
    return irradiance * omega_new / omega
