from __future__ import annotations

import os
from dataclasses import dataclass

from typing_extensions import Literal

from visionsim.types import MemSize


@dataclass
class RenderConfig:
    executable: str | os.PathLike | None = None
    """Path to blender executable"""
    height: int = 512
    """Height of rendered frames"""
    width: int = 512
    """Width of rendered frames"""
    bit_depth: int = 8
    """Bit depth for intensity frames. Usually 8 for pngs, 32 or 16 bits for OPEN_EXR"""
    file_format: str = "PNG"
    """File format to use for intensity frames"""
    exr_codec: str = "ZIP"
    """Encoding used to compress EXRs, used for all supported ground truths"""
    depths: bool = False
    """If true, enable depth map outputs"""
    normals: bool = False
    """If true, enable normal map outputs"""
    flows: bool = False
    """If true, enable optical flow outputs"""
    flow_direction: Literal["forward", "backward", "both"] = "forward"
    """Direction of flow to colorize for debug visualization. Only used when debug is true"""
    segmentations: bool = False
    """If true, enable segmentation map outputs"""
    debug: bool = True
    """If true, also save debug visualizations for auxiliary outputs"""
    keyframe_multiplier: float = 1.0
    """Stretch keyframes by this amount, eg: 2.0 will slow down time"""
    timeout: int = -1
    """Maximum allowed time in seconds to wait to connect to render instance"""
    autoexec: bool = True
    """If true, allow python execution of embedded scripts (warning: potentially dangerous)"""
    device_type: Literal["cpu", "cuda", "optix", "metal"] = "optix"
    """Name of device to use, one of "cpu", "cuda", "optix", "metal", etc"""
    adaptive_threshold: float = 0.05
    """Noise threshold of rendered images, for higher quality frames make this threshold smaller. 
    The default value is intentionally a little high to speed up renders"""
    max_samples: int = 256
    """Maximum number of samples per pixel to take"""
    use_denoising: bool = True
    """If enabled, a denoising pass will be used"""
    log_dir: str | os.PathLike = "logs/"
    """Directory to use for logging"""
    allow_skips: bool = True
    """If true, skip rendering a frame if it already exists"""
    unbind_camera: bool = False
    """Free the camera from it's parents, any constraints and animations it may have. 
    Ensures it uses the world's coordinate frame and the provided camera trajectory"""
    use_animations: bool = True
    """Allow any animations to play out, if false, scene will be static"""
    use_motion_blur: bool | None = None
    """Enable realistic motion blur. cannot be used if also rendering optical flow"""
    addons: list[str] | None = None
    """List of extra addons to enable"""
    jobs: int = 1
    """Number of concurrent render jobs"""
    autoscale: bool = False
    """Set number of jobs automatically based on available VRAM and `max_job_vram` when enabled"""
    max_job_vram: MemSize | None = None
    """Maximum allowable VRAM per job in bytes (limit is not enforced, simply used for `autoscale`)"""
