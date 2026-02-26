from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from visionsim.types import EXR_CODECS, FILE_FORMATS, MemSize


@dataclass
class CompositesConfig:
    """Same as :meth:`FramesConfig`, except with all defaults set to ``None`` such that they are inherited from the blendfile.
    For more information see :meth:`include_composites <visionsim.simulate.blender.BlenderService.exposed_include_composites>`."""

    file_format: FILE_FORMATS | None = None
    """File format used to save composited frames"""
    color_mode: Literal["BW", "RGB", "RGBA"] | None = None
    """Mode to save composited frames in: grayscale, color or color+alpha"""
    exr_codec: EXR_CODECS | None = None
    """Encoding used to compress EXRs, only used when ``file_format='OPEN_EXR'``"""
    bit_depth: int | None = None
    """Bit depth for intensity frames. Usually 8 for pngs, 32 or 16 bits for OPEN_EXR"""


@dataclass
class FramesConfig:
    """Same as :meth:`CompositesConfig`, except with sensible defaults.
    For more information see :meth:`include_frames <visionsim.simulate.blender.BlenderService.exposed_include_frames>`."""

    file_format: FILE_FORMATS = "PNG"
    """File format used to save ground truth frames"""
    color_mode: Literal["BW", "RGB", "RGBA"] = "RGB"
    """Mode to save ground truth frames in: grayscale, color or color+alpha"""
    exr_codec: EXR_CODECS = "DWAA"
    """Encoding used to compress EXRs, only used when ``file_format='OPEN_EXR'``"""
    bit_depth: Literal[8, 16, 32] = 8
    """Bit depth for intensity frames. Usually 8 for pngs, 32 or 16 bits for OPEN_EXR"""


@dataclass
class DepthsConfig:
    """For more information see :meth:`include_depths <visionsim.simulate.blender.BlenderService.exposed_include_depths>`."""

    preview: bool = True
    """Also save colorized depth maps as PNGs"""
    file_format: Literal["OPEN_EXR", "HDR"] = "OPEN_EXR"
    """File format used to save depth maps"""
    exr_codec: EXR_CODECS = "DWAA"
    """Encoding used to compress EXRs, only used when ``file_format='OPEN_EXR'``"""
    bit_depth: Literal[16, 32] = 32
    """Bit depth used for saving depth maps. Usually 32 or 16 bits for OPEN_EXR and 32 for HDR"""


@dataclass
class NormalsConfig:
    """For more information see :meth:`include_normals <visionsim.simulate.blender.BlenderService.exposed_include_normals>`."""

    preview: bool = True
    """Also save colorized normal maps as PNGs"""
    exr_codec: EXR_CODECS = "DWAA"
    """Encoding used to compress EXRs"""
    bit_depth: Literal[16, 32] = 32
    """Bit depth used for saving normal maps"""


@dataclass
class FlowsConfig:
    """For more information see :meth:`include_flows <visionsim.simulate.blender.BlenderService.exposed_include_flows>`."""

    preview: bool = True
    """Also save colorized flow maps as PNGs"""
    direction: Literal["forward", "backward", "both"] = "forward"
    """Direction of flow to colorize for preview visualization. Only used when ``preview`` is true"""
    exr_codec: EXR_CODECS = "DWAA"
    """Encoding used to compress EXRs"""
    bit_depth: Literal[16, 32] = 32
    """Bit depth used for saving flow maps"""


@dataclass
class SegmentationsConfig:
    """For more information see :meth:`include_segmentations <visionsim.simulate.blender.BlenderService.exposed_include_segmentations>`."""

    preview: bool = True
    """Also save colorized segmentation maps as PNGs"""
    shuffle: bool = True
    """Shuffle preview colors, helps differentiate object instances"""
    seed: int = 1234
    """Random seed used when shuffling colors"""
    exr_codec: EXR_CODECS = "DWAA"
    """Encoding used to compress EXRs"""
    bit_depth: Literal[16, 32] = 32
    """Bit depth used for saving segmentation maps"""


@dataclass
class MaterialsConfig:
    """For more information see :meth:`include_materials <visionsim.simulate.blender.BlenderService.exposed_include_materials>`."""

    preview: bool = True
    """Also save colorized material passes as PNGs"""
    shuffle: bool = True
    """Shuffle preview colors, helps differentiate material instances"""
    seed: int = 1234
    """Random seed used when shuffling colors"""
    exr_codec: EXR_CODECS = "DWAA"
    """Encoding used to compress EXRs"""
    bit_depth: Literal[16, 32] = 32
    """Bit depth used for saving material maps"""


@dataclass
class RenderConfig:
    executable: Path | None = None
    """Path to blender executable"""
    height: int | None = None
    """Height of rendered frames"""
    width: int | None = None
    """Width of rendered frames"""
    include_composites: bool = False
    """If true, enable composited outputs"""
    composites: CompositesConfig = field(default_factory=CompositesConfig)
    """Composited frames configuration options"""
    include_frames: bool = True
    """If true, enable ground truth frame outputs"""
    frames: FramesConfig = field(default_factory=FramesConfig)
    """Ground truth frames configuration options"""
    include_depths: bool = False
    """If true, enable depth map outputs"""
    depths: DepthsConfig = field(default_factory=DepthsConfig)
    """Depth maps configuration options"""
    include_normals: bool = False
    """If true, enable normal map outputs"""
    normals: NormalsConfig = field(default_factory=NormalsConfig)
    """Normal maps configuration options"""
    include_flows: bool = False
    """If true, enable optical flow outputs"""
    flows: FlowsConfig = field(default_factory=FlowsConfig)
    """Optical flow configuration options"""
    include_segmentations: bool = False
    """If true, enable segmentation map outputs"""
    segmentations: SegmentationsConfig = field(default_factory=SegmentationsConfig)
    """Segmentation maps configuration options"""
    include_materials: bool = False
    """If true, enable material map outputs"""
    materials: MaterialsConfig = field(default_factory=MaterialsConfig)
    """Material maps configuration options"""
    previews: bool = True
    """If false, disable all preview visualizations of auxiliary outputs"""
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
    log_dir: Path = Path("logs/")
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

    def __post_init__(self):
        # Note: Using post init with tyro is not best practice, as it will be called multiple
        #   times. However here we are just propagating values of aliases, so it should be ok.
        # See: https://brentyi.github.io/tyro/examples/overriding_configs/#dataclasses-defaults
        self.depths.preview &= self.previews
        self.normals.preview &= self.previews
        self.flows.preview &= self.previews
        self.segmentations.preview &= self.previews
        self.materials.preview &= self.previews
