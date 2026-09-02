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
class DiffusePassConfig:
    """For more information see :meth:`include_diffuse_pass <visionsim.simulate.blender.BlenderService.exposed_include_diffuse_pass>`."""

    file_format: FILE_FORMATS = "OPEN_EXR"
    """File format used to save diffuse passes"""
    color_mode: Literal["BW", "RGB", "RGBA"] = "RGB"
    """Mode to save diffuse passes in: grayscale, color or color+alpha"""
    exr_codec: EXR_CODECS = "DWAA"
    """Encoding used to compress EXRs"""
    bit_depth: Literal[8, 16, 32] = 32
    """Bit depth used for saving diffuse passes"""
    denoise: bool = True
    """If true, apply denoising to the direct and indirect diffuse passes (Cycles only)"""


@dataclass
class SpecularPassConfig:
    """For more information see :meth:`include_specular_pass <visionsim.simulate.blender.BlenderService.exposed_include_specular_pass>`."""

    file_format: FILE_FORMATS = "OPEN_EXR"
    """File format used to save specular passes"""
    color_mode: Literal["BW", "RGB", "RGBA"] = "RGB"
    """Mode to save specular passes in: grayscale, color or color+alpha"""
    exr_codec: EXR_CODECS = "DWAA"
    """Encoding used to compress EXRs"""
    bit_depth: Literal[8, 16, 32] = 32
    """Bit depth used for saving specular passes"""
    denoise: bool = True
    """If true, apply denoising to the direct and indirect specular passes (Cycles only)"""


@dataclass
class PointsConfig:
    """For more information see :meth:`include_points <visionsim.simulate.blender.BlenderService.exposed_include_points>`."""

    preview: bool = True
    """Also save colorized point maps as PNGs"""
    exr_codec: EXR_CODECS = "DWAA"
    """Encoding used to compress EXRs"""
    bit_depth: Literal[16, 32] = 32
    """Bit depth used for saving point maps"""


@dataclass
class ThermalConfig:
    # --- outputs ---
    radiance: bool = True
    """If true, also render the gray-body thermal-camera radiance image (second render pass)"""
    preview: bool = True
    """Also save an inferno-colormap PNG preview of the temperature map"""
    assignments: Path | None = None
    """Path to a thermal material assignment sidecar (``<scene>.thermal.json``). When set, thermal properties
    are resolved per material slot from the sidecar; when unset the global defaults below are used for every
    surface. Sidecars are authored offline by ``scripts/thermal_assign.py`` and committed."""
    # --- per-object override hook (else globals below) ---
    # overrides: dict[str, ...]  # (future: per-object params by object name; today, globals + obj.heat_sim_material)
    # --- global material defaults (used where no per-object value is set) ---
    initial_temperature_K: float = 295.0
    """Default initial temperature for meshes without a per-object value"""
    thermal_diffusivity_mm2_s: float = 0.17
    """Default thermal diffusivity (mm^2/s)"""
    density_kg_m3: float = 1330.0
    """Default material density (kg/m^3)"""
    specific_heat_J_kgK: float = 880.0
    """Default specific heat (J/kg*K)"""
    emissivity: float = 0.9
    """Default surface emissivity in [0, 1]"""
    # --- solver ---
    irradiance_scale: float = 100.0
    """Scale factor applied to computed irradiance (heating input)"""
    sim_time_s: float = 1.0
    """Total simulated time in seconds (static scene mode)"""
    timestep_s: float = 0.05
    """Solver timestep in seconds"""
    domain: Literal["POINTS", "MESH"] = "POINTS"
    """FEM domain: surface point cloud (recommended) or mesh"""
    laplacian_backend: Literal["ROBUST", "IGL"] = "ROBUST"
    """Laplacian backend"""
    irradiance_source: Literal["DIRECT_KERNEL", "CYCLES_BAKE"] = "DIRECT_KERNEL"
    """Where absorbed flux comes from.

    ``DIRECT_KERNEL`` (default) is analytic: per-light form factors, Embree shadow rays
    and a 9-coefficient SH sky. Fast, but it counts only objects of type ``LIGHT`` plus
    the world sky, and models no indirect bounce -- so a scene lit by emissive geometry
    (window planes, light portals, emissive fixture panels) receives no flux from its
    actual light source.

    ``CYCLES_BAKE`` bakes DIFFUSE DIRECT+INDIRECT per object instead, resolving emissive
    meshes, indirect bounce, portals and HDRI transport. Prefer it whenever the scene's
    real light source is not a lamp object. The symptom of the wrong choice is a room
    that renders flat at its initial temperature while its RGB render is well lit.
    """
    bake_samples: int = 1024
    """Cycles bake samples for the irradiance map (``CYCLES_BAKE`` only).

    Adaptive sampling is disabled for the bake, so this is a true per-texel sample count
    rather than a cap. It is set here rather than inherited from the blend because
    production blends are tuned for a look, often with a loose adaptive threshold that
    terminates texels well below the nominal cap -- fine for an image, not for a physical
    input, since baked irradiance noise propagates into the temperature field.

    A steady-state surface sits at ``T ~ (E/(eps*sigma))**0.25``, so a relative error in
    irradiance appears as roughly a quarter of that in temperature. Noise falls as
    ``1/sqrt(N)``, so quadrupling this buys a little under half the noise. Denoising does
    not apply to a bake; sample count is the only lever.
    """
    irradiance_texture_size: int = 512
    """Resolution of the Cycles bakes, in pixels per side (square).

    Governs both the albedo bake and, under ``CYCLES_BAKE``, the irradiance bake. This is
    the *spatial detail* of the baked flux, as distinct from ``bake_samples``, which is its
    *noise*: more samples make a smoother bake at the same resolution, and cannot recover
    detail the resolution never captured. A large surface unwrapped into one 512px tile
    gets few texels per square metre however many samples you throw at it, so raise this
    for scenes with big floors, walls or ceilings. Cost is quadratic in this value.
    """
    device: Literal["cuda", "cpu"] = "cuda"
    """Torch device for the solve; falls back to cpu if cuda is unavailable"""
    # --- thermal atlas (texel-domain render) ---
    render_domain: Literal["VERTEX", "TEXEL"] = "VERTEX"
    """Where solved temperatures live for rendering: per-vertex (today's behavior, byte-identical)
    or in a shared texture atlas sampled per-pixel by the shader (denser surfaces, no reliance on
    mesh vertex density)."""
    atlas_texel_density: float = 1500.0
    """Target texels/m^2 for atlas-eligible objects (those whose native vertex density is below
    this). Provisional default pending an in-render timing benchmark (see the design spec's
    validation plan); tune down for large/slow scenes."""
    atlas_tile_min: int = 16
    """Minimum atlas tile side, in texels (per object)."""
    atlas_tile_max: int = 512
    """Maximum atlas tile side, in texels (per object)."""
    atlas_texel_soft_max: int = 500_000
    """Soft ceiling on total atlas texels + retained vertices; exceeding it rescales the
    effective density down uniformly and warns, rather than allocating an unbounded solve."""
    # --- radiance render ---
    radiance_scale: float = 1.0
    """Gray-body emission magnitude knob for the thermal_radiance render"""
    # --- file formats (mirror DepthsConfig) ---
    exr_codec: EXR_CODECS = "DWAA"
    """Encoding used to compress EXRs"""
    bit_depth: Literal[16, 32] = 32
    """Bit depth for temperature/radiance EXRs"""
    # --- animated (transient) ---
    animated: bool = False
    """Solve heat transfer per-frame as geometry animates (transient). When False, the static single-shot solve is used."""
    substeps_per_frame: int = 4
    """Solver substeps per Blender frame in animated mode (dt = (1/fps)/substeps_per_frame)."""
    frame_start: int | None = None
    """First frame of the animated solve; defaults to the scene frame_start."""
    frame_end: int | None = None
    """Last frame of the animated solve; defaults to the scene frame_end."""
    every_n_frames: int = 1
    """Solve every Nth frame (cost control); skipped frames hold the last solved field."""


@dataclass
class RenderConfig:
    executable: Path | None = None
    """Path to blender executable"""
    height: int | None = None
    """Height of rendered frames in pixels"""
    width: int | None = None
    """Width of rendered frames in pixels"""
    resolution_percentage: int = 100
    """Percentage of the original resolution to render at"""
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
    include_diffuse_pass: bool = False
    """If true, enable diffuse light pass outputs"""
    diffuse_pass: DiffusePassConfig = field(default_factory=DiffusePassConfig)
    """Diffuse light passes configuration options"""
    include_specular_pass: bool = False
    """If true, enable specular light pass outputs"""
    specular_pass: SpecularPassConfig = field(default_factory=SpecularPassConfig)
    """Specular light passes configuration options"""
    include_points: bool = False
    """If true, enable world-space point map outputs"""
    points: PointsConfig = field(default_factory=PointsConfig)
    """Point maps configuration options"""
    include_thermal: bool = False
    """If true, enable thermal outputs (temperature map + thermal-camera radiance)"""
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    """Thermal modality configuration options"""
    include_all: bool = False
    """If true, enable all ground truth outputs"""
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
    camera_offset: tuple[float, float, float] | None = None
    """Camera offset to apply in local coordinates (x, y, z)."""

    def __post_init__(self):
        # Note: Using post init with tyro is not best practice, as it will be called multiple
        #   times. However here we are just propagating values of aliases, so it should be ok.
        # See: https://brentyi.github.io/tyro/examples/overriding_configs/#dataclasses-defaults
        if self.include_all:
            self.include_composites = True
            self.include_frames = True
            self.include_depths = True
            self.include_normals = True
            self.include_flows = True
            self.include_segmentations = True
            self.include_materials = True
            self.include_diffuse_pass = True
            self.include_specular_pass = True
            self.include_points = True
            self.include_thermal = True

        self.depths.preview &= self.previews
        self.normals.preview &= self.previews
        self.flows.preview &= self.previews
        self.segmentations.preview &= self.previews
        self.materials.preview &= self.previews
        self.points.preview &= self.previews
        self.thermal.preview &= self.previews
