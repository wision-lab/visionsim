import functools
import logging
import os
import socket
import subprocess
from collections.abc import Callable, Collection, Generator, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from multiprocessing import Process
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, type_check_only

import bpy  # type: ignore
import multiprocess  # type: ignore
import numpy as np
import numpy.typing as npt
import rpyc  # type: ignore
import rpyc.utils.registry  # type: ignore
import rpyc.utils.server  # type: ignore
from _typeshed import Incomplete
from typing_extensions import Self

from visionsim.types import COLOR_MODES, EXR_CODECS, FILE, FILE_FORMATS, UpdateFn

handlers: Iterable[logging.Handler] | None
server_log: logging.Logger
EXPOSED_PREFIX: str
REGISTRY: tuple[Process, rpyc.utils.registry.UDPRegistryClient] | None
ITEMS_PER_SUBFOLDER: int
INDEX_PADDING: int
FORMATS: dict[str, str]
COLOR_MODE_CHANNELS: Incomplete

def require_connected_client(func: Callable[..., Any]) -> Callable[..., Any]:
    ...

def require_connected_clients(func: Callable[..., Any]) -> Callable[..., Any]:
    ...

def require_initialized_service(func: Callable[..., Any]) -> Callable[..., Any]:
    ...

def validate_camera_moved(func: Callable[..., Any]) -> Callable[..., Any]:
    ...

class BlenderServer(rpyc.utils.server.Server):

    def __init__(
        self,
        hostname: bytes | str | None = None,
        port: bytes | str | int | None = 0,
        service: type[BlenderService] | None = None,
        extra_config: dict | None = None,
        **kwargs,
    ) -> None:
        ...

    @contextmanager
    @staticmethod
    def spawn(
        jobs: int = 1,
        timeout: float = -1.0,
        log: str | os.PathLike | FILE | tuple[FILE, FILE] = ...,
        autoexec: bool = False,
        executable: str | os.PathLike | None = None,
    ) -> Generator[tuple[list[subprocess.Popen], list[tuple[str, int]]]]:
        ...

    @staticmethod
    def spawn_registry() -> tuple[Process, rpyc.utils.registry.UDPRegistryClient]:
        ...

    @staticmethod
    def _launch_registry() -> None: ...
    @staticmethod
    def discover() -> list[tuple[str, int]]:
        ...

    def _accept_method(self, sock: socket.socket) -> None: ...

class BlenderService(rpyc.Service):

    ALIASES: tuple[str]
    _conn: rpyc.Connection | None
    log: logging.Logger
    _initialized: bool
    _keyframe_scale: float
    _warned_no_outputs: bool
    _outputs: dict[str, Any]
    _camera: bpy.types.Camera | None
    _thermal_radiance: dict[str, Any] | None
    _thermal_assignment: Any | None
    _loaded_persistent_data: bool | None
    _persistent_data_before_thermal: bool | None
    _thermal_atlas_plan: Any | None

    def __init__(self) -> None:
        ...

    def _clear_cached_properties(self) -> None: ...
    def on_connect(self, conn: rpyc.Connection) -> None:
        ...

    def on_disconnect(self, conn: rpyc.Connection) -> None:
        ...

    def reset(self) -> None:
        ...

    def register_output_type(
        self,
        subpath: str,
        node: bpy.types.CompositorNodeOutputFile,
        slot: bpy.types.NodeOutputFileSlotFile | bpy.types.NodeCompositorFileOutputItem,
        **camera_defaults,
    ) -> None:
        ...

    def _include_output(
        self,
        subpath: str,
        source_socket: bpy.types.NodeSocket,
        label: str | None = None,
        file_format: FILE_FORMATS = "OPEN_EXR",
        color_mode: COLOR_MODES = "RGB",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: int = 32,
        preview: bool = False,
        preview_view_transform: str | None = None,
        c: int | None = None,
        denoise: bool = False,
    ) -> None:
        ...

    def _save_metadata(
        self,
        paths: dict[str, Path],
        camera_info: dict[str, str | float | int],
        transform_matrix: list[float],
        index: int,
    ) -> None:
        ...

    @property
    @require_initialized_service
    def scene(self) -> bpy.types.Scene:
        ...

    @property
    @require_initialized_service
    def tree(self) -> bpy.types.CompositorNodeTree:
        ...

    @functools.cached_property
    @require_initialized_service
    def render_layers(self) -> bpy.types.CompositorNodeRLayers:
        ...

    @property
    @require_initialized_service
    def view_layer(self) -> bpy.types.ViewLayer:
        ...

    @property
    @require_initialized_service
    def camera(self) -> bpy.types.Camera:
        ...

    @require_initialized_service
    def get_parents(self, obj: bpy.types.Object) -> list[bpy.types.Object]:
        ...

    def exposed_with_logger(self, log: logging.Logger) -> None:
        ...
    root_path: Path
    blend_file: Path
    _use_animation: bool
    _disabled_fcurves: set[bpy.types.Action]

    def exposed_initialize(self, blend_file: str | os.PathLike, root_path: str | os.PathLike, **kwargs) -> None:
        ...

    @require_initialized_service
    def exposed_iter_fcurves(self, actions: list[bpy.types.Action] | None = None) -> Iterator[bpy.types.FCurve]:
        ...

    @require_initialized_service
    def exposed_get_original_fps(self) -> float:
        ...

    @require_initialized_service
    def exposed_animation_range(self) -> range:
        ...

    @require_initialized_service
    def exposed_animation_range_tuple(self) -> tuple[int, int, int]:
        ...

    @require_initialized_service
    def exposed_include_composites(
        self,
        file_format: FILE_FORMATS | None = None,
        color_mode: COLOR_MODES | None = None,
        exr_codec: EXR_CODECS | None = None,
        bit_depth: Literal[8, 16, 32] | None = None,
    ) -> None:
        ...

    @require_initialized_service
    def exposed_include_frames(
        self,
        file_format: FILE_FORMATS = "PNG",
        color_mode: COLOR_MODES = "RGB",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[8, 16, 32] = 8,
    ) -> None:
        ...

    @require_initialized_service
    def exposed_include_depths(
        self,
        preview: bool = True,
        file_format: FILE_FORMATS = "OPEN_EXR",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
    ) -> None:
        ...

    @require_initialized_service
    def exposed_include_normals(
        self, preview: bool = True, exr_codec: EXR_CODECS = "DWAA", bit_depth: Literal[16, 32] = 32
    ) -> None:
        ...

    @require_initialized_service
    def exposed_include_flows(
        self,
        preview: bool = True,
        direction: Literal["forward", "backward", "both"] = "forward",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
    ) -> None:
        ...

    @require_initialized_service
    def _include_ids(
        self,
        id_type: Literal["segmentations", "materials"],
        preview: bool = True,
        shuffle: bool = True,
        seed: int = 1234,
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
        shade: bool = False,
    ) -> None:
        ...

    @require_initialized_service
    def exposed_include_segmentations(
        self,
        preview: bool = True,
        shuffle: bool = True,
        seed: int = 1234,
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
    ) -> None:
        ...

    @require_initialized_service
    def exposed_include_materials(
        self,
        preview: bool = True,
        shuffle: bool = True,
        seed: int = 1234,
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
    ) -> None:
        ...

    @require_initialized_service
    def exposed_include_diffuse_pass(
        self,
        file_format: FILE_FORMATS = "OPEN_EXR",
        color_mode: COLOR_MODES = "RGB",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[8, 16, 32] = 32,
        denoise: bool = True,
    ) -> None:
        ...

    @require_initialized_service
    def exposed_include_specular_pass(
        self,
        file_format: FILE_FORMATS = "OPEN_EXR",
        color_mode: COLOR_MODES = "RGB",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[8, 16, 32] = 32,
        denoise: bool = True,
    ) -> None:
        ...

    @require_initialized_service
    def exposed_include_points(
        self, preview: bool = True, exr_codec: EXR_CODECS = "DWAA", bit_depth: Literal[16, 32] = 32
    ) -> None:
        ...

    def _thermal_config(
        self,
        *,
        initial_temperature_K: float,
        thermal_diffusivity_mm2_s: float,
        density_kg_m3: float,
        specific_heat_J_kgK: float,
        emissivity: float,
        irradiance_scale: float,
        sim_time_s: float,
        timestep_s: float,
        bake_samples: int = 1024,
        irradiance_texture_size: int = 512,
        device: Literal["cuda", "cpu"],
        assignments: str | None = None,
    ) -> tuple[dict, dict, Path, Any]:
        ...

    def _thermal_solve(
        self,
        *,
        initial_temperature_K: float,
        thermal_diffusivity_mm2_s: float,
        density_kg_m3: float,
        specific_heat_J_kgK: float,
        emissivity: float,
        irradiance_scale: float,
        sim_time_s: float,
        timestep_s: float,
        bake_samples: int = 1024,
        irradiance_texture_size: int = 512,
        device: Literal["cuda", "cpu"],
        assignments: str | None = None,
        render_domain: Literal["AUTO", "VERTEX", "TEXEL"] = "AUTO",
        atlas_texel_density: float = 1500.0,
        atlas_tile_min: int = 16,
        atlas_tile_max: int = 512,
        atlas_texel_soft_max: int = 500000,
        recompute: bool = False,
    ) -> tuple[dict, Any, Path]:
        ...

    def _thermal_load_pack_atlas_image(self, atlas_path: Path) -> None:
        ...
    _thermal_temp_range: Incomplete

    @require_initialized_service
    def exposed_prepare_thermal(
        self,
        radiance: bool = True,
        preview: bool = True,
        initial_temperature_K: float = 295.0,
        thermal_diffusivity_mm2_s: float = 0.17,
        density_kg_m3: float = 1330.0,
        specific_heat_J_kgK: float = 880.0,
        emissivity: float = 0.9,
        irradiance_scale: float = 100.0,
        sim_time_s: float = 1.0,
        timestep_s: float = 0.05,
        device: Literal["cuda", "cpu"] = "cuda",
        render_domain: Literal["AUTO", "VERTEX", "TEXEL"] = "AUTO",
        atlas_texel_density: float = 1500.0,
        atlas_tile_min: int = 16,
        atlas_tile_max: int = 512,
        atlas_texel_soft_max: int = 500000,
        recompute: bool = False,
        radiance_scale: float = 1.0,
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
        assignments: str | None = None,
    ) -> None:
        ...

    def _thermal_write_frame(self, frame_number: int) -> None:
        ...

    @require_initialized_service
    def exposed_heatsim_solve(
        self,
        radiance: bool = True,
        preview: bool = True,
        initial_temperature_K: float = 295.0,
        thermal_diffusivity_mm2_s: float = 0.17,
        density_kg_m3: float = 1330.0,
        specific_heat_J_kgK: float = 880.0,
        emissivity: float = 0.9,
        irradiance_scale: float = 100.0,
        sim_time_s: float = 1.0,
        timestep_s: float = 0.05,
        device: Literal["cuda", "cpu"] = "cuda",
        render_domain: Literal["AUTO", "VERTEX", "TEXEL"] = "AUTO",
        atlas_texel_density: float = 1500.0,
        atlas_tile_min: int = 16,
        atlas_tile_max: int = 512,
        atlas_texel_soft_max: int = 500000,
        recompute: bool = False,
        radiance_scale: float = 1.0,
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
        assignments: str | None = None,
    ) -> None:
        ...

    @require_initialized_service
    def exposed_include_thermal(
        self,
        radiance: bool = True,
        preview: bool = True,
        initial_temperature_K: float = 295.0,
        thermal_diffusivity_mm2_s: float = 0.17,
        density_kg_m3: float = 1330.0,
        specific_heat_J_kgK: float = 880.0,
        emissivity: float = 0.9,
        irradiance_scale: float = 100.0,
        sim_time_s: float = 1.0,
        timestep_s: float = 0.05,
        device: Literal["cuda", "cpu"] = "cuda",
        render_domain: Literal["AUTO", "VERTEX", "TEXEL"] = "AUTO",
        atlas_texel_density: float = 1500.0,
        atlas_tile_min: int = 16,
        atlas_tile_max: int = 512,
        atlas_texel_soft_max: int = 500000,
        recompute: bool = False,
        radiance_scale: float = 1.0,
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
        assignments: str | None = None,
    ) -> None:
        ...

    @require_initialized_service
    @staticmethod
    def _thermal_values(config: dict[str, Any]) -> dict[str, Any]:
        ...

    def exposed_configure_thermal(self, config: dict[str, Any]) -> None:
        ...

    def exposed_heatsim_solve_config(self, config: dict[str, Any]) -> None:
        ...

    def exposed_load_addons(self, *addons: str) -> None:
        ...

    @require_initialized_service
    def exposed_set_resolution(
        self,
        height: tuple[int] | list[int] | int | None = None,
        width: int | None = None,
        resolution_percentage: int = 100,
    ) -> None:
        ...

    @require_initialized_service
    def exposed_use_motion_blur(self, enable: bool) -> None:
        ...

    @require_initialized_service
    def exposed_use_animations(self, enable: bool) -> None:
        ...

    @require_initialized_service
    def exposed_cycles_settings(
        self,
        device_type: str | None = None,
        use_cpu: bool | None = None,
        adaptive_threshold: float | None = None,
        max_samples: int | None = None,
        use_denoising: bool | None = None,
    ) -> list[str]:
        ...

    @require_initialized_service
    def exposed_unbind_camera(self, clear_animations: bool = True) -> None:
        ...

    @require_initialized_service
    def exposed_move_keyframes(self, scale: float = 1.0, shift: float = 0.0) -> None:
        ...

    @require_initialized_service
    def exposed_set_current_frame(self, frame_number: int) -> None:
        ...

    @require_initialized_service
    def exposed_camera_info(self) -> dict[str, Any]:
        ...

    @require_initialized_service
    def exposed_camera_extrinsics(self) -> npt.NDArray[np.floating]:
        ...

    @require_initialized_service
    @validate_camera_moved
    def exposed_position_camera(
        self,
        location: npt.ArrayLike | None = None,
        rotation: npt.ArrayLike | None = None,
        look_at: npt.ArrayLike | None = None,
        in_order: bool = True,
    ) -> None:
        ...

    @require_initialized_service
    @validate_camera_moved
    def exposed_rotate_camera(self, angle: float) -> None:
        ...

    @require_initialized_service
    @validate_camera_moved
    def exposed_offset_camera(self, offset: npt.ArrayLike) -> None:
        ...

    @require_initialized_service
    def exposed_set_camera_keyframe(self, frame_num: int, matrix: npt.ArrayLike | None = None) -> None:
        ...

    @require_initialized_service
    def exposed_set_animation_range(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> None:
        ...

    @require_initialized_service
    def exposed_render_current_frame(self, allow_skips: bool = True, dry_run: bool = False) -> None:
        ...

    @require_initialized_service
    def exposed_render_frame(self, frame_number: int, allow_skips: bool = True, dry_run: bool = False) -> None:
        ...

    @require_initialized_service
    def exposed_render_frames(
        self,
        frame_numbers: Iterable[int],
        allow_skips: bool = True,
        dry_run: bool = False,
        update_fn: UpdateFn | None = None,
    ) -> None:
        ...

    @require_initialized_service
    def exposed_render_animation(
        self,
        frame_start: int | None = None,
        frame_end: int | None = None,
        frame_step: int | None = None,
        allow_skips: bool = True,
        dry_run: bool = False,
        update_fn: UpdateFn | None = None,
    ) -> None:
        ...

    @require_initialized_service
    def exposed_save_file(self, path: str | os.PathLike) -> None:
        ...

class BlenderClient:

    addr: tuple[str, int]
    conn: rpyc.Connection | None
    awaitable: rpyc.AsyncResult | None
    process: subprocess.Popen | None
    timeout: float

    def __init__(self, addr: tuple[str, int], timeout: float = 10.0) -> None:
        ...

    @classmethod
    def auto_connect(cls, timeout: float = 10.0) -> Self:
        ...

    @classmethod
    @contextmanager
    def spawn(
        cls,
        timeout: float = -1.0,
        log: str | os.PathLike | FILE | tuple[FILE, FILE] = ...,
        autoexec: bool = False,
        executable: str | os.PathLike | None = None,
    ) -> Generator[Self]:
        ...

    @require_connected_client
    def render_animation_async(self, *args, **kwargs) -> rpyc.AsyncResult:
        ...

    @require_connected_client
    def render_frames_async(self, *args, **kwargs) -> rpyc.AsyncResult:
        ...

    def wait(self) -> None:
        ...

    def __enter__(self) -> Self:
        ...

    def __getattr__(self, name: str) -> rpyc.BaseNetref:
        ...

    def __exit__(
        self, type: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        ...

    @type_check_only
    def with_logger(self, log: logging.Logger) -> None:
        ...

    @type_check_only
    def initialize(self, blend_file: str | os.PathLike, root_path: str | os.PathLike, **kwargs) -> None:
        ...

    @type_check_only
    def iter_fcurves(self, actions: list[bpy.types.Action] | None = None) -> Iterator[bpy.types.FCurve]:
        ...

    @type_check_only
    def get_original_fps(self) -> float:
        ...

    @type_check_only
    def animation_range(self) -> range:
        ...

    @type_check_only
    def animation_range_tuple(self) -> tuple[int, int, int]:
        ...

    @type_check_only
    def include_composites(
        self,
        file_format: FILE_FORMATS | None = None,
        color_mode: COLOR_MODES | None = None,
        exr_codec: EXR_CODECS | None = None,
        bit_depth: Literal[8, 16, 32] | None = None,
    ) -> None:
        ...

    @type_check_only
    def include_frames(
        self,
        file_format: FILE_FORMATS = "PNG",
        color_mode: COLOR_MODES = "RGB",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[8, 16, 32] = 8,
    ) -> None:
        ...

    @type_check_only
    def include_depths(
        self,
        preview: bool = True,
        file_format: FILE_FORMATS = "OPEN_EXR",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
    ) -> None:
        ...

    @type_check_only
    def include_normals(
        self, preview: bool = True, exr_codec: EXR_CODECS = "DWAA", bit_depth: Literal[16, 32] = 32
    ) -> None:
        ...

    @type_check_only
    def include_flows(
        self,
        preview: bool = True,
        direction: Literal["forward", "backward", "both"] = "forward",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
    ) -> None:
        ...

    @type_check_only
    def include_segmentations(
        self,
        preview: bool = True,
        shuffle: bool = True,
        seed: int = 1234,
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
    ) -> None:
        ...

    @type_check_only
    def include_materials(
        self,
        preview: bool = True,
        shuffle: bool = True,
        seed: int = 1234,
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
    ) -> None:
        ...

    @type_check_only
    def include_diffuse_pass(
        self,
        file_format: FILE_FORMATS = "OPEN_EXR",
        color_mode: COLOR_MODES = "RGB",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[8, 16, 32] = 32,
        denoise: bool = True,
    ) -> None:
        ...

    @type_check_only
    def include_specular_pass(
        self,
        file_format: FILE_FORMATS = "OPEN_EXR",
        color_mode: COLOR_MODES = "RGB",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[8, 16, 32] = 32,
        denoise: bool = True,
    ) -> None:
        ...

    @type_check_only
    def include_points(
        self, preview: bool = True, exr_codec: EXR_CODECS = "DWAA", bit_depth: Literal[16, 32] = 32
    ) -> None:
        ...

    @type_check_only
    def prepare_thermal(
        self,
        radiance: bool = True,
        preview: bool = True,
        initial_temperature_K: float = 295.0,
        thermal_diffusivity_mm2_s: float = 0.17,
        density_kg_m3: float = 1330.0,
        specific_heat_J_kgK: float = 880.0,
        emissivity: float = 0.9,
        irradiance_scale: float = 100.0,
        sim_time_s: float = 1.0,
        timestep_s: float = 0.05,
        device: Literal["cuda", "cpu"] = "cuda",
        render_domain: Literal["AUTO", "VERTEX", "TEXEL"] = "AUTO",
        atlas_texel_density: float = 1500.0,
        atlas_tile_min: int = 16,
        atlas_tile_max: int = 512,
        atlas_texel_soft_max: int = 500000,
        recompute: bool = False,
        radiance_scale: float = 1.0,
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
        assignments: str | None = None,
    ) -> None:
        ...

    @type_check_only
    def heatsim_solve(
        self,
        radiance: bool = True,
        preview: bool = True,
        initial_temperature_K: float = 295.0,
        thermal_diffusivity_mm2_s: float = 0.17,
        density_kg_m3: float = 1330.0,
        specific_heat_J_kgK: float = 880.0,
        emissivity: float = 0.9,
        irradiance_scale: float = 100.0,
        sim_time_s: float = 1.0,
        timestep_s: float = 0.05,
        device: Literal["cuda", "cpu"] = "cuda",
        render_domain: Literal["AUTO", "VERTEX", "TEXEL"] = "AUTO",
        atlas_texel_density: float = 1500.0,
        atlas_tile_min: int = 16,
        atlas_tile_max: int = 512,
        atlas_texel_soft_max: int = 500000,
        recompute: bool = False,
        radiance_scale: float = 1.0,
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
        assignments: str | None = None,
    ) -> None:
        ...

    @type_check_only
    def include_thermal(
        self,
        radiance: bool = True,
        preview: bool = True,
        initial_temperature_K: float = 295.0,
        thermal_diffusivity_mm2_s: float = 0.17,
        density_kg_m3: float = 1330.0,
        specific_heat_J_kgK: float = 880.0,
        emissivity: float = 0.9,
        irradiance_scale: float = 100.0,
        sim_time_s: float = 1.0,
        timestep_s: float = 0.05,
        device: Literal["cuda", "cpu"] = "cuda",
        render_domain: Literal["AUTO", "VERTEX", "TEXEL"] = "AUTO",
        atlas_texel_density: float = 1500.0,
        atlas_tile_min: int = 16,
        atlas_tile_max: int = 512,
        atlas_texel_soft_max: int = 500000,
        recompute: bool = False,
        radiance_scale: float = 1.0,
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
        assignments: str | None = None,
    ) -> None:
        ...

    @type_check_only
    def configure_thermal(self, config: dict[str, Any]) -> None:
        ...

    def heatsim_solve_config(self, config: dict[str, Any]) -> None:
        ...

    def load_addons(self, *addons: str) -> None:
        ...

    @type_check_only
    def set_resolution(
        self,
        height: tuple[int] | list[int] | int | None = None,
        width: int | None = None,
        resolution_percentage: int = 100,
    ) -> None:
        ...

    @type_check_only
    def use_motion_blur(self, enable: bool) -> None:
        ...

    @type_check_only
    def use_animations(self, enable: bool) -> None:
        ...

    @type_check_only
    def cycles_settings(
        self,
        device_type: str | None = None,
        use_cpu: bool | None = None,
        adaptive_threshold: float | None = None,
        max_samples: int | None = None,
        use_denoising: bool | None = None,
    ) -> list[str]:
        ...

    @type_check_only
    def unbind_camera(self, clear_animations: bool = True) -> None:
        ...

    @type_check_only
    def move_keyframes(self, scale: float = 1.0, shift: float = 0.0) -> None:
        ...

    @type_check_only
    def set_current_frame(self, frame_number: int) -> None:
        ...

    @type_check_only
    def camera_info(self) -> dict[str, Any]:
        ...

    @type_check_only
    def camera_extrinsics(self) -> npt.NDArray[np.floating]:
        ...

    @type_check_only
    def position_camera(
        self,
        location: npt.ArrayLike | None = None,
        rotation: npt.ArrayLike | None = None,
        look_at: npt.ArrayLike | None = None,
        in_order: bool = True,
    ) -> None:
        ...

    @type_check_only
    def rotate_camera(self, angle: float) -> None:
        ...

    @type_check_only
    def offset_camera(self, offset: npt.ArrayLike) -> None:
        ...

    @type_check_only
    def set_camera_keyframe(self, frame_num: int, matrix: npt.ArrayLike | None = None) -> None:
        ...

    @type_check_only
    def set_animation_range(self, start: int | None = None, stop: int | None = None, step: int | None = None) -> None:
        ...

    @type_check_only
    def render_current_frame(self, allow_skips: bool = True, dry_run: bool = False) -> None:
        ...

    @type_check_only
    def render_frame(self, frame_number: int, allow_skips: bool = True, dry_run: bool = False) -> None:
        ...

    @type_check_only
    def render_frames(
        self,
        frame_numbers: Iterable[int],
        allow_skips: bool = True,
        dry_run: bool = False,
        update_fn: UpdateFn | None = None,
    ) -> None:
        ...

    @type_check_only
    def render_animation(
        self,
        frame_start: int | None = None,
        frame_end: int | None = None,
        frame_step: int | None = None,
        allow_skips: bool = True,
        dry_run: bool = False,
        update_fn: UpdateFn | None = None,
    ) -> None:
        ...

    @type_check_only
    def save_file(self, path: str | os.PathLike) -> None:
        ...

class BlenderClients(tuple):

    def __getattr__(self, name: str) -> Callable[..., Any]:
        ...

    def __new__(cls, *objs: Iterator[BlenderClient | tuple[str, int]]) -> Self:
        ...
    stack: ExitStack

    def __init__(self, *objs) -> None:
        ...

    def _method_dispatch_factory(self, name: str, method: Callable) -> Callable: ...
    def __enter__(self) -> Self:
        ...

    def __exit__(
        self, type: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        ...

    @classmethod
    @contextmanager
    def spawn(
        cls,
        jobs: int = 1,
        timeout: float = -1.0,
        log: str | os.PathLike | FILE | tuple[FILE, FILE] = ...,
        autoexec: bool = False,
        executable: str | os.PathLike | None = None,
    ) -> Generator[Self]:
        ...

    @contextmanager
    @staticmethod
    def pool(
        jobs: int = 1,
        timeout: float = -1.0,
        log: str | os.PathLike | FILE | tuple[FILE, FILE] = ...,
        autoexec: bool = False,
        executable: str | os.PathLike | None = None,
        conns: list[tuple[str, int]] | None = None,
    ) -> Generator[multiprocess.Pool]:
        ...

    @require_connected_clients
    def common_animation_range(self) -> range:
        ...

    @require_connected_clients
    def common_animation_range_tuple(self) -> tuple[int, int, int]:
        ...

    @require_connected_clients
    def render_frames(
        self,
        frame_numbers: Collection[int],
        allow_skips: bool = True,
        dry_run: bool = False,
        update_fn: UpdateFn | None = None,
    ) -> None:
        ...

    @require_connected_clients
    def render_animation(
        self,
        frame_start: int | None = None,
        frame_end: int | None = None,
        frame_step: int | None = None,
        allow_skips: bool = True,
        dry_run: bool = False,
        update_fn: UpdateFn | None = None,
    ) -> None:
        ...

    @require_connected_clients
    def save_file(self, path: str | os.PathLike) -> None:
        ...

    def wait(self) -> None:
        ...

    @type_check_only
    def with_logger(self, log: logging.Logger) -> None:
        ...

    @type_check_only
    def initialize(self, blend_file: str | os.PathLike, root_path: str | os.PathLike, **kwargs) -> None:
        ...

    @type_check_only
    def iter_fcurves(self, actions: list[bpy.types.Action] | None = None) -> tuple[Iterator[bpy.types.FCurve],]:
        ...

    @type_check_only
    def get_original_fps(self) -> tuple[float,]:
        ...

    @type_check_only
    def animation_range(self) -> tuple[range,]:
        ...

    @type_check_only
    def animation_range_tuple(self) -> tuple[tuple[int, int, int],]:
        ...

    @type_check_only
    def include_composites(
        self,
        file_format: FILE_FORMATS | None = None,
        color_mode: COLOR_MODES | None = None,
        exr_codec: EXR_CODECS | None = None,
        bit_depth: Literal[8, 16, 32] | None = None,
    ) -> None:
        ...

    @type_check_only
    def include_frames(
        self,
        file_format: FILE_FORMATS = "PNG",
        color_mode: COLOR_MODES = "RGB",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[8, 16, 32] = 8,
    ) -> None:
        ...

    @type_check_only
    def include_depths(
        self,
        preview: bool = True,
        file_format: FILE_FORMATS = "OPEN_EXR",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
    ) -> None:
        ...

    @type_check_only
    def include_normals(
        self, preview: bool = True, exr_codec: EXR_CODECS = "DWAA", bit_depth: Literal[16, 32] = 32
    ) -> None:
        ...

    @type_check_only
    def include_flows(
        self,
        preview: bool = True,
        direction: Literal["forward", "backward", "both"] = "forward",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
    ) -> None:
        ...

    @type_check_only
    def include_segmentations(
        self,
        preview: bool = True,
        shuffle: bool = True,
        seed: int = 1234,
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
    ) -> None:
        ...

    @type_check_only
    def include_materials(
        self,
        preview: bool = True,
        shuffle: bool = True,
        seed: int = 1234,
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
    ) -> None:
        ...

    @type_check_only
    def include_diffuse_pass(
        self,
        file_format: FILE_FORMATS = "OPEN_EXR",
        color_mode: COLOR_MODES = "RGB",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[8, 16, 32] = 32,
        denoise: bool = True,
    ) -> None:
        ...

    @type_check_only
    def include_specular_pass(
        self,
        file_format: FILE_FORMATS = "OPEN_EXR",
        color_mode: COLOR_MODES = "RGB",
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[8, 16, 32] = 32,
        denoise: bool = True,
    ) -> None:
        ...

    @type_check_only
    def include_points(
        self, preview: bool = True, exr_codec: EXR_CODECS = "DWAA", bit_depth: Literal[16, 32] = 32
    ) -> None:
        ...

    @type_check_only
    def prepare_thermal(
        self,
        radiance: bool = True,
        preview: bool = True,
        initial_temperature_K: float = 295.0,
        thermal_diffusivity_mm2_s: float = 0.17,
        density_kg_m3: float = 1330.0,
        specific_heat_J_kgK: float = 880.0,
        emissivity: float = 0.9,
        irradiance_scale: float = 100.0,
        sim_time_s: float = 1.0,
        timestep_s: float = 0.05,
        device: Literal["cuda", "cpu"] = "cuda",
        render_domain: Literal["AUTO", "VERTEX", "TEXEL"] = "AUTO",
        atlas_texel_density: float = 1500.0,
        atlas_tile_min: int = 16,
        atlas_tile_max: int = 512,
        atlas_texel_soft_max: int = 500000,
        recompute: bool = False,
        radiance_scale: float = 1.0,
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
        assignments: str | None = None,
    ) -> None:
        ...

    @type_check_only
    def heatsim_solve(
        self,
        radiance: bool = True,
        preview: bool = True,
        initial_temperature_K: float = 295.0,
        thermal_diffusivity_mm2_s: float = 0.17,
        density_kg_m3: float = 1330.0,
        specific_heat_J_kgK: float = 880.0,
        emissivity: float = 0.9,
        irradiance_scale: float = 100.0,
        sim_time_s: float = 1.0,
        timestep_s: float = 0.05,
        device: Literal["cuda", "cpu"] = "cuda",
        render_domain: Literal["AUTO", "VERTEX", "TEXEL"] = "AUTO",
        atlas_texel_density: float = 1500.0,
        atlas_tile_min: int = 16,
        atlas_tile_max: int = 512,
        atlas_texel_soft_max: int = 500000,
        recompute: bool = False,
        radiance_scale: float = 1.0,
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
        assignments: str | None = None,
    ) -> None:
        ...

    @type_check_only
    def include_thermal(
        self,
        radiance: bool = True,
        preview: bool = True,
        initial_temperature_K: float = 295.0,
        thermal_diffusivity_mm2_s: float = 0.17,
        density_kg_m3: float = 1330.0,
        specific_heat_J_kgK: float = 880.0,
        emissivity: float = 0.9,
        irradiance_scale: float = 100.0,
        sim_time_s: float = 1.0,
        timestep_s: float = 0.05,
        device: Literal["cuda", "cpu"] = "cuda",
        render_domain: Literal["AUTO", "VERTEX", "TEXEL"] = "AUTO",
        atlas_texel_density: float = 1500.0,
        atlas_tile_min: int = 16,
        atlas_tile_max: int = 512,
        atlas_texel_soft_max: int = 500000,
        recompute: bool = False,
        radiance_scale: float = 1.0,
        exr_codec: EXR_CODECS = "DWAA",
        bit_depth: Literal[16, 32] = 32,
        assignments: str | None = None,
    ) -> None:
        ...

    @type_check_only
    def configure_thermal(self, config: dict[str, Any]) -> None:
        ...

    def heatsim_solve_config(self, config: dict[str, Any]) -> None:
        ...

    def load_addons(self, *addons: str) -> None:
        ...

    @type_check_only
    def set_resolution(
        self,
        height: tuple[int] | list[int] | int | None = None,
        width: int | None = None,
        resolution_percentage: int = 100,
    ) -> None:
        ...

    @type_check_only
    def use_motion_blur(self, enable: bool) -> None:
        ...

    @type_check_only
    def use_animations(self, enable: bool) -> None:
        ...

    @type_check_only
    def cycles_settings(
        self,
        device_type: str | None = None,
        use_cpu: bool | None = None,
        adaptive_threshold: float | None = None,
        max_samples: int | None = None,
        use_denoising: bool | None = None,
    ) -> tuple[list[str],]:
        ...

    @type_check_only
    def unbind_camera(self, clear_animations: bool = True) -> None:
        ...

    @type_check_only
    def move_keyframes(self, scale: float = 1.0, shift: float = 0.0) -> None:
        ...

    @type_check_only
    def set_current_frame(self, frame_number: int) -> None:
        ...

    @type_check_only
    def camera_info(self) -> tuple[dict[str, Any],]:
        ...

    @type_check_only
    def camera_extrinsics(self) -> tuple[npt.NDArray[np.floating],]:
        ...

    @type_check_only
    def position_camera(
        self,
        location: npt.ArrayLike | None = None,
        rotation: npt.ArrayLike | None = None,
        look_at: npt.ArrayLike | None = None,
        in_order: bool = True,
    ) -> None:
        ...

    @type_check_only
    def rotate_camera(self, angle: float) -> None:
        ...

    @type_check_only
    def offset_camera(self, offset: npt.ArrayLike) -> None:
        ...

    @type_check_only
    def set_camera_keyframe(self, frame_num: int, matrix: npt.ArrayLike | None = None) -> None:
        ...

    @type_check_only
    def set_animation_range(self, start: int | None = None, stop: int | None = None, step: int | None = None) -> None:
        ...

    @type_check_only
    def render_current_frame(self, allow_skips: bool = True, dry_run: bool = False) -> None:
        ...

    @type_check_only
    def render_frame(self, frame_number: int, allow_skips: bool = True, dry_run: bool = False) -> None:
        ...
