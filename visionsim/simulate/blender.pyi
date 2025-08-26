import functools
import logging
import os
import subprocess
from collections.abc import Callable, Collection, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from multiprocessing import Process
from pathlib import Path
from types import TracebackType

import bpy  # type: ignore
import multiprocess  # type: ignore
import numpy as np
import numpy.typing as npt
import rpyc  # type: ignore
import rpyc.utils  # type: ignore
import rpyc.utils.registry  # type: ignore
from typing_extensions import Any, Concatenate, ParamSpec, Self, TypeVar

from visionsim.types import UpdateFn

T = TypeVar("T")
P = ParamSpec("P")
server_log: logging.Logger
EXPOSED_PREFIX: str
REGISTRY: tuple[Process, rpyc.utils.registry.UDPRegistryClient] | None

def require_connected_client(
    func: Callable[Concatenate[BlenderClient, P], T],
) -> Callable[Concatenate[BlenderClient, P], T]: ...
def require_connected_clients(
    func: Callable[Concatenate[BlenderClients, P], T],
) -> Callable[Concatenate[BlenderClients, P], T]: ...
def require_initialized_service(
    func: Callable[Concatenate[BlenderService, P], T],
) -> Callable[Concatenate[BlenderService, P], T]: ...
def validate_camera_moved(
    func: Callable[Concatenate[BlenderService, P], T],
) -> Callable[Concatenate[BlenderService, P], T]: ...

class BlenderServer(rpyc.utils.server.Server):
    """Expose a `BlenderService` to the outside world via RPCs.

    Example:
        Once created, it can be started, which will block and await for an external connection from a `BlenderClient`:

        .. code-block:: python

            server = BlenderServer()
            server.start()

        However, this needs to be called within blender's runtime. Instead one can use `BlenderServer.spawn`
        to spawn one or more blender instances, each with their own server.

    """

    def __init__(
        self,
        hostname: bytes | str | None = None,
        port: bytes | str | int | None = 0,
        service: type[BlenderService] | None = None,
        extra_config: dict | None = None,
        **kwargs,
    ) -> None:
        """Initialize a `BlenderServer` instance

        Args:
            hostname (bytes | str | None, optional): the host to bind to. By default, the 'wildcard address' is used
                to listen on all interfaces. If not properly secured, the server can receive traffic from
                unintended or even malicious sources. Defaults to None (wildcard).
            port (bytes | str | int | None, optional): the TCP port to bind to. Defaults to 0 (bind to a random open port).
            service (type[BlenderService], optional): the service to expose, must be a `BlenderService` subclass. Defaults to `BlenderService`.
            extra_config (dict, optional): the configuration dictionary that is passed to the RPyC connection.
                Defaults to `{"allow_all_attrs": True, "allow_setattr": True}`.

        Raises:
            RuntimeError: a `BlenderServer` needs to be instantiated from within a blender instance.
            ValueError: the exposed service must be `BlenderService` or subclass.
        """

    @staticmethod
    @contextmanager
    def spawn(
        jobs: int = 1,
        timeout: float = -1.0,
        log_dir: str | os.PathLike | None = None,
        autoexec: bool = False,
        executable: str | os.PathLike | None = None,
    ) -> Iterator[tuple[list[subprocess.Popen], list[tuple[str, int]]]]:
        """Spawn one or more blender instances and start a `BlenderServer` in each.

        This is roughly equivalent to calling `blender -b --python blender.py` in many subprocesses,
        where `blender.py` initializes and `start`s a server instance. Proper logging and termination of
        these processes is also taken care of.

        Note: The returned processes and connection settings are not guaranteed to be in the same order.

        Args:
            jobs (int, optional): number of jobs to spawn. Defaults to 1.
            timeout (float, optional): try to discover spawned instances for `timeout`
                (in seconds) before giving up. If negative, a port will be randomly selected and assigned to the
                spawned server, bypassing the need for discovery and timeouts. Note that when a port is assigned
                this context manager will immediately yield, even if the server is not yet ready to accept
                incoming connections. Defaults to assigning a port to spawned server (-1 seconds).
            log_dir (str | os.PathLike | None, optional): path to log directory,
                stdout/err will be captured if set, otherwise outputs will go to os.devnull.
                Defaults to None (devnull).
            autoexec (bool, optional): if true, allow execution of any embedded python scripts within blender.
                For more, see blender's CLI documentation. Defaults to False.
            executable (str | os.PathLike | None, optional): path to Blender's executable. Defaults to looking
                for blender on $PATH, but is useful when targeting a specific blender install, or when it's installed
                via a package manager such as flatpak. Setting it to "flatpak run --die-with-parent org.blender.Blender"
                might be required when using flatpaks. Defaults to None (system PATH).

        Raises:
            TimeoutError: raise if unable to discover spawned servers in `timeout` seconds and kill any spawned processes.

        Returns:
            procs: List of `subprocess.Popen` corresponding to all spawned servers.
            conns: List of connection setting for each server, where each element is a (hostname, port) tuple.
        """

    @staticmethod
    def spawn_registry() -> tuple[Process, rpyc.utils.registry.UDPRegistryClient]:
        """Spawn a registry server and client to aid in server discovery, or return cached result.
        While this method can be called directly, it will be invoked automatically by `discover` and `spawn`.

        Returns:
            registry: global registry server,
            client: global registry client
        """

    @staticmethod
    def discover() -> list[tuple[str, int]]:
        """Discover any `BlenderServer`s that are already running and return their connection parameters.
        Note: A discoverable server might already be in use and can refuse connection attempts.

        Returns:
            conns: List of connection setting for each server, where each element is a (hostname, port) tuple.
        """

class BlenderService(rpyc.Service):
    """Server-side API to interact with blender and render novel views.

    Most of the methods of a `BlenderClient` instance are remote procedure calls to
    a connected blender service. These methods are prefixed by `exposed_`.
    """

    ALIASES: list[str]
    log: logging.Logger
    initialized: bool

    def __init__(self) -> None: ...
    def on_connect(self, conn: rpyc.Connection) -> None:
        """Called when the connection is established

        Args:
            conn (rpyc.Connection): Connection object
        """

    def on_disconnect(self, conn: rpyc.Connection) -> None:
        """Called when the connection has already terminated. Resets blender runtime.
        (must not perform any IO on the connection)

        Args:
            conn (rpyc.Connection): Connection object
        """

    def reset(self) -> None:
        """Cleans up and resets blender runtime.

        De-initialize service by restoring blender to it's startup state,
        ensuring any cached attributes are cleaned (otherwise objects will be stale),
        and resetting any instance variables that were previously initialized.
        """

    @property
    @require_initialized_service
    def scene(self) -> bpy.types.Scene:
        """Get current blender scene"""

    @property
    @require_initialized_service
    def tree(self) -> bpy.types.CompositorNodeTree:
        """Get current scene's node tree"""

    @functools.cached_property
    @require_initialized_service
    def render_layers(self) -> bpy.types.CompositorNodeRLayers:
        """Get and cache render layers node, create one if needed."""

    @property
    @require_initialized_service
    def view_layer(self) -> bpy.types.ViewLayer:
        """Get current view layer"""

    @functools.cached_property
    @require_initialized_service
    def camera(self) -> bpy.types.Camera:
        """Get and cache active camera"""

    @require_initialized_service
    def get_parents(self, obj: bpy.types.Object) -> list[bpy.types.Object]:
        """Recursively retrieves parent objects of a given object in Blender

        Args:
            obj: Object to find parent of.

        Return:
            List of parent objects of obj.
        """

    def exposed_with_logger(self, log: logging.Logger) -> None:
        """Use supplied logger, if logger is initialized in client, messages will log to the client.

        Args:
            log (logging.Logger): Logger to use for messages
        """
    blend_file: Path
    root_path: Path
    depth_path: bpy.types.CompositorNodeOutputFile | None
    normal_path: bpy.types.CompositorNodeOutputFile | None
    flow_path: bpy.types.CompositorNodeOutputFile | None
    segmentation_path: bpy.types.CompositorNodeOutputFile | None
    depth_extension: str
    unbind_camera: bool
    use_animation: bool
    disabled_fcurves: set[bpy.types.Action]

    def exposed_initialize(self, blend_file: str | os.PathLike, root_path: str | os.PathLike, **kwargs) -> None:
        """Initialize BlenderService and load blendfile.

        Args:
            blend_file (str | os.PathLike): path of scene file to load.
            root_path (str | os.PathLike): path at which to save rendered results.
        """

    @require_initialized_service
    def exposed_empty_transforms(self) -> dict[str, Any]:
        """Return a dictionary with camera intrinsics. Forms the basis of
        a `transforms.json` file, but contains no frame data.

        Returns:
            empty_transforms: mapping containing camera parameters.
        """

    @require_initialized_service
    def exposed_original_fps(self) -> int: ...
    @require_initialized_service
    def exposed_animation_range(self) -> range:
        """Get animation range of current scene as range(start, end+1, step)."""

    @require_initialized_service
    def exposed_animation_range_tuple(self) -> tuple[int, int, int]:
        """Get animation range of current scene as a tuple of (start, end, step)."""

    @require_initialized_service
    def exposed_include_depths(self, debug: bool = True, file_format: str = "OPEN_EXR", exr_codec: str = "ZIP") -> None:
        """Sets up Blender compositor to include depth map for rendered images.

        Args:
            debug (bool, optional): if true, colorized depth maps, helpful for quick visualizations,
                will be generated alongside ground-truth depth maps. Defaults to True.
            file_format (str, optional): format of depth maps, one of "OPEN_EXR" or "HDR". The former
                is lossless, but can require significant storage, the later is lossy and more compressed.
                If depth is needed to compute scene-flow, use open-exr. Defaults to "OPEN_EXR".
            exr_codec (str, optional): codec used to compress exr file. Only used when `file_format="OPEN_EXR"`,
                options vary depending on the version of Blender, with the following being broadly available:
                ('NONE', 'PXR24', 'ZIP', 'PIZ', 'RLE', 'ZIPS', 'DWAA', 'DWAB'). Defaults to "ZIP".

        Note:
            The debug colormap is re-normalized on a per-frame basis, to visually
            compare across frames, apply colorization after rendering using the CLI.

        Raises:
            ValueError: raise if file-format nor understood.
        """

    @require_initialized_service
    def exposed_include_normals(self, debug: bool = True, exr_codec: str = "ZIP") -> None:
        """Sets up Blender compositor to include normal map for rendered images.

        Args:
            debug (bool, optional): if true, colorized normal maps will also be generated with each vector
                component being remapped from [-1, 1] to [0-255] with xyz becoming rgb. Defaults to True.
            exr_codec (str, optional): codec used to compress exr file. Options vary depending on the version of Blender,
                with the following being broadly available: ('NONE', 'PXR24', 'ZIP', 'PIZ', 'RLE', 'ZIPS', 'DWAA', 'DWAB').
                Defaults to "ZIP".
        """

    @require_initialized_service
    def exposed_include_flows(self, direction: str = "forward", debug: bool = True, exr_codec: str = "ZIP") -> None:
        """Sets up Blender compositor to include optical flow for rendered images.

        Args:
            direction (str, optional): One of 'forward', 'backward' or 'both'. Direction of flow to colorize
                for debug visualization. Only used when debug is true, otherwise both forward and backward
                flows are saved. Defaults to "forward".
            debug (bool, optional): If true, also save debug visualizations of flow. Defaults to True.
            exr_codec (str, optional): codec used to compress exr file. Options vary depending on the version of Blender,
                with the following being broadly available: ('NONE', 'PXR24', 'ZIP', 'PIZ', 'RLE', 'ZIPS', 'DWAA', 'DWAB').
                Defaults to "ZIP".

        Note:
            The debug colormap is re-normalized on a per-frame basis, to visually
            compare across frames, apply colorization after rendering using the CLI.

        Raises:
            ValueError: raised when `direction` is not understood.
            RuntimeError: raised when motion blur is enabled as flow cannot be computed.
        """

    @require_initialized_service
    def exposed_include_segmentations(
        self, shuffle: bool = True, debug: bool = True, seed: int = 1234, exr_codec: str = "ZIP"
    ) -> None:
        """Sets up Blender compositor to include segmentation maps for rendered images.

        The debug visualization simply assigns a color to each object ID by mapping the
        objects ID value to a hue using a HSV node with saturation=1 and value=1 (except
        for the background which will have a value of 0 to ensure it is black).

        Args:
            shuffle (bool, optional): shuffle debug colors, helps differentiate object instances. Defaults to True.
            debug (bool, optional): If true, also save debug visualizations of segmentation. Defaults to True.
            seed (int, optional): random seed used when shuffling colors. Defaults to 1234.
            exr_codec (str, optional): codec used to compress exr file. Options vary depending on the version of Blender,
                with the following being broadly available: ('NONE', 'PXR24', 'ZIP', 'PIZ', 'RLE', 'ZIPS', 'DWAA', 'DWAB').
                Defaults to "ZIP".

        Raises:
            RuntimeError: raised when not using CYCLES, as other renderers do not support a segmentation pass.
        """

    @require_initialized_service
    def exposed_load_addons(self, *addons: str) -> None:
        """Load blender addons by name (case-insensitive)"""

    @require_initialized_service
    def exposed_set_resolution(
        self, height: tuple[int] | list[int] | int | None = None, width: int | None = None
    ) -> None:
        """Set frame resolution (height, width) in pixels.

        If a single tuple is passed, instead of using keyword arguments, it will be parsed as (height, width).
        """

    @require_initialized_service
    def exposed_image_settings(
        self, file_format: str | None = None, bit_depth: int | None = None, color_mode: str | None = None
    ) -> None:
        """Set the render's output format and bit-depth.
        Useful for linear intensity renders, using "OPEN_EXR" and 32 or 16 bits.
        """

    @require_initialized_service
    def exposed_use_motion_blur(self, enable: bool) -> None:
        """Enable/disable motion blur."""

    @require_initialized_service
    def exposed_use_animations(self, enable: bool) -> None:
        """Enable/disable all animations."""

    @require_initialized_service
    def exposed_cycles_settings(
        self,
        device_type: str | None = None,
        use_cpu: bool | None = None,
        adaptive_threshold: float | None = None,
        max_samples: int | None = None,
        use_denoising: bool | None = None,
    ) -> list[str]:
        """Enables/activates cycles render devices and settings.

        Note: A default arguments of `None` means do not change setting inherited from blendfile.

        Args:
            device_type (str, optional): Name of device to use, one of "cpu", "cuda", "optix", "metal", etc.
                See `blender docs <https://docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html>`_
                for full list. Defaults to None.
            use_cpu (bool, optional): Boolean flag to enable CPUs alongside GPU devices. Defaults to None.
            adaptive_threshold (float, optional): Set noise threshold upon which to stop taking samples. Defaults to None.
            max_samples (int, optional): Maximum number of samples per pixel to take. Defaults to None.
            use_denoising (bool, optional): If enabled, a denoising pass will be used. Defaults to None.

        Raises:
            RuntimeError: raised when no devices are found.
            ValueError: raised when setting `use_cpu` is required.

        Returns:
            devices: List of activated devices.
        """

    @require_initialized_service
    def exposed_unbind_camera(self) -> None:
        """Remove constraints, animations and parents from main camera.

        Note: In order to undo this, you'll need to re-initialize.
        """

    @require_initialized_service
    def exposed_move_keyframes(self, scale: float = 1.0, shift: float = 0.0) -> None:
        """Adjusts keyframes in Blender animations, keypoints are first scaled then shifted.

        Args:
            scale (float, optional): Factor used to rescale keyframe positions along x-axis. Defaults to 1.0.
            shift (float, optional): Factor used to shift keyframe positions along x-axis. Defaults to 0.0.

        Raises:
            RuntimeError: raised if trying to move keyframes beyond blender's limits.
        """

    @require_initialized_service
    def exposed_set_current_frame(self, frame_number: int) -> None:
        """Set current frame number. This might advance any animations."""

    @require_initialized_service
    def exposed_camera_extrinsics(self) -> npt.NDArray[np.floating]:
        """Get the 4x4 transform matrix encoding the current camera pose."""

    @require_initialized_service
    def exposed_camera_intrinsics(self) -> npt.NDArray[np.floating]:
        """Get the 3x3 camera intrinsics matrix for active camera,
        which defines how 3D points are projected onto 2D.

        Note: Assumes pinhole camera model.

        Returns:
            Camera intrinsics matrix based on camera properties.
        """

    @require_initialized_service
    @validate_camera_moved
    def exposed_position_camera(
        self,
        location: npt.ArrayLike | None = None,
        rotation: npt.ArrayLike | None = None,
        look_at: npt.ArrayLike | None = None,
        in_order: bool = True,
    ) -> None:
        """Positions and orients camera in Blender scene according to specified parameters.

        Note: Only one of `look_at` or `rotation` can be set at once.

        Args:
            location (npt.ArrayLike, optional): Location to place camera in 3D space. Defaults to none.
            rotation (npt.ArrayLike, optional): Rotation matrix for camera. Defaults to none.
            look_at (npt.ArrayLike, optional): Location to point camera. Defaults to none.
            in_order (bool, optional): If set, assume current camera pose is from previous/next
                frame and ensure new rotation set by `look_at` is compatible with current position.
                Without this, a rotations will stay in the [-pi, pi] range and this wrapping will
                mess up interpolations. Only used when `look_at` is set. Defaults to True.

        Raises:
            ValueError: raised if camera orientation is over-defined.
        """

    @require_initialized_service
    @validate_camera_moved
    def exposed_rotate_camera(self, angle: float) -> None:
        """Rotate camera around it's optical axis, relative to current orientation.

        Args:
            angle: Relative amount to rotate by (clockwise, in radians).
        """

    @require_initialized_service
    def exposed_set_camera_keyframe(self, frame_num: int, matrix: npt.ArrayLike | None = None) -> None:
        """Set camera keyframe at given frame number.
        If camera matrix is not supplied, currently set camera position/rotation/scale will be used,
        this allows users to set camera position using `position_camera` and `rotate_camera`.

        Args:
            frame_num (int): index of frame to set keyframe for.
            matrix (npt.ArrayLike | None, optional): 4x4 camera transform, if not supplied,
                use current camera matrix. Defaults to None.
        """

    @require_initialized_service
    def exposed_set_animation_range(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> None:
        """Set animation range for scene.

        Args:
            start (int | None, optional): frame start, inclusive. Defaults to None.
            stop (int | None, optional): frame stop, exclusive. Defaults to None.
            step (int | None, optional): frame interval. Defaults to None.
        """

    @require_initialized_service
    def exposed_render_current_frame(self, allow_skips: bool = True, dry_run: bool = False) -> dict[str, Any]:
        """Generates a single frame in Blender at the current camera location,
        return the file paths for that frame, potentially including depth, normals, etc.

        Args:
            allow_skips (bool, optional): if true, blender will not re-render and overwrite existing frames.
                This does not however apply to depth/normals/etc, which cannot be skipped. Defaults to True.
            dry_run (bool, optional): if true, nothing will be rendered at all. Defaults to False.

        Returns:
            dictionary containing paths to rendered frames for this index and camera pose.
        """

    @require_initialized_service
    def exposed_render_frame(self, frame_number: int, allow_skips: bool = True, dry_run: bool = False) -> dict[str, Any]:
        """Same as first setting current frame then rendering it.

        Warning:
            Calling this has the side-effect of changing the current frame.
        """

    @require_initialized_service
    def exposed_render_frames(
        self,
        frame_numbers: Iterable[int],
        allow_skips: bool = True,
        dry_run: bool = False,
        update_fn: UpdateFn | None = None,
    ) -> dict[str, Any]:
        """Render all requested frames and return associated transforms dictionary.

        Args:
            frame_numbers (Iterable[int]): frames to render.
            update_fn (UpdateFn, optional): callback function to track render progress. Will first be called with `total` kwarg,
                indicating number of steps to be taken, then will be called with `advance=1` at every step. Closely mirrors the
                `rich.Progress API <https://rich.readthedocs.io/en/stable/reference/progress.html#rich.progress.Progress.update>`_.
                Defaults to None.
            See `exposed_render_current_frame` for other arguments.

        Raises:
            RuntimeError: raised if trying to render frames beyond blender's limits.

        Returns:
            transforms dictionary
        """

    @require_initialized_service
    def exposed_render_animation(
        self,
        frame_start: int | None = None,
        frame_end: int | None = None,
        frame_step: int | None = None,
        allow_skips: bool = True,
        dry_run: bool = False,
        update_fn: UpdateFn | None = None,
    ) -> dict[str, Any]:
        """Determines frame range to render, sets camera positions and orientations, and renders all frames in animation range.

        Note: All frame start/end/step arguments are absolute quantities, applied after any keyframe moves.
              If the animation is from (1-100) and you've scaled it by calling `move_keyframes(scale=2.0)`
              then calling `render_animation(frame_start=1, frame_end=100)` will only render half of the animation.
              By default the whole animation will render when no start/end and step values are set.

        Args:
            frame_start (int, optional): Starting index (inclusive) of frames to render as seen in blender. Defaults to None, meaning value from `.blend` file.
            frame_end (int, optional): Ending index (inclusive) of frames to render as seen in blender. Defaults to None, meaning value from `.blend` file.
            frame_step (int, optional): Skip every nth frame. Defaults to None, meaning value from `.blend` file.
            allow_skips (bool, optional): Same as `(exposed_)render_frame_here`.
            dry_run (bool, optional): Same as `(exposed_)render_frame_here`.
            update_fn (UpdateFn, optional): Same as `(exposed_)render_frames`.

        Raises:
            ValueError: raised if scene and camera are entirely static.

        Returns:
            transforms dictionary
        """

    @require_initialized_service
    def exposed_save_file(self, path: str | os.PathLike) -> None:
        """Save opened blender file. This is useful for introspecting the state of the compositor/scene/etc."""

class BlenderClient:
    """Client-side API to interact with blender and render novel views.

    The `BlenderClient` is responsible for communicating with (and potentially spawning)
    separate `BlenderServer`s that will actually perform the rendering via a `BlenderService`.

    The client acts as a context manager, it will connect to it's server when the context is
    entered and cleanly disconnect and close the connection in case of errors or when exiting
    the with-block.

    Many useful methods to interact with blender are provided, such as `set_resolution` or
    `render_animation`. These methods are dynamically generated when the client connects to
    the server. Available methods are directly inherited from `BlenderService` (or whichever
    service the server is exposing), specifically any service method starting with `exposed_`
    will be accessible to the client at runtime. for example, `BlenderClient.include_depths`
    is a remote procedure call to `BlenderService.exposed_include_depths`.
    """

    addr: tuple[str, int]
    conn: rpyc.Connection | None
    awaitable: rpyc.AsyncResult | None
    process: subprocess.Popen | None
    timeout: float

    def __init__(self, addr: tuple[str, int], timeout: float = 10.0) -> None:
        """Initialize a client with known address of server.
        Note: Using `auto_connect` or `spawn` is often more convenient.

        Args:
            addr (tuple[str, int]): Connection tuple containing the hostname and port
            timeout (float, optional): Maximum time in seconds the client will attempt
                to connect to the server for before an error is thrown. Only used when
                entering context manager. Defaults to 10 seconds.
        """

    @classmethod
    def auto_connect(cls, timeout: float = 10.0) -> Self:
        """Automatically connect to available server.

        Use `BlenderServer.discover` to find available server within `timeout`.

        Note: This doesn't actually connect to the server instance, the connection happens
            when the context manager is entered. This simply creates a client instance with
            the connection settings (i.e: hostname, port) of an existing server. The connection
            might still fail when entering the with-block.

        Args:
            timeout (float, optional): try to discover server instance for `timeout`
                (in seconds) before giving up. Defaults to 10.0 seconds.

        Raises:
            TimeoutError: raise if unable to discover server in `timeout` seconds.

        Returns:
            client: client instance initialized with connection settings of existing server.
        """

    @classmethod
    @contextmanager
    def spawn(
        cls,
        timeout: float = -1.0,
        log_dir: str | os.PathLike | None = None,
        autoexec: bool = False,
        executable: str | os.PathLike | None = None,
    ) -> Iterator[Self]:
        """Spawn and connect to a blender server.
        The spawned process is accessible through the client's `process` attribute.

        Args:
            Same as `BlenderServer.spawn`.

        Returns:
            client: the connected client
        """

    @require_connected_client
    def render_animation_async(self, *args, **kwargs) -> rpyc.AsyncResult:
        """Asynchronously call `render_animation` and return an rpyc.AsyncResult.

        Parameters:
            Same as `BlendService.exposed_render_animation`

        Returns:
            result: AsyncResult encapsulating the return value of `render_animation`.
                After `wait`ing for the render to finish, it can be accessed using
                the `.value` attribute.
        """

    @require_connected_client
    def render_frames_async(self, *args, **kwargs) -> rpyc.AsyncResult:
        """Asynchronously call `render_frames` and return an rpyc.AsyncResult.

        Parameters:
            Same as `BlendService.exposed_render_frames`

        Returns:
            result: AsyncResult encapsulating the return value of `render_frames`.
                After `wait`ing for the render to finish, it can be accessed using
                the `.value` attribute.
        """

    def wait(self) -> None:
        """Block and await any async results."""

    def __enter__(self) -> Self: ...
    def __getattr__(self, name: str) -> Any: ...
    def __exit__(
        self, type: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None
    ) -> None: ...
    def with_logger(self, log: logging.Logger) -> None:
        """Use supplied logger, if logger is initialized in client, messages will log to the client.

        Args:
            log (logging.Logger): Logger to use for messages
        """

    def initialize(self, blend_file: str | os.PathLike, root_path: str | os.PathLike, **kwargs) -> None:
        """Initialize BlenderService and load blendfile.

        Args:
            blend_file (str | os.PathLike): path of scene file to load.
            root_path (str | os.PathLike): path at which to save rendered results.
        """

    @require_initialized_service
    def empty_transforms(self) -> dict[str, Any]:
        """Return a dictionary with camera intrinsics. Forms the basis of
        a `transforms.json` file, but contains no frame data.

        Returns:
            empty_transforms: mapping containing camera parameters.
        """

    @require_initialized_service
    def original_fps(self) -> int: ...
    @require_initialized_service
    def animation_range(self) -> range:
        """Get animation range of current scene as range(start, end+1, step)."""

    @require_initialized_service
    def animation_range_tuple(self) -> tuple[int, int, int]:
        """Get animation range of current scene as a tuple of (start, end, step)."""

    @require_initialized_service
    def include_depths(self, debug: bool = True, file_format: str = "OPEN_EXR", exr_codec: str = "ZIP") -> None:
        """Sets up Blender compositor to include depth map for rendered images.

        Args:
            debug (bool, optional): if true, colorized depth maps, helpful for quick visualizations,
                will be generated alongside ground-truth depth maps. Defaults to True.
            file_format (str, optional): format of depth maps, one of "OPEN_EXR" or "HDR". The former
                is lossless, but can require significant storage, the later is lossy and more compressed.
                If depth is needed to compute scene-flow, use open-exr. Defaults to "OPEN_EXR".
            exr_codec (str, optional): codec used to compress exr file. Only used when `file_format="OPEN_EXR"`,
                options vary depending on the version of Blender, with the following being broadly available:
                ('NONE', 'PXR24', 'ZIP', 'PIZ', 'RLE', 'ZIPS', 'DWAA', 'DWAB'). Defaults to "ZIP".

        Note:
            The debug colormap is re-normalized on a per-frame basis, to visually
            compare across frames, apply colorization after rendering using the CLI.

        Raises:
            ValueError: raise if file-format nor understood.
        """

    @require_initialized_service
    def include_normals(self, debug: bool = True, exr_codec: str = "ZIP") -> None:
        """Sets up Blender compositor to include normal map for rendered images.

        Args:
            debug (bool, optional): if true, colorized normal maps will also be generated with each vector
                component being remapped from [-1, 1] to [0-255] with xyz becoming rgb. Defaults to True.
            exr_codec (str, optional): codec used to compress exr file. Options vary depending on the version of Blender,
                with the following being broadly available: ('NONE', 'PXR24', 'ZIP', 'PIZ', 'RLE', 'ZIPS', 'DWAA', 'DWAB').
                Defaults to "ZIP".
        """

    @require_initialized_service
    def include_flows(self, direction: str = "forward", debug: bool = True, exr_codec: str = "ZIP") -> None:
        """Sets up Blender compositor to include optical flow for rendered images.

        Args:
            direction (str, optional): One of 'forward', 'backward' or 'both'. Direction of flow to colorize
                for debug visualization. Only used when debug is true, otherwise both forward and backward
                flows are saved. Defaults to "forward".
            debug (bool, optional): If true, also save debug visualizations of flow. Defaults to True.
            exr_codec (str, optional): codec used to compress exr file. Options vary depending on the version of Blender,
                with the following being broadly available: ('NONE', 'PXR24', 'ZIP', 'PIZ', 'RLE', 'ZIPS', 'DWAA', 'DWAB').
                Defaults to "ZIP".

        Note:
            The debug colormap is re-normalized on a per-frame basis, to visually
            compare across frames, apply colorization after rendering using the CLI.

        Raises:
            ValueError: raised when `direction` is not understood.
            RuntimeError: raised when motion blur is enabled as flow cannot be computed.
        """

    @require_initialized_service
    def include_segmentations(
        self, shuffle: bool = True, debug: bool = True, seed: int = 1234, exr_codec: str = "ZIP"
    ) -> None:
        """Sets up Blender compositor to include segmentation maps for rendered images.

        The debug visualization simply assigns a color to each object ID by mapping the
        objects ID value to a hue using a HSV node with saturation=1 and value=1 (except
        for the background which will have a value of 0 to ensure it is black).

        Args:
            shuffle (bool, optional): shuffle debug colors, helps differentiate object instances. Defaults to True.
            debug (bool, optional): If true, also save debug visualizations of segmentation. Defaults to True.
            seed (int, optional): random seed used when shuffling colors. Defaults to 1234.
            exr_codec (str, optional): codec used to compress exr file. Options vary depending on the version of Blender,
                with the following being broadly available: ('NONE', 'PXR24', 'ZIP', 'PIZ', 'RLE', 'ZIPS', 'DWAA', 'DWAB').
                Defaults to "ZIP".

        Raises:
            RuntimeError: raised when not using CYCLES, as other renderers do not support a segmentation pass.
        """

    @require_initialized_service
    def load_addons(self, *addons: str) -> None:
        """Load blender addons by name (case-insensitive)"""

    @require_initialized_service
    def set_resolution(self, height: tuple[int] | list[int] | int | None = None, width: int | None = None) -> None:
        """Set frame resolution (height, width) in pixels.

        If a single tuple is passed, instead of using keyword arguments, it will be parsed as (height, width).
        """

    @require_initialized_service
    def image_settings(
        self, file_format: str | None = None, bit_depth: int | None = None, color_mode: str | None = None
    ) -> None:
        """Set the render's output format and bit-depth.
        Useful for linear intensity renders, using "OPEN_EXR" and 32 or 16 bits.
        """

    @require_initialized_service
    def use_motion_blur(self, enable: bool) -> None:
        """Enable/disable motion blur."""

    @require_initialized_service
    def use_animations(self, enable: bool) -> None:
        """Enable/disable all animations."""

    @require_initialized_service
    def cycles_settings(
        self,
        device_type: str | None = None,
        use_cpu: bool | None = None,
        adaptive_threshold: float | None = None,
        max_samples: int | None = None,
        use_denoising: bool | None = None,
    ) -> list[str]:
        """Enables/activates cycles render devices and settings.

        Note: A default arguments of `None` means do not change setting inherited from blendfile.

        Args:
            device_type (str, optional): Name of device to use, one of "cpu", "cuda", "optix", "metal", etc.
                See `blender docs <https://docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html>`_
                for full list. Defaults to None.
            use_cpu (bool, optional): Boolean flag to enable CPUs alongside GPU devices. Defaults to None.
            adaptive_threshold (float, optional): Set noise threshold upon which to stop taking samples. Defaults to None.
            max_samples (int, optional): Maximum number of samples per pixel to take. Defaults to None.
            use_denoising (bool, optional): If enabled, a denoising pass will be used. Defaults to None.

        Raises:
            RuntimeError: raised when no devices are found.
            ValueError: raised when setting `use_cpu` is required.

        Returns:
            devices: List of activated devices.
        """

    @require_initialized_service
    def unbind_camera(self) -> None:
        """Remove constraints, animations and parents from main camera.

        Note: In order to undo this, you'll need to re-initialize.
        """

    @require_initialized_service
    def move_keyframes(self, scale: float = 1.0, shift: float = 0.0) -> None:
        """Adjusts keyframes in Blender animations, keypoints are first scaled then shifted.

        Args:
            scale (float, optional): Factor used to rescale keyframe positions along x-axis. Defaults to 1.0.
            shift (float, optional): Factor used to shift keyframe positions along x-axis. Defaults to 0.0.

        Raises:
            RuntimeError: raised if trying to move keyframes beyond blender's limits.
        """

    @require_initialized_service
    def set_current_frame(self, frame_number: int) -> None:
        """Set current frame number. This might advance any animations."""

    @require_initialized_service
    def camera_extrinsics(self) -> npt.NDArray[np.floating]:
        """Get the 4x4 transform matrix encoding the current camera pose."""

    @require_initialized_service
    def camera_intrinsics(self) -> npt.NDArray[np.floating]:
        """Get the 3x3 camera intrinsics matrix for active camera,
        which defines how 3D points are projected onto 2D.

        Note: Assumes pinhole camera model.

        Returns:
            Camera intrinsics matrix based on camera properties.
        """

    @require_initialized_service
    @validate_camera_moved
    def position_camera(
        self,
        location: npt.ArrayLike | None = None,
        rotation: npt.ArrayLike | None = None,
        look_at: npt.ArrayLike | None = None,
        in_order: bool = True,
    ) -> None:
        """Positions and orients camera in Blender scene according to specified parameters.

        Note: Only one of `look_at` or `rotation` can be set at once.

        Args:
            location (npt.ArrayLike, optional): Location to place camera in 3D space. Defaults to none.
            rotation (npt.ArrayLike, optional): Rotation matrix for camera. Defaults to none.
            look_at (npt.ArrayLike, optional): Location to point camera. Defaults to none.
            in_order (bool, optional): If set, assume current camera pose is from previous/next
                frame and ensure new rotation set by `look_at` is compatible with current position.
                Without this, a rotations will stay in the [-pi, pi] range and this wrapping will
                mess up interpolations. Only used when `look_at` is set. Defaults to True.

        Raises:
            ValueError: raised if camera orientation is over-defined.
        """

    @require_initialized_service
    @validate_camera_moved
    def rotate_camera(self, angle: float) -> None:
        """Rotate camera around it's optical axis, relative to current orientation.

        Args:
            angle: Relative amount to rotate by (clockwise, in radians).
        """

    @require_initialized_service
    def set_camera_keyframe(self, frame_num: int, matrix: npt.ArrayLike | None = None) -> None:
        """Set camera keyframe at given frame number.
        If camera matrix is not supplied, currently set camera position/rotation/scale will be used,
        this allows users to set camera position using `position_camera` and `rotate_camera`.

        Args:
            frame_num (int): index of frame to set keyframe for.
            matrix (npt.ArrayLike | None, optional): 4x4 camera transform, if not supplied,
                use current camera matrix. Defaults to None.
        """

    @require_initialized_service
    def set_animation_range(self, start: int | None = None, stop: int | None = None, step: int | None = None) -> None:
        """Set animation range for scene.

        Args:
            start (int | None, optional): frame start, inclusive. Defaults to None.
            stop (int | None, optional): frame stop, exclusive. Defaults to None.
            step (int | None, optional): frame interval. Defaults to None.
        """

    @require_initialized_service
    def render_current_frame(self, allow_skips: bool = True, dry_run: bool = False) -> dict[str, Any]:
        """Generates a single frame in Blender at the current camera location,
        return the file paths for that frame, potentially including depth, normals, etc.

        Args:
            allow_skips (bool, optional): if true, blender will not re-render and overwrite existing frames.
                This does not however apply to depth/normals/etc, which cannot be skipped. Defaults to True.
            dry_run (bool, optional): if true, nothing will be rendered at all. Defaults to False.

        Returns:
            dictionary containing paths to rendered frames for this index and camera pose.
        """

    @require_initialized_service
    def render_frame(self, frame_number: int, allow_skips: bool = True, dry_run: bool = False) -> dict[str, Any]:
        """Same as first setting current frame then rendering it.

        Warning:
            Calling this has the side-effect of changing the current frame.
        """

    @require_initialized_service
    def render_frames(
        self,
        frame_numbers: Iterable[int],
        allow_skips: bool = True,
        dry_run: bool = False,
        update_fn: UpdateFn | None = None,
    ) -> dict[str, Any]:
        """Render all requested frames and return associated transforms dictionary.

        Args:
            frame_numbers (Iterable[int]): frames to render.
            update_fn (UpdateFn, optional): callback function to track render progress. Will first be called with `total` kwarg,
                indicating number of steps to be taken, then will be called with `advance=1` at every step. Closely mirrors the
                `rich.Progress API <https://rich.readthedocs.io/en/stable/reference/progress.html#rich.progress.Progress.update>`_.
                Defaults to None.
            See `exposed_render_current_frame` for other arguments.

        Raises:
            RuntimeError: raised if trying to render frames beyond blender's limits.

        Returns:
            transforms dictionary
        """

    @require_initialized_service
    def render_animation(
        self,
        frame_start: int | None = None,
        frame_end: int | None = None,
        frame_step: int | None = None,
        allow_skips: bool = True,
        dry_run: bool = False,
        update_fn: UpdateFn | None = None,
    ) -> dict[str, Any]:
        """Determines frame range to render, sets camera positions and orientations, and renders all frames in animation range.

        Note: All frame start/end/step arguments are absolute quantities, applied after any keyframe moves.
              If the animation is from (1-100) and you've scaled it by calling `move_keyframes(scale=2.0)`
              then calling `render_animation(frame_start=1, frame_end=100)` will only render half of the animation.
              By default the whole animation will render when no start/end and step values are set.

        Args:
            frame_start (int, optional): Starting index (inclusive) of frames to render as seen in blender. Defaults to None, meaning value from `.blend` file.
            frame_end (int, optional): Ending index (inclusive) of frames to render as seen in blender. Defaults to None, meaning value from `.blend` file.
            frame_step (int, optional): Skip every nth frame. Defaults to None, meaning value from `.blend` file.
            allow_skips (bool, optional): Same as `(exposed_)render_frame_here`.
            dry_run (bool, optional): Same as `(exposed_)render_frame_here`.
            update_fn (UpdateFn, optional): Same as `(exposed_)render_frames`.

        Raises:
            ValueError: raised if scene and camera are entirely static.

        Returns:
            transforms dictionary
        """

    @require_initialized_service
    def save_file(self, path: str | os.PathLike) -> None:
        """Save opened blender file. This is useful for introspecting the state of the compositor/scene/etc."""

class BlenderClients(tuple):
    """Collection of `BlenderClient` instances.

    Most methods in this class simply call the equivalent method of each client, that is,
    calling `clients.set_resolution` is equivalent to calling `set_resolution` for each
    client in clients. Some special methods, namely the `render_frames` and `render_animation`
    methods will instead distribute the rendering load to all clients.

    Finally, entering each client's context-manager, and closing each client connection
    is ensured by using this class' context-manager.
    """

    def __new__(cls, *objs: Iterator[BlenderClient | tuple[str, int]]) -> Self: ...
    stack: ExitStack

    def __init__(self, *objs) -> None:
        """Initialize collection of `BlenderClient` from iterable of clients, or their connection settings.

        Args:
            *objs (Iterator[BlenderClient | tuple[str, int]]): `BlenderClient` instances or their hostnames and ports.
        """

    def __enter__(self) -> Self: ...
    def __exit__(
        self, type: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None
    ) -> bool | None: ...
    @classmethod
    @contextmanager
    def spawn(
        cls,
        jobs: int = 1,
        timeout: float = -1.0,
        log_dir: str | os.PathLike | None = None,
        autoexec: bool = False,
        executable: str | os.PathLike | None = None,
    ) -> Iterator[Self]:
        """Spawn and connect to one or more blender servers.
        The spawned processes are accessible through the client's `process` attribute.

        Args:
            Same as `BlenderServer.spawn`.

        Returns:
            clients: the connected clients
        """

    @staticmethod
    @contextmanager
    def pool(
        jobs: int = 1,
        timeout: float = -1.0,
        log_dir: str | os.PathLike | None = None,
        autoexec: bool = False,
        executable: str | os.PathLike | None = None,
        conns: list[tuple[str, int]] | None = None,
    ) -> multiprocess.Pool:
        """Spawns a multiprocessing-like worker pool, each with their own `BlenderClient` instance.
        The function supplied to pool.map/imap/starmap and their async variants will be automagically
        passed a client instance as their first argument that they can use for rendering.

        Example:
            .. code-block:: python

                def render(client, blend_file):
                    root = Path("renders") / Path(blend_file).stem
                    client.initialize(blend_file, root)
                    client.render_animation()

                if __name__ == "__main__":
                    with BlenderClients.pool(2) as pool:
                        pool.map(render, ["monkey.blend", "cube.blend", "metaballs.blend"])

        Note:
            Here we use `multiprocess` instead of the builtin multiprocessing library to take
            advantage of the more advanced dill serialization (as opposed to the standard pickling).

        Args:
            conns: List of connection tuples containing the hostnames and ports of existing servers.
                If specified, the pool will use these servers (and `jobs` and other spawn arguments will
                be ignored) instead of spawning new ones.

            For other arguments, see `BlenderServer.spawn`

        Returns:
            A `multiprocess.Pool` instance which has had it's applicator methods (map/imap/starmap/etc)
            monkey-patched to inject a client instance as first argument.
        """

    @require_connected_clients
    def common_animation_range(self) -> range:
        """Get animation range shared by all clients as range(start, end+1, step).

        Raises:
            RuntimeError: animation ranges for all clients are expected to be the same.
        """

    @require_connected_clients
    def common_animation_range_tuple(self) -> tuple[int, int, int]:
        """Get animation range shared by all clients as a tuple of (start, end, step).

        Raises:
            RuntimeError: animation ranges for all clients are expected to be the same.
        """

    @require_connected_clients
    def render_frames(
        self,
        frame_numbers: Collection[int],
        allow_skips: bool = True,
        dry_run: bool = False,
        update_fn: UpdateFn | None = None,
    ) -> dict[str, Any]: ...
    @require_connected_clients
    def render_animation(
        self,
        frame_start: int | None = None,
        frame_end: int | None = None,
        frame_step: int | None = None,
        allow_skips: bool = True,
        dry_run: bool = False,
        update_fn: UpdateFn | None = None,
    ) -> dict[str, Any]: ...
    @require_connected_clients
    def save_file(self, path: str | os.PathLike) -> None:
        """Save opened blender file. This is useful for introspecting the state of the compositor/scene/etc.

        Note: Only saves file once (from a single connected client), assumes all clients have
        been initialized in the same manner.
        """

    def wait(self) -> None:
        """Wait for all clients at once."""

    def with_logger(self, log: logging.Logger) -> None:
        """Use supplied logger, if logger is initialized in client, messages will log to the client.

        Args:
            log (logging.Logger): Logger to use for messages
        """

    def initialize(self, blend_file: str | os.PathLike, root_path: str | os.PathLike, **kwargs) -> None:
        """Initialize BlenderService and load blendfile.

        Args:
            blend_file (str | os.PathLike): path of scene file to load.
            root_path (str | os.PathLike): path at which to save rendered results.
        """

    @require_initialized_service
    def empty_transforms(self) -> tuple[dict[str, Any],]:
        """Return a dictionary with camera intrinsics. Forms the basis of
        a `transforms.json` file, but contains no frame data.

        Returns:
            empty_transforms: mapping containing camera parameters.
        """

    @require_initialized_service
    def original_fps(self) -> tuple[int,]: ...
    @require_initialized_service
    def animation_range(self) -> tuple[range,]:
        """Get animation range of current scene as range(start, end+1, step)."""

    @require_initialized_service
    def animation_range_tuple(self) -> tuple[tuple[int, int, int],]:
        """Get animation range of current scene as a tuple of (start, end, step)."""

    @require_initialized_service
    def include_depths(self, debug: bool = True, file_format: str = "OPEN_EXR", exr_codec: str = "ZIP") -> None:
        """Sets up Blender compositor to include depth map for rendered images.

        Args:
            debug (bool, optional): if true, colorized depth maps, helpful for quick visualizations,
                will be generated alongside ground-truth depth maps. Defaults to True.
            file_format (str, optional): format of depth maps, one of "OPEN_EXR" or "HDR". The former
                is lossless, but can require significant storage, the later is lossy and more compressed.
                If depth is needed to compute scene-flow, use open-exr. Defaults to "OPEN_EXR".
            exr_codec (str, optional): codec used to compress exr file. Only used when `file_format="OPEN_EXR"`,
                options vary depending on the version of Blender, with the following being broadly available:
                ('NONE', 'PXR24', 'ZIP', 'PIZ', 'RLE', 'ZIPS', 'DWAA', 'DWAB'). Defaults to "ZIP".

        Note:
            The debug colormap is re-normalized on a per-frame basis, to visually
            compare across frames, apply colorization after rendering using the CLI.

        Raises:
            ValueError: raise if file-format nor understood.
        """

    @require_initialized_service
    def include_normals(self, debug: bool = True, exr_codec: str = "ZIP") -> None:
        """Sets up Blender compositor to include normal map for rendered images.

        Args:
            debug (bool, optional): if true, colorized normal maps will also be generated with each vector
                component being remapped from [-1, 1] to [0-255] with xyz becoming rgb. Defaults to True.
            exr_codec (str, optional): codec used to compress exr file. Options vary depending on the version of Blender,
                with the following being broadly available: ('NONE', 'PXR24', 'ZIP', 'PIZ', 'RLE', 'ZIPS', 'DWAA', 'DWAB').
                Defaults to "ZIP".
        """

    @require_initialized_service
    def include_flows(self, direction: str = "forward", debug: bool = True, exr_codec: str = "ZIP") -> None:
        """Sets up Blender compositor to include optical flow for rendered images.

        Args:
            direction (str, optional): One of 'forward', 'backward' or 'both'. Direction of flow to colorize
                for debug visualization. Only used when debug is true, otherwise both forward and backward
                flows are saved. Defaults to "forward".
            debug (bool, optional): If true, also save debug visualizations of flow. Defaults to True.
            exr_codec (str, optional): codec used to compress exr file. Options vary depending on the version of Blender,
                with the following being broadly available: ('NONE', 'PXR24', 'ZIP', 'PIZ', 'RLE', 'ZIPS', 'DWAA', 'DWAB').
                Defaults to "ZIP".

        Note:
            The debug colormap is re-normalized on a per-frame basis, to visually
            compare across frames, apply colorization after rendering using the CLI.

        Raises:
            ValueError: raised when `direction` is not understood.
            RuntimeError: raised when motion blur is enabled as flow cannot be computed.
        """

    @require_initialized_service
    def include_segmentations(
        self, shuffle: bool = True, debug: bool = True, seed: int = 1234, exr_codec: str = "ZIP"
    ) -> None:
        """Sets up Blender compositor to include segmentation maps for rendered images.

        The debug visualization simply assigns a color to each object ID by mapping the
        objects ID value to a hue using a HSV node with saturation=1 and value=1 (except
        for the background which will have a value of 0 to ensure it is black).

        Args:
            shuffle (bool, optional): shuffle debug colors, helps differentiate object instances. Defaults to True.
            debug (bool, optional): If true, also save debug visualizations of segmentation. Defaults to True.
            seed (int, optional): random seed used when shuffling colors. Defaults to 1234.
            exr_codec (str, optional): codec used to compress exr file. Options vary depending on the version of Blender,
                with the following being broadly available: ('NONE', 'PXR24', 'ZIP', 'PIZ', 'RLE', 'ZIPS', 'DWAA', 'DWAB').
                Defaults to "ZIP".

        Raises:
            RuntimeError: raised when not using CYCLES, as other renderers do not support a segmentation pass.
        """

    @require_initialized_service
    def load_addons(self, *addons: str) -> None:
        """Load blender addons by name (case-insensitive)"""

    @require_initialized_service
    def set_resolution(self, height: tuple[int] | list[int] | int | None = None, width: int | None = None) -> None:
        """Set frame resolution (height, width) in pixels.

        If a single tuple is passed, instead of using keyword arguments, it will be parsed as (height, width).
        """

    @require_initialized_service
    def image_settings(
        self, file_format: str | None = None, bit_depth: int | None = None, color_mode: str | None = None
    ) -> None:
        """Set the render's output format and bit-depth.
        Useful for linear intensity renders, using "OPEN_EXR" and 32 or 16 bits.
        """

    @require_initialized_service
    def use_motion_blur(self, enable: bool) -> None:
        """Enable/disable motion blur."""

    @require_initialized_service
    def use_animations(self, enable: bool) -> None:
        """Enable/disable all animations."""

    @require_initialized_service
    def cycles_settings(
        self,
        device_type: str | None = None,
        use_cpu: bool | None = None,
        adaptive_threshold: float | None = None,
        max_samples: int | None = None,
        use_denoising: bool | None = None,
    ) -> tuple[list[str],]:
        """Enables/activates cycles render devices and settings.

        Note: A default arguments of `None` means do not change setting inherited from blendfile.

        Args:
            device_type (str, optional): Name of device to use, one of "cpu", "cuda", "optix", "metal", etc.
                See `blender docs <https://docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html>`_
                for full list. Defaults to None.
            use_cpu (bool, optional): Boolean flag to enable CPUs alongside GPU devices. Defaults to None.
            adaptive_threshold (float, optional): Set noise threshold upon which to stop taking samples. Defaults to None.
            max_samples (int, optional): Maximum number of samples per pixel to take. Defaults to None.
            use_denoising (bool, optional): If enabled, a denoising pass will be used. Defaults to None.

        Raises:
            RuntimeError: raised when no devices are found.
            ValueError: raised when setting `use_cpu` is required.

        Returns:
            devices: List of activated devices.
        """

    @require_initialized_service
    def unbind_camera(self) -> None:
        """Remove constraints, animations and parents from main camera.

        Note: In order to undo this, you'll need to re-initialize.
        """

    @require_initialized_service
    def move_keyframes(self, scale: float = 1.0, shift: float = 0.0) -> None:
        """Adjusts keyframes in Blender animations, keypoints are first scaled then shifted.

        Args:
            scale (float, optional): Factor used to rescale keyframe positions along x-axis. Defaults to 1.0.
            shift (float, optional): Factor used to shift keyframe positions along x-axis. Defaults to 0.0.

        Raises:
            RuntimeError: raised if trying to move keyframes beyond blender's limits.
        """

    @require_initialized_service
    def set_current_frame(self, frame_number: int) -> None:
        """Set current frame number. This might advance any animations."""

    @require_initialized_service
    def camera_extrinsics(self) -> tuple[npt.NDArray[np.floating],]:
        """Get the 4x4 transform matrix encoding the current camera pose."""

    @require_initialized_service
    def camera_intrinsics(self) -> tuple[npt.NDArray[np.floating],]:
        """Get the 3x3 camera intrinsics matrix for active camera,
        which defines how 3D points are projected onto 2D.

        Note: Assumes pinhole camera model.

        Returns:
            Camera intrinsics matrix based on camera properties.
        """

    @require_initialized_service
    @validate_camera_moved
    def position_camera(
        self,
        location: npt.ArrayLike | None = None,
        rotation: npt.ArrayLike | None = None,
        look_at: npt.ArrayLike | None = None,
        in_order: bool = True,
    ) -> None:
        """Positions and orients camera in Blender scene according to specified parameters.

        Note: Only one of `look_at` or `rotation` can be set at once.

        Args:
            location (npt.ArrayLike, optional): Location to place camera in 3D space. Defaults to none.
            rotation (npt.ArrayLike, optional): Rotation matrix for camera. Defaults to none.
            look_at (npt.ArrayLike, optional): Location to point camera. Defaults to none.
            in_order (bool, optional): If set, assume current camera pose is from previous/next
                frame and ensure new rotation set by `look_at` is compatible with current position.
                Without this, a rotations will stay in the [-pi, pi] range and this wrapping will
                mess up interpolations. Only used when `look_at` is set. Defaults to True.

        Raises:
            ValueError: raised if camera orientation is over-defined.
        """

    @require_initialized_service
    @validate_camera_moved
    def rotate_camera(self, angle: float) -> None:
        """Rotate camera around it's optical axis, relative to current orientation.

        Args:
            angle: Relative amount to rotate by (clockwise, in radians).
        """

    @require_initialized_service
    def set_camera_keyframe(self, frame_num: int, matrix: npt.ArrayLike | None = None) -> None:
        """Set camera keyframe at given frame number.
        If camera matrix is not supplied, currently set camera position/rotation/scale will be used,
        this allows users to set camera position using `position_camera` and `rotate_camera`.

        Args:
            frame_num (int): index of frame to set keyframe for.
            matrix (npt.ArrayLike | None, optional): 4x4 camera transform, if not supplied,
                use current camera matrix. Defaults to None.
        """

    @require_initialized_service
    def set_animation_range(self, start: int | None = None, stop: int | None = None, step: int | None = None) -> None:
        """Set animation range for scene.

        Args:
            start (int | None, optional): frame start, inclusive. Defaults to None.
            stop (int | None, optional): frame stop, exclusive. Defaults to None.
            step (int | None, optional): frame interval. Defaults to None.
        """

    @require_initialized_service
    def render_current_frame(self, allow_skips: bool = True, dry_run: bool = False) -> tuple[dict[str, Any],]:
        """Generates a single frame in Blender at the current camera location,
        return the file paths for that frame, potentially including depth, normals, etc.

        Args:
            allow_skips (bool, optional): if true, blender will not re-render and overwrite existing frames.
                This does not however apply to depth/normals/etc, which cannot be skipped. Defaults to True.
            dry_run (bool, optional): if true, nothing will be rendered at all. Defaults to False.

        Returns:
            dictionary containing paths to rendered frames for this index and camera pose.
        """

    @require_initialized_service
    def render_frame(self, frame_number: int, allow_skips: bool = True, dry_run: bool = False) -> tuple[dict[str, Any],]:
        """Same as first setting current frame then rendering it.

        Warning:
            Calling this has the side-effect of changing the current frame.
        """
