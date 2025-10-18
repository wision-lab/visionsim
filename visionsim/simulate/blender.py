from __future__ import annotations

import argparse
import functools
import inspect
import itertools
import logging
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager, nullcontext
from multiprocessing import Process
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import rpyc  # type: ignore
import rpyc.utils.registry  # type: ignore

if TYPE_CHECKING:
    # Import only when type checking as to not introduce
    # dependency for blender. Block module typechecking.
    from collections.abc import Callable, Collection, Iterable, Iterator
    from types import TracebackType

    import multiprocess  # type: ignore
    import multiprocess.pool  # type: ignore
    import numpy.typing as npt
    from typing_extensions import Any, Concatenate, ParamSpec, Self, TypeVar

    from visionsim.types import UpdateFn

    T = TypeVar("T")
    P = ParamSpec("P")

try:
    # These are blender specific modules which aren't easily installed but
    # are loaded in when this script is ran from blender.
    import addon_utils  # type: ignore
    import bpy  # type: ignore
    import mathutils  # type: ignore

    # Allow relative imports to this file without forcing the user to
    # install visionsim into their blender install
    sys.path.insert(0, str(Path(__file__).parent.resolve()))
    from nodes import (  # type: ignore
        flowdebug_node_group,
        normaldebug_node_group,
        segmentationdebug_node_group,
        vec2rgba_node_group,
    )
except ImportError:
    addon_utils = None
    bpy = None
    mathutils = None


# Enable server-side logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
)
server_log: logging.Logger = logging.getLogger(__name__)
server_log.setLevel(logging.INFO)

EXPOSED_PREFIX: str = "exposed_"
REGISTRY: tuple[Process, rpyc.utils.registry.UDPRegistryClient] | None = None


def require_connected_client(
    func: Callable[Concatenate[BlenderClient, P], T],
) -> Callable[Concatenate[BlenderClient, P], T]:
    """Decorator which ensures a client is connected.

    Args:
        func (Callable[Concatenate[BlenderClient, P], T]): Function to decorate

    Raises:
        RuntimeError: raised if client is not connected.

    Returns:
        Callable[Concatenate[BlenderClient, P], T]: Decorated function.
    """

    @functools.wraps(func)
    def _decorator(self: BlenderClient, *args: P.args, **kwargs: P.kwargs) -> T:
        if self.conn is None:
            raise RuntimeError(
                f"'BlenderClient' must be connected to a server instance before calling '{func.__name__}'"
            )
        return func(self, *args, **kwargs)

    return _decorator


def require_connected_clients(
    func: Callable[Concatenate[BlenderClients, P], T],
) -> Callable[Concatenate[BlenderClients, P], T]:
    """Decorator which ensures all clients are connected.

    Args:
        func (Callable[Concatenate[BlenderClients, P], T]): Function to decorate

    Raises:
        RuntimeError: raised if at least one client is not connected.

    Returns:
        Callable[Concatenate[BlenderClients, P], T]: Decorated function.
    """

    @functools.wraps(func)
    def _decorator(self: BlenderClients, *args: P.args, **kwargs: P.kwargs):
        if any(c.conn is None for c in self):
            raise RuntimeError(
                f"All client instances in 'BlenderClients' must be connected before calling '{func.__name__}'"
            )
        return func(self, *args, **kwargs)

    return _decorator


def require_initialized_service(
    func: Callable[Concatenate[BlenderService, P], T],
) -> Callable[Concatenate[BlenderService, P], T]:
    """Decorator which ensures the render service was initialized.

    Args:
        func (Callable[Concatenate[BlenderService, P], T]): Function to decorate

    Raises:
        RuntimeError: raised if :meth:`client.initialize <BlenderService.exposed_initialize>` has not been previously called.

    Returns:
        Callable[Concatenate[BlenderService, P], T]: Decorated function.
    """

    @functools.wraps(func)
    def _decorator(self: BlenderService, *args: P.args, **kwargs: P.kwargs):
        if not self.initialized:
            raise RuntimeError(f"'BlenderService' must be initialized before calling '{func.__name__}'")
        return func(self, *args, **kwargs)

    return _decorator


def validate_camera_moved(
    func: Callable[Concatenate[BlenderService, P], T],
) -> Callable[Concatenate[BlenderService, P], T]:
    """Decorator which emits a warning if the camera was not moved.

    Args:
        func (Callable[Concatenate[BlenderService, P], T]): Function to decorate

    Returns:
        Callable[Concatenate[BlenderService, P], T]: Decorated function.
    """

    @functools.wraps(func)
    def _decorator(self: BlenderService, *args: P.args, **kwargs: P.kwargs):
        prev_matrix = np.array(self.camera.matrix_world.copy())
        retval = func(self, *args, **kwargs)
        post_matrix = np.array(self.camera.matrix_world.copy())

        if np.allclose(prev_matrix, post_matrix):
            self.log.warning("Camera has not moved as intended, perhaps it is still bound by parent or animation?")
        return retval

    return _decorator


class BlenderServer(rpyc.utils.server.Server):
    """Expose a :class:`BlenderService` to the outside world via RPCs.

    Example:
        Once created, it can be started, which will block and await for an external connection from a :class:`BlenderClient`:

        .. code-block:: python

            server = BlenderServer()
            server.start()

        However, this needs to be called within blender's runtime. Instead one can use :meth:`BlenderServer.spawn`
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
        """Initialize a :class:`BlenderServer` instance

        Args:
            hostname (bytes | str | None, optional): the host to bind to. By default, the 'wildcard address' is used
                to listen on all interfaces. If not properly secured, the server can receive traffic from
                unintended or even malicious sources. Defaults to None (wildcard).
            port (bytes | str | int | None, optional): the TCP port to bind to. Defaults to 0 (bind to a random open port).
            service (type[BlenderService], optional): the service to expose, must be a :class:`BlenderService` subclass. Defaults to :class:`BlenderService`.
            extra_config (dict, optional): the configuration dictionary that is passed to the RPyC connection.
                Defaults to ``{"allow_all_attrs": True, "allow_setattr": True}``.
            **kwargs: Additional keyword arguments which are passed to the
                `rpyc.utils.server.Server <https://rpyc.readthedocs.io/en/latest/api/utils_server.html#rpyc.utils.server.Server>`_ constructor.

        Raises:
            RuntimeError: a :class:`BlenderServer` needs to be instantiated from within a blender instance.
            ValueError: the exposed service must be :class:`BlenderService` or subclass.
        """
        if bpy is None:
            raise RuntimeError(f"{type(self).__name__} needs to be instantiated from within blender's python runtime.")
        if service and not issubclass(service, BlenderService):
            raise ValueError("Parameter 'service' must be 'BlenderService' or subclass.")

        super().__init__(
            service or BlenderService,
            hostname=hostname,
            port=port,
            protocol_config=rpyc.core.protocol.DEFAULT_CONFIG | (extra_config or {"allow_all_attrs": True}),
            auto_register=True,
            **kwargs,
        )
        server_log.info(f"Started listening on {self.host}:{self.port}")

    @staticmethod
    @contextmanager
    def spawn(
        jobs: int = 1,
        timeout: float = -1.0,
        log_dir: str | os.PathLike | None = None,
        autoexec: bool = False,
        executable: str | os.PathLike | None = None,
    ) -> Iterator[tuple[list[subprocess.Popen], list[tuple[str, int]]]]:
        """Spawn one or more blender instances and start a :class:`BlenderServer` in each.

        This is roughly equivalent to calling ``blender -b --python blender.py`` in many subprocesses,
        where ``blender.py`` initializes and ``start``s a server instance. Proper logging and termination of
        these processes is also taken care of.

        Note: The returned processes and connection settings are not guaranteed to be in the same order.

        Args:
            jobs (int, optional): number of jobs to spawn. Defaults to 1.
            timeout (float, optional): try to discover spawned instances for ``timeout``
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
            TimeoutError: raise if unable to discover spawned servers in ``timeout`` seconds and kill any spawned processes.

        Returns:
            tuple[list[subprocess.Popen], list[tuple[str, int]]]:  A tuple containing:
                - list[subprocess.Popen]: List of ``subprocess.Popen`` corresponding to all spawned servers.
                - list[tuple[str, int]]: List of connection setting for each server, where each element is a (hostname, port) tuple.
        """
        # Note import here as this is a dependency only on the client-side
        from ephemeral_port_reserve import reserve as port_reserve  # type: ignore

        @contextmanager
        def terminate_jobs(procs):
            try:
                yield

            finally:
                for p in procs:
                    # We need to send two CTRL+C events to blender to kill it
                    p.send_signal(signal.SIGINT)
                    p.send_signal(signal.SIGINT)

                for p in procs:
                    # Ensure process is killed if CTRL+C failed
                    p.terminate()

        BlenderServer.spawn_registry()
        existing = BlenderServer.discover()
        procs, ports = [], []

        if log_dir:
            log_dir_path = Path(log_dir).expanduser().resolve()
            log_dir_path.mkdir(parents=True, exist_ok=True)
        else:
            log_dir_path = None

        with ExitStack() as stack:
            for i in range(jobs):
                port = port_reserve("localhost") if timeout < 0 else 0
                autoexec_cmd = "--enable-autoexec" if autoexec else "--disable-autoexec"
                cmd = shlex.split(
                    f"{executable or 'blender'} -b {autoexec_cmd} --python-use-system-env --python {__file__} -- --port {port}"
                )

                if log_dir_path:
                    (log_dir_path / f"job{i:03}").mkdir(parents=True, exist_ok=True)
                    stdout = stack.enter_context(open(log_dir_path / f"job{i:03}" / "stdout.log", "w"))
                    stderr = stack.enter_context(open(log_dir_path / f"job{i:03}" / "stderr.log", "w"))
                    proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, universal_newlines=True)
                else:
                    proc = subprocess.Popen(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, universal_newlines=True
                    )
                procs.append(proc)
                ports.append(port)
            stack.enter_context(terminate_jobs(procs))

            if timeout > 0:
                start = time.time()

                while True:
                    if (time.time() - start) > timeout:
                        # Terminate all procs and close fds
                        stack.close()
                        raise TimeoutError("Unable to spawn and discover server(s) in alloted time.")
                    if len(conns := set(BlenderServer.discover()) - set(existing)) == jobs:
                        break
                    time.sleep(0.1)
            else:
                conns = set(("localhost", p) for p in ports)

            yield (procs, list(conns))

    @staticmethod
    def spawn_registry() -> tuple[Process, rpyc.utils.registry.UDPRegistryClient]:
        """Spawn a registry server and client to aid in server discovery, or return cached result.
        While this method can be called directly, it will be invoked automatically by :meth:`discover` and :meth:`spawn`.

        Returns:
            tuple[Process, rpyc.utils.registry.UDPRegistryClient]: A tuple containing:
                - Process: process running the global registry server,
                - rpyc.utils.registry.UDPRegistryClient: global registry client
        """
        global REGISTRY

        if not REGISTRY or not REGISTRY[0].is_alive():
            registry = Process(target=BlenderServer._launch_registry, daemon=True)
            client = rpyc.utils.registry.UDPRegistryClient()
            registry.start()
            REGISTRY = (registry, client)
        return REGISTRY

    @staticmethod
    def _launch_registry():
        try:
            registry = rpyc.utils.registry.UDPRegistryServer()
            registry.start()
        except OSError:
            # Note: Address is likely already in use, meaning there's
            #   already a spawned registry in another thread/process which
            #   we should be able to use. No need to re-spawn one then.
            pass

    @staticmethod
    def discover() -> list[tuple[str, int]]:
        """Discover any :class:`BlenderServer`s that are already running and return their connection parameters.

        Note:
            A discoverable server might already be in use and can refuse connection attempts.

        Returns:
            list[tuple[str, int]]: List of connection setting for each server, where each element is a (hostname, port) tuple.
        """
        _, client = BlenderServer.spawn_registry()
        return list(client.discover("BLENDER"))

    def _accept_method(self, sock: socket.socket) -> None:
        # Accept a single connection, and block here until it closes. Any other incoming
        # connections will stall, and run out the `sync_request_timeout` while attempting to connect.
        self._authenticate_and_serve_client(sock)


class BlenderService(rpyc.Service):
    """Server-side API to interact with blender and render novel views.

    Most of the methods of a :class:`BlenderClient` instance are remote procedure calls to
    a connected blender service. These methods are prefixed by ``exposed_``.
    """

    # Note: This alias is used when discovering servers using the registry.
    #   By default the service name is extracted from the class name, so here
    #   it would be `blender` anyways, but we define an alias here to support
    #   subclasses which might be named differently and not discovered.
    ALIASES: list[str] = ["BLENDER"]

    def __init__(self) -> None:
        """Initialize render service.

        Raises:
            RuntimeError: raised if not within blender's runtime.
        """
        if bpy is None:
            raise RuntimeError(f"{type(self).__name__} needs to be instantiated from within blender's python runtime.")
        self._conn: rpyc.Connection | None = None
        self.log: logging.Logger = server_log
        self.initialized = False

    def _clear_cached_properties(self) -> None:
        # Based on: https://stackoverflow.com/a/71579485
        for name in dir(type(self)):
            if isinstance(getattr(type(self), name), functools.cached_property):
                vars(self).pop(name, None)  # type: ignore

    def on_connect(self, conn: rpyc.Connection) -> None:
        """Called when the connection is established

        Args:
            conn (rpyc.Connection): Connection object
        """
        # Log to server as logging is likely not setup yet.
        server_log.info("Successfully connected to BlenderClient instance.")
        self._conn = conn

    def on_disconnect(self, conn: rpyc.Connection) -> None:
        """Called when the connection has already terminated. Resets blender runtime.
        (must not perform any IO on the connection)

        Args:
            conn (rpyc.Connection): Connection object
        """
        server_log.info("Successfully disconnected from BlenderClient instance.")
        self._conn = None
        self.reset()

    def reset(self) -> None:
        """Cleans up and resets blender runtime.

        De-initialize service by restoring blender to it's startup state,
        ensuring any cached attributes are cleaned (otherwise objects will be stale),
        and resetting any instance variables that were previously initialized.
        """
        bpy.ops.wm.read_factory_settings()
        self._clear_cached_properties()
        self.initialized = False

    @property
    @require_initialized_service
    def scene(self) -> bpy.types.Scene:
        """Get current blender scene"""
        return bpy.context.scene

    @property
    @require_initialized_service
    def tree(self) -> bpy.types.CompositorNodeTree:
        """Get current scene's node tree"""
        return self.scene.node_tree

    @functools.cached_property
    @require_initialized_service
    def render_layers(self) -> bpy.types.CompositorNodeRLayers:
        """Get and cache render layers node, create one if needed."""
        for node in self.tree.nodes:
            if node.type == "R_LAYERS":
                return node
        return self.tree.nodes.new("CompositorNodeRLayers")

    @property
    @require_initialized_service
    def view_layer(self) -> bpy.types.ViewLayer:
        """Get current view layer"""
        if not bpy.context.view_layer:
            raise ValueError("Expected at least one view layer, cannot render without it. Please add one manually.")
        return bpy.context.view_layer

    @functools.cached_property
    @require_initialized_service
    def camera(self) -> bpy.types.Camera:
        """Get and cache active camera"""
        # Make sure there's a camera
        cameras = [ob for ob in self.scene.objects if ob.type == "CAMERA"]
        if not cameras:
            raise RuntimeError("No camera found, please add one manually.")
        elif len(cameras) > 1 and self.scene.camera:
            self.log.warning(f"Multiple cameras found. Using active camera named: '{self.scene.camera.name}'.")
            return self.scene.camera
        else:
            self.log.warning(f"No active camera was found. Using camera named: '{cameras[0].name}'.")
            return cameras[0]

    @require_initialized_service
    def get_parents(self, obj: bpy.types.Object) -> list[bpy.types.Object]:
        """Recursively retrieves parent objects of a given object in Blender

        Args:
            obj: Object to find parent of.

        Returns:
            list[bpy.types.Object]: Parent objects of obj.
        """
        if getattr(obj, "parent", None):
            return [obj.parent] + self.get_parents(obj.parent)
        return []

    def exposed_with_logger(self, log: logging.Logger) -> None:
        """Use supplied logger, if logger is initialized in client, messages will log to the client.

        Args:
            log (logging.Logger): Logger to use for messages
        """
        self.log = log

    def exposed_initialize(self, blend_file: str | os.PathLike, root_path: str | os.PathLike, **kwargs) -> None:
        """Initialize BlenderService and load blendfile.

        Args:
            blend_file (str | os.PathLike): path of scene file to load.
            root_path (str | os.PathLike): path at which to save rendered results.
            **kwargs: Additional keyword arguments to be passed to
                `bpy.ops.wm.open_mainfile <https://docs.blender.org/api/current/bpy.ops.wm.html#bpy.ops.wm.open_mainfile>`_.
        """
        # TODO: This should perhaps be `exposed_load_file`, and the root_path logic should be moved
        #   to another method which would facilitate writing to local disk/sending renders over the wire.
        if self.initialized:
            self.reset()

        # Load blendfile
        self.blend_file: Path = Path(str(blend_file))
        bpy.ops.wm.open_mainfile(filepath=str(blend_file), **kwargs)
        self.log.info(f"Successfully loaded {blend_file}")

        # Ensure root paths exist
        self.root_path: Path = Path(str(root_path)).resolve()
        self.root_path.mkdir(parents=True, exist_ok=True)
        (self.root_path / "frames").mkdir(parents=True, exist_ok=True)

        # Init various variables to track state
        self.depth_path: bpy.types.CompositorNodeOutputFile | None = None
        self.normal_path: bpy.types.CompositorNodeOutputFile | None = None
        self.flow_path: bpy.types.CompositorNodeOutputFile | None = None
        self.segmentation_path: bpy.types.CompositorNodeOutputFile | None = None
        self.depth_extension = ".exr"
        self.unbind_camera: bool = False
        self.use_animation: bool = True
        self.initialized = True

        # Ensure we are using the compositor, and node tree.
        self.scene.use_nodes = True
        self.scene.render.use_compositing = True

        # Set default render settings
        self.scene.render.use_persistent_data = True
        self.scene.render.film_transparent = False

        # Warn if extra file output pipelines are found
        for n in getattr(self.tree, "nodes", []):
            if isinstance(n, bpy.types.CompositorNodeOutputFile) and not n.mute:
                self.log.warning(f"Found unexpected output node {n}")

        # Catalogue any animations that are already disabled, otherwise
        # disabling and re-enabling animations would enable them.
        self.disabled_fcurves: set[bpy.types.Action] = set(
            [fcurve for action in bpy.data.actions for fcurve in (action.fcurves or []) if fcurve.mute]
        )

    @require_initialized_service
    def exposed_empty_transforms(self) -> dict[str, Any]:
        """Return a dictionary with camera intrinsics. Forms the basis of
        a ``transforms.json`` file, but contains no frame data.

        Returns:
            dict[str, Any]: empty transforms dictionary containing only camera parameters.
        """
        transforms = {
            k: getattr(self.camera.data, k)
            for k in [
                "angle",
                "angle_x",
                "angle_y",
                "clip_start",
                "clip_end",
                "lens",
                "lens_unit",
                "sensor_height",
                "sensor_width",
                "sensor_fit",
                "shift_x",
                "shift_y",
                "type",
            ]
        }
        transforms["frames"] = []

        # Note: This might be a blender bug, but when height==width,
        #   angle_x != angle_y, so here we just use angle.
        transforms["c"] = {"BW": 1, "RGB": 3, "RGBA": 4}.get(self.scene.render.image_settings.color_mode)
        transforms["w"] = self.scene.render.resolution_x
        transforms["h"] = self.scene.render.resolution_y
        transforms["fl_x"] = float(1 / 2 * self.scene.render.resolution_x / np.tan(1 / 2 * self.camera.data.angle))
        transforms["fl_y"] = float(1 / 2 * self.scene.render.resolution_y / np.tan(1 / 2 * self.camera.data.angle))
        transforms["cx"] = 1 / 2 * self.scene.render.resolution_x + transforms["shift_x"]
        transforms["cy"] = 1 / 2 * self.scene.render.resolution_y + transforms["shift_y"]
        transforms["intrinsics"] = self.exposed_camera_intrinsics().tolist()
        return transforms

    @require_initialized_service
    def exposed_original_fps(self) -> int:
        """Get effective framerate (fps/fps_base).

        Returns:
            int: Frame rate of scene.
        """
        # Note: Exposed properties are not supported by rpyc.
        return int(round(self.scene.render.fps / self.scene.render.fps_base))

    @require_initialized_service
    def exposed_animation_range(self) -> range:
        """Get animation range of current scene as range(start, end+1, step).

        Returns:
            range: Range of frames in animation.
        """
        return range(self.scene.frame_start, self.scene.frame_end + 1, self.scene.frame_step)

    @require_initialized_service
    def exposed_animation_range_tuple(self) -> tuple[int, int, int]:
        """Get animation range of current scene as a tuple of (start, end, step).

        Returns:
            tuple[int, int, int]: Frame start, end, and step of animation.
        """
        return (self.scene.frame_start, self.scene.frame_end, self.scene.frame_step)  # type: ignore

    @require_initialized_service
    def exposed_include_depths(self, debug=True, file_format="OPEN_EXR", exr_codec="ZIP") -> None:
        """Sets up Blender compositor to include depth map for rendered images.

        Args:
            debug (bool, optional): if true, colorized depth maps, helpful for quick visualizations,
                will be generated alongside ground-truth depth maps. Defaults to True.
            file_format (str, optional): format of depth maps, one of "OPEN_EXR" or "HDR". The former
                is lossless, but can require significant storage, the later is lossy and more compressed.
                If depth is needed to compute scene-flow, use open-exr. Defaults to "OPEN_EXR".
            exr_codec (str, optional): codec used to compress exr file. Only used when ``file_format="OPEN_EXR"``,
                options vary depending on the version of Blender, with the following being broadly available:
                ('NONE', 'PXR24', 'ZIP', 'PIZ', 'RLE', 'ZIPS', 'DWAA', 'DWAB'). Defaults to "ZIP".

        Note:
            The debug colormap is re-normalized on a per-frame basis, to visually
            compare across frames, apply colorization after rendering using the CLI.

        Raises:
            ValueError: raise if file-format nor understood.
        """
        # TODO: Add colormap option?
        self.view_layer.use_pass_z = True
        (self.root_path / "depths").mkdir(parents=True, exist_ok=True)

        if file_format.upper() not in ("OPEN_EXR", "HDR"):
            raise ValueError(f"Expected one of OPEN_EXR/HDR for file_format, got {file_format}.")

        if debug:
            debug_depth_path = self.tree.nodes.new(type="CompositorNodeOutputFile")
            debug_depth_path.base_path = str(self.root_path / "depths")
            debug_depth_path.label = "Debug Depth Output"
            debug_depth_path.file_slots[0].path = f"debug_depth_{'#' * 6}"
            debug_depth_path.format.file_format = "PNG"
            debug_depth_path.format.compression = 90
            debug_depth_path.format.color_depth = "8"
            debug_depth_path.format.color_mode = "BW"

            # Important! Set the view settings to raw otherwise result is tonemapped
            if bpy.app.version >= (3, 2, 0):
                debug_depth_path.format.color_management = "OVERRIDE"
            debug_depth_path.format.view_settings.view_transform = "Raw"
            debug_depth_path.format.view_settings.look = "None"
            debug_depth_path.format.view_settings.gamma = 0
            debug_depth_path.format.view_settings.exposure = 1
            debug_depth_path.format.view_settings.use_curve_mapping = False

            if bpy.app.version >= (4, 3, 0):
                debug_depth_path.format.view_settings.use_white_balance = False

            normalize = self.tree.nodes.new("CompositorNodeNormalize")
            self.tree.links.new(self.render_layers.outputs["Depth"], normalize.inputs[0])
            self.tree.links.new(normalize.outputs[0], debug_depth_path.inputs[0])

        self.depth_path = self.tree.nodes.new(type="CompositorNodeOutputFile")
        self.depth_path.label = "Depth Output"
        self.depth_path.file_slots[0].path = f"depth_{'#' * 6}"

        if bpy.app.version >= (3, 2, 0):
            self.depth_path.format.color_management = "OVERRIDE"
            self.depth_path.format.linear_colorspace_settings.name = "Non-Color"
        self.depth_path.format.file_format = file_format

        if file_format.upper() == "OPEN_EXR":
            self.depth_path.format.exr_codec = exr_codec

            if bpy.app.version < (4, 3, 0):
                self.depth_path.format.color_mode = "RGB"
            else:
                self.depth_path.format.color_mode = "BW"

            self.depth_extension = ".exr"
        else:
            self.depth_path.format.color_mode = "RGB"
            self.depth_extension = ".hdr"

        self.tree.links.new(self.render_layers.outputs["Depth"], self.depth_path.inputs[0])
        self.depth_path.base_path = str(self.root_path / "depths")

    @require_initialized_service
    def exposed_include_normals(self, debug=True, exr_codec="ZIP") -> None:
        """Sets up Blender compositor to include normal map for rendered images.

        Args:
            debug (bool, optional): if true, colorized normal maps will also be generated with each vector
                component being remapped from [-1, 1] to [0-255] with xyz becoming rgb. Defaults to True.
            exr_codec (str, optional): codec used to compress exr file. Options vary depending on the version of Blender,
                with the following being broadly available: ('NONE', 'PXR24', 'ZIP', 'PIZ', 'RLE', 'ZIPS', 'DWAA', 'DWAB').
                Defaults to "ZIP".
        """
        self.view_layer.use_pass_normal = True
        (self.root_path / "normals").mkdir(parents=True, exist_ok=True)

        # The node group `normaldebug` transforms normals from the global
        # coordinate frame to the camera's, and also colors normals as RGB
        normal_group = self.tree.nodes.new("CompositorNodeGroup")
        normal_group.label = "NormalDebug"
        normal_group.node_tree = normaldebug_node_group()
        self.tree.links.new(self.render_layers.outputs["Normal"], normal_group.inputs[0])

        if debug:
            debug_normal_path = self.tree.nodes.new(type="CompositorNodeOutputFile")
            debug_normal_path.base_path = str(self.root_path / "normals")
            debug_normal_path.label = "Normals Debug Output"
            debug_normal_path.file_slots[0].path = f"debug_normal_{'#' * 6}"
            debug_normal_path.format.file_format = "PNG"
            debug_normal_path.format.compression = 90
            debug_normal_path.format.color_depth = "8"
            debug_normal_path.format.color_mode = "RGB"

            # Important! Set the view settings to raw otherwise result is tonemapped
            if bpy.app.version >= (3, 2, 0):
                debug_normal_path.format.color_management = "OVERRIDE"
            debug_normal_path.format.view_settings.view_transform = "Raw"
            debug_normal_path.format.view_settings.look = "None"
            debug_normal_path.format.view_settings.gamma = 0
            debug_normal_path.format.view_settings.exposure = 1
            debug_normal_path.format.view_settings.use_curve_mapping = False

            if bpy.app.version >= (4, 3, 0):
                debug_normal_path.format.view_settings.use_white_balance = False

            self.tree.links.new(normal_group.outputs["RGBA"], debug_normal_path.inputs[0])

        self.normal_path = self.tree.nodes.new(type="CompositorNodeOutputFile")
        self.normal_path.label = "Normal Output"

        if bpy.app.version >= (3, 2, 0):
            self.normal_path.format.color_management = "OVERRIDE"
            self.normal_path.format.linear_colorspace_settings.name = "Non-Color"
        self.normal_path.format.file_format = "OPEN_EXR"
        self.normal_path.format.exr_codec = exr_codec
        self.normal_path.format.color_mode = "RGB"

        vec2rgba = self.tree.nodes.new("CompositorNodeGroup")
        vec2rgba.label = "Vector2RGBA"
        vec2rgba.node_tree = vec2rgba_node_group()

        self.tree.links.new(normal_group.outputs["Vector"], vec2rgba.inputs["Image"])
        self.tree.links.new(vec2rgba.outputs["Image"], self.normal_path.inputs[0])

        self.normal_path.base_path = str(self.root_path / "normals")
        self.normal_path.file_slots[0].path = f"normal_{'#' * 6}"

    @require_initialized_service
    def exposed_include_flows(self, direction="forward", debug=True, exr_codec="ZIP") -> None:
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
            ValueError: raised when ``direction`` is not understood.
            RuntimeError: raised when motion blur is enabled as flow cannot be computed.
        """
        if direction.lower() not in ("forward", "backward", "both"):
            raise ValueError(f"Direction argument should be one of forward, backward or both, got {direction}.")
        if self.scene.render.use_motion_blur:
            raise RuntimeError("Cannot compute optical flow if motion blur is enabled.")

        self.view_layer.use_pass_vector = True
        (self.root_path / "flows").mkdir(parents=True, exist_ok=True)

        if debug:
            # Separate forward and backward flows (with a separate color not vector node)
            if bpy.app.version >= (3, 3, 0):
                split_flow = self.tree.nodes.new(type="CompositorNodeSeparateColor")
                split_flow.mode = "RGB"
            else:
                split_flow = self.tree.nodes.new(type="CompositorNodeSepRGBA")
            self.tree.links.new(self.render_layers.outputs["Vector"], split_flow.inputs["Image"])

            # Create output node
            debug_flow_path = self.tree.nodes.new(type="CompositorNodeOutputFile")
            debug_flow_path.base_path = str(self.root_path / "flows")
            debug_flow_path.label = "Flow Debug Output"
            debug_flow_path.file_slots.clear()
            debug_flow_path.format.file_format = "PNG"
            debug_flow_path.format.compression = 90
            debug_flow_path.format.color_depth = "8"
            debug_flow_path.format.color_mode = "RGB"

            # Important! Set the view settings to raw otherwise result is tonemapped
            if bpy.app.version >= (3, 2, 0):
                debug_flow_path.format.color_management = "OVERRIDE"
            debug_flow_path.format.view_settings.view_transform = "Raw"
            debug_flow_path.format.view_settings.look = "None"
            debug_flow_path.format.view_settings.gamma = 0
            debug_flow_path.format.view_settings.exposure = 1
            debug_flow_path.format.view_settings.use_curve_mapping = False

            if bpy.app.version >= (4, 3, 0):
                debug_flow_path.format.view_settings.use_white_balance = False

            # Instantiate flow debug node group(s) and connect them
            if direction.lower() in ("forward", "both"):
                flow_group = self.tree.nodes.new("CompositorNodeGroup")
                flow_group.label = "Forward FlowDebug"
                flow_group.node_tree = flowdebug_node_group()

                self.tree.links.new(split_flow.outputs[0], flow_group.inputs["x"])
                self.tree.links.new(split_flow.outputs[1], flow_group.inputs["y"])

                slot = debug_flow_path.file_slots.new(f"debug_fwd_flow_{'#' * 6}")
                self.tree.links.new(flow_group.outputs["Image"], slot)
            if direction.lower() in ("backward", "both"):
                flow_group = self.tree.nodes.new("CompositorNodeGroup")
                flow_group.label = "Backward FlowDebug"
                flow_group.node_tree = flowdebug_node_group()

                self.tree.links.new(split_flow.outputs[2], flow_group.inputs["x"])
                self.tree.links.new(split_flow.outputs[3], flow_group.inputs["y"])

                slot = debug_flow_path.file_slots.new(f"debug_bwd_flow_{'#' * 6}")
                self.tree.links.new(flow_group.outputs["Image"], slot)

        # Save flows as EXRs, flows are a 4-vec of forward flows x/y then backwards flows x/y
        # before blender 4.3, saving a vector as an image saved only 3 channels even if `color_mode`
        # is set to RGBA. So we add a dummy vec2rgba node to trick blender into treating the
        # vector as an image with 4 channels. This dummy node just splits and recombines channels.
        self.flow_path = self.tree.nodes.new(type="CompositorNodeOutputFile")
        self.flow_path.label = "Flow Debug Output"

        if bpy.app.version >= (3, 2, 0):
            self.flow_path.format.color_management = "OVERRIDE"
            self.flow_path.format.linear_colorspace_settings.name = "Non-Color"
        self.flow_path.format.file_format = "OPEN_EXR"
        self.flow_path.format.exr_codec = exr_codec
        self.flow_path.format.color_mode = "RGBA"
        self.flow_path.base_path = str(self.root_path / "flows")
        self.flow_path.file_slots[0].path = f"flow_{'#' * 6}"

        vec2rgba = self.tree.nodes.new("CompositorNodeGroup")
        vec2rgba.label = "Vector2RGBA"
        vec2rgba.node_tree = vec2rgba_node_group()

        self.tree.links.new(self.render_layers.outputs["Vector"], vec2rgba.inputs["Image"])
        self.tree.links.new(vec2rgba.outputs["Image"], self.flow_path.inputs["Image"])

    @require_initialized_service
    def exposed_include_segmentations(self, shuffle=True, debug=True, seed=1234, exr_codec="ZIP") -> None:
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
        # TODO: Enable assignment of custom IDs for certain objects via a dictionary.

        if self.scene.render.engine.upper() != "CYCLES":
            raise RuntimeError("Cannot produce segmentation maps when not using CYCLES.")

        self.view_layer.use_pass_object_index = True
        (self.root_path / "segmentations").mkdir(parents=True, exist_ok=True)

        # Assign IDs to every object (background will be 0)
        indices = np.arange(len(bpy.data.objects))

        if shuffle:
            np.random.seed(seed=seed)
            np.random.shuffle(indices)

        for i, obj in zip(indices, bpy.data.objects):
            obj.pass_index = i + 1

        if debug:
            seg_group = self.tree.nodes.new("CompositorNodeGroup")
            seg_group.label = "SegmentationDebug"
            seg_group.node_tree = segmentationdebug_node_group()
            seg_group.node_tree.nodes["NormalizeIdx"].inputs["From Max"].default_value = len(bpy.data.objects)

            debug_seg_path = self.tree.nodes.new(type="CompositorNodeOutputFile")
            debug_seg_path.base_path = str(self.root_path / "segmentations")
            debug_seg_path.label = "Segmentations Debug Output"
            debug_seg_path.file_slots[0].path = f"debug_segmentation_{'#' * 6}"
            debug_seg_path.format.file_format = "PNG"
            debug_seg_path.format.compression = 90
            debug_seg_path.format.color_depth = "8"
            debug_seg_path.format.color_mode = "RGB"

            # Important! Set the view settings to raw otherwise result is tonemapped
            if bpy.app.version >= (3, 2, 0):
                debug_seg_path.format.color_management = "OVERRIDE"
            debug_seg_path.format.view_settings.view_transform = "Raw"
            debug_seg_path.format.view_settings.look = "None"
            debug_seg_path.format.view_settings.gamma = 0
            debug_seg_path.format.view_settings.exposure = 1
            debug_seg_path.format.view_settings.use_curve_mapping = False

            if bpy.app.version >= (4, 3, 0):
                debug_seg_path.format.view_settings.use_white_balance = False

            self.tree.links.new(self.render_layers.outputs["IndexOB"], seg_group.inputs["Value"])
            self.tree.links.new(seg_group.outputs["Image"], debug_seg_path.inputs[0])

        self.segmentation_path = self.tree.nodes.new(type="CompositorNodeOutputFile")
        self.segmentation_path.label = "Segmentation Output"

        if bpy.app.version >= (3, 2, 0):
            self.segmentation_path.format.color_management = "OVERRIDE"
            self.segmentation_path.format.linear_colorspace_settings.name = "Non-Color"
        self.segmentation_path.format.file_format = "OPEN_EXR"
        self.segmentation_path.format.exr_codec = exr_codec

        if bpy.app.version < (4, 3, 0):
            self.segmentation_path.format.color_mode = "RGB"
        else:
            self.segmentation_path.format.color_mode = "BW"

        self.tree.links.new(self.render_layers.outputs["IndexOB"], self.segmentation_path.inputs[0])
        self.segmentation_path.base_path = str(self.root_path / "segmentations")
        self.segmentation_path.file_slots[0].path = f"segmentation_{'#' * 6}"

    @require_initialized_service
    def exposed_load_addons(self, *addons: str) -> None:
        """Load blender addons by name (case-insensitive).

        Args:
            *addons (str): name of addons to load.
        """
        for addon in addons:
            addon = addon.strip().lower()
            addon_module = addon_utils.enable(addon, default_set=True)
            self.log.info(f"Loaded addon {addon}: {addon_module}")

    @require_initialized_service
    def exposed_set_resolution(
        self, height: tuple[int] | list[int] | int | None = None, width: int | None = None
    ) -> None:
        """Set frame resolution (height, width) in pixels.
        If a single tuple is passed, instead of using keyword arguments, it will be parsed as (height, width).

        Args:
            height (tuple[int] | list[int] | int | None, optional): Height of render in pixels. Defaults to value from file.
            width (int | None, optional): Width of render in pixels. Defaults to value from file.

        Raises:
            ValueError: raised if resolution is not understood.
        """
        if isinstance(height, (tuple, list)):
            if width is not None:
                raise ValueError(
                    "Cannot understand desired resolution, either pass a (h, w) tuple, or use keyword arguments."
                )
            height, width = height  # type: ignore

        if height:
            self.scene.render.resolution_y = int(height)
        if width:
            self.scene.render.resolution_x = int(width)
        self.scene.render.resolution_percentage = 100

    @require_initialized_service
    def exposed_image_settings(
        self, file_format: str | None = None, bit_depth: int | None = None, color_mode: str | None = None
    ) -> None:
        """Set the render's output format and bit-depth.
        Useful for linear intensity renders, using "OPEN_EXR" and 32 or 16 bits.

        Note: A default arguments of ``None`` means do not change setting inherited from blendfile.

        Args:
            file_format (str | None, optional): Format to save render as. Options vary depending on the version of Blender,
                with the following being broadly available: ('BMP', 'IRIS', 'PNG', 'JPEG', 'JPEG2000', 'TARGA', 'TARGA_RAW',
                'CINEON', 'DPX', 'OPEN_EXR_MULTILAYER', 'OPEN_EXR', 'HDR', 'TIFF', 'WEBP', 'AVI_JPEG', 'AVI_RAW', 'FFMPEG').
                Defaults to None.
            bit_depth (int | None, optional): Bit depth per channel, also referred to as color-depth. Options depend on the
                chosen file format, with 8, 16 and 32bits being common. Defaults to None.
            color_mode (str | None, optional): Typically one of ('BW', 'RGB', 'RGBA'). Defaults to None.
        """
        if file_format is not None:
            self.scene.render.image_settings.file_format = file_format.upper()
        if bit_depth is not None:
            self.scene.render.image_settings.color_depth = str(bit_depth)
        if color_mode is not None:
            self.scene.render.image_settings.color_mode = color_mode.upper()

    @require_initialized_service
    def exposed_use_motion_blur(self, enable: bool) -> None:
        """Enable/disable motion blur.

        Args:
            enable (bool): If true, enable motion blur.
        """
        self.scene.render.use_motion_blur = enable

    @require_initialized_service
    def exposed_use_animations(self, enable: bool) -> None:
        """Enable/disable all animations.

        Args:
            enable (bool): If true, enable animations.
        """
        for action in bpy.data.actions:
            for fcurve in action.fcurves or []:
                if fcurve not in self.disabled_fcurves:
                    fcurve.mute = not enable
        self.use_animation = enable

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

        Note: A default arguments of ``None`` means do not change setting inherited from blendfile.

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
            ValueError: raised when setting ``use_cpu`` is required.

        Returns:
            list[str]: Name of activated devices.
        """
        if self.scene.render.engine.upper() != "CYCLES":
            self.log.warning(
                f"Using {self.scene.render.engine.upper()} rendering engine, with default OpenGL rendering device(s)."
            )
            return []

        if use_denoising is not None:
            self.scene.cycles.use_denoising = use_denoising

        if max_samples is not None:
            self.scene.cycles.samples = max_samples

        if adaptive_threshold is not None:
            self.scene.cycles.adaptive_threshold = adaptive_threshold

        preferences = bpy.context.preferences
        cycles_preferences = preferences.addons["cycles"].preferences
        cycles_preferences.refresh_devices()

        if not cycles_preferences.devices:
            raise RuntimeError("No devices found!")

        if device_type:
            if use_cpu is None:
                raise ValueError("Parameter `use_cpu` needs to be set if setting device type.")

            for device in cycles_preferences.devices:
                device.use = False

            activated_devices = []
            devices = filter(lambda d: d.type.upper() == device_type.upper(), cycles_preferences.devices)

            for device in itertools.chain(devices, filter(lambda d: d.type == "CPU" and use_cpu, devices)):
                self.log.info(f"Activated device {device.name}, {device.type}")
                activated_devices.append(device.name)
                device.use = True
            cycles_preferences.compute_device_type = "NONE" if device_type.upper() == "CPU" else device_type.upper()
            self.scene.cycles.device = "CPU" if device_type.upper() == "CPU" else "GPU"
            return activated_devices
        return []

    @require_initialized_service
    def exposed_unbind_camera(self) -> None:
        """Remove constraints, animations and parents from main camera.

        Note: In order to undo this, you'll need to re-initialize.
        """
        for c in self.camera.constraints:
            self.camera.constraints.remove(c)
        self.camera.animation_data_clear()
        self.unbind_camera = True
        self.camera.parent = None

    @require_initialized_service
    def exposed_move_keyframes(self, scale=1.0, shift=0.0) -> None:
        """Adjusts keyframes in Blender animations, keypoints are first scaled then shifted.

        Args:
            scale (float, optional): Factor used to rescale keyframe positions along x-axis. Defaults to 1.0.
            shift (float, optional): Factor used to shift keyframe positions along x-axis. Defaults to 0.0.

        Raises:
            RuntimeError: raised if trying to move keyframes beyond blender's limits.
        """
        # TODO: Refactor this into `exposed_remap_keyframes`, which would allow arbitrary transformation
        #   using a user-supplied remapping function, and redefine move_keyframes in terms of it.
        # TODO: This method can be slow if there's a lot of keyframes
        #   See: https://blender.stackexchange.com/questions/111644
        if scale == 1.0 and shift == 0.0:
            return

        # No idea why, but if we don't break this out into separate
        # variables the value we store is incorrect, often off by one.
        # We add, then remove one because frame_start and frame_end are inclusive,
        # consider [0, 99], which has length of 100, if scaled by 5, we'd get
        # [0, 495] which has length of 496 instead of 500. So we make end exclusive,
        # shift and scale it, then make it inclusive again.
        start = round(self.scene.frame_start * scale + shift)
        end = round((self.scene.frame_end + 1) * scale + shift) - 1

        if start < 0 or start >= 1_048_574 or end < 0 or end >= 1_048_574:
            raise RuntimeError(
                "Cannot scale and shift keyframes past render limits. For more please see: "
                "https://docs.blender.org/manual/en/latest/advanced/limits.html"
            )
        self.scene.frame_start = start
        self.scene.frame_end = end

        for action in bpy.data.actions:
            for fcurve in action.fcurves or []:
                for kfp in fcurve.keyframe_points or []:
                    # Note: Updating `co_ui` does not move the handles properly!!!
                    kfp.co.x = kfp.co.x * scale + shift
                    kfp.handle_left.x = kfp.handle_left.x * scale + shift
                    kfp.handle_right.x = kfp.handle_right.x * scale + shift
                    kfp.period *= scale
        self.scene.render.motion_blur_shutter /= scale

    @require_initialized_service
    def exposed_set_current_frame(self, frame_number: int) -> None:
        """Set current frame number. This might advance any animations.

        Args:
            frame_number (int): index of frame to skip to.
        """
        self.scene.frame_set(frame_number)

    @require_initialized_service
    def exposed_camera_extrinsics(self) -> npt.NDArray[np.floating]:
        """Get the 4x4 transform matrix encoding the current camera pose.

        Returns:
            npt.NDArray[np.floating]: Current camera pose in matrix form.
        """
        pose = np.array(self.camera.matrix_world)
        pose[:3, :3] /= np.linalg.norm(pose[:3, :3], axis=0)
        return pose

    @require_initialized_service
    def exposed_camera_intrinsics(self) -> npt.NDArray[np.floating]:
        """Get the 3x3 camera intrinsics matrix for active camera,
        which defines how 3D points are projected onto 2D.

        Note: Assumes pinhole camera model.

        Returns:
            npt.NDArray[np.floating]: Camera intrinsics matrix based on camera properties.
        """
        scale = self.scene.render.resolution_percentage / 100
        width = self.scene.render.resolution_x * scale
        height = self.scene.render.resolution_y * scale
        camera_data = self.scene.camera.data

        aspect_ratio = width / height
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = width / 2.0 / np.tan(camera_data.angle / 2)
        K[1, 1] = height / 2.0 / np.tan(camera_data.angle / 2) * aspect_ratio
        K[0, 2] = width / 2.0
        K[1, 2] = height / 2.0
        return K

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

        Note: Only one of ``look_at`` or ``rotation`` can be set at once.

        Args:
            location (npt.ArrayLike, optional): Location to place camera in 3D space. Defaults to none.
            rotation (npt.ArrayLike, optional): Rotation matrix for camera. Defaults to none.
            look_at (npt.ArrayLike, optional): Location to point camera. Defaults to none.
            in_order (bool, optional): If set, assume current camera pose is from previous/next
                frame and ensure new rotation set by ``look_at`` is compatible with current position.
                Without this, a rotations will stay in the [-pi, pi] range and this wrapping will
                mess up interpolations. Only used when ``look_at`` is set. Defaults to True.

        Raises:
            ValueError: raised if camera orientation is over-defined.
        """
        if look_at is not None and rotation is not None:
            raise ValueError("Only one of `look_at` or `rotation` can be set.")

        if location is not None:
            self.camera.location = mathutils.Vector(location)

        if look_at is not None:
            # point the camera's '-Z' towards `look_at` and use its 'Y' as up
            direction = mathutils.Vector(look_at) - self.camera.location
            rot_euler = direction.to_track_quat("-Z", "Y").to_euler()

            if in_order:
                rot_euler.make_compatible(self.camera.rotation_euler)
            self.camera.rotation_euler = rot_euler

        if rotation is not None:
            location = self.camera.location.copy()
            self.camera.matrix_world = mathutils.Matrix(rotation).to_4x4()
            self.camera.location = location
        self.view_layer.update()

    @require_initialized_service
    @validate_camera_moved
    def exposed_rotate_camera(self, angle: float) -> None:
        """Rotate camera around it's optical axis, relative to current orientation.

        Args:
            angle: Relative amount to rotate by (clockwise, in radians).
        """
        # Camera's '-Z' point outwards, so we negate angle such that
        # camera turns clockwise for a positive angle
        rot_euler = self.camera.rotation_euler.copy()
        rot_euler.rotate_axis("Z", -angle)
        rot_euler.make_compatible(self.camera.rotation_euler)
        self.camera.rotation_euler = rot_euler
        self.view_layer.update()

    @require_initialized_service
    def exposed_set_camera_keyframe(self, frame_num: int, matrix: npt.ArrayLike | None = None) -> None:
        """Set camera keyframe at given frame number.
        If camera matrix is not supplied, currently set camera position/rotation/scale will be used,
        this allows users to set camera position using :meth:`position_camera <exposed_position_camera>`
        and :meth:`rotate_camera <exposed_rotate_camera>`.

        Args:
            frame_num (int): index of frame to set keyframe for.
            matrix (npt.ArrayLike | None, optional): 4x4 camera transform, if not supplied,
                use current camera matrix. Defaults to None.
        """
        if matrix is not None:
            self.camera.matrix_world = mathutils.Matrix(matrix)
        self.camera.keyframe_insert(data_path="location", frame=frame_num)
        self.camera.keyframe_insert(data_path="rotation_euler", frame=frame_num)
        self.camera.keyframe_insert(data_path="scale", frame=frame_num)

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
        if start is not None:
            self.scene.frame_start = start
        if stop:
            self.scene.frame_end = stop - 1
        if step is not None:
            self.scene.frame_step = step

    @require_initialized_service
    def exposed_render_current_frame(self, allow_skips=True, dry_run=False) -> dict[str, Any]:
        """Generates a single frame in Blender at the current camera location,
        return the file paths for that frame, potentially including depth, normals, etc.

        Args:
            allow_skips (bool, optional): if true, blender will not re-render and overwrite existing frames.
                This does not however apply to depth/normals/etc, which cannot be skipped. Defaults to True.
            dry_run (bool, optional): if true, nothing will be rendered at all. Defaults to False.

        Returns:
            dict[str, Any]: dictionary containing paths to rendered frames for this index and camera pose.
        """
        # TODO: Implement skipping for depth/normals/flow?

        # Assumes the camera position, frame number and all other params have been set
        index = self.scene.frame_current
        self.scene.render.filepath = str(self.root_path / "frames" / f"frame_{index:06}")
        paths = {"file_path": Path(f"frames/frame_{index:06}").with_suffix(self.scene.render.file_extension)}

        if self.depth_path is not None:
            paths["depth_file_path"] = Path(f"depths/depth_{index:06}{self.depth_extension}")
        if self.normal_path is not None:
            paths["normal_file_path"] = Path(f"normals/normal_{index:06}.exr")
        if self.flow_path is not None:
            paths["flow_file_path"] = Path(f"flows/flow_{index:06}.exr")
        if self.segmentation_path is not None:
            paths["segmentation_file_path"] = Path(f"segmentations/segmentation_{index:06}.exr")

        if not dry_run:
            # Render frame(s), skip the render iff all files exist and `allow_skips`
            if not allow_skips or any(not Path(self.root_path / p).exists() for p in paths.values()):
                # If `write_still` is false, depth & normals can be written but rgb will be skipped
                skip_frame = Path(self.root_path / paths["file_path"]).exists() and allow_skips
                bpy.ops.render.render(animation=False, write_still=not skip_frame)

        # Returns paths that were written
        return {
            **{k: str(p) for k, p in paths.items()},
            "transform_matrix": self.exposed_camera_extrinsics().tolist(),
        }

    @require_initialized_service
    def exposed_render_frame(self, frame_number: int, allow_skips=True, dry_run=False) -> dict[str, Any]:
        """Same as first setting current frame then rendering it.

        Warning:
            Calling this has the side-effect of changing the current frame.

        Args:
            frame_number (int): frame to render
            allow_skips (bool, optional): if true, blender will not re-render and overwrite existing frames.
                This does not however apply to depth/normals/etc, which cannot be skipped. Defaults to True.
            dry_run (bool, optional): if true, nothing will be rendered at all. Defaults to False.

        Returns:
            dict[str, Any]: dictionary containing paths to rendered frames for this index and camera pose.
        """
        self.exposed_set_current_frame(frame_number)
        return self.exposed_render_current_frame(allow_skips=allow_skips, dry_run=dry_run)

    @require_initialized_service
    def exposed_render_frames(
        self, frame_numbers: Iterable[int], allow_skips=True, dry_run=False, update_fn: UpdateFn | None = None
    ) -> dict[str, Any]:
        """Render all requested frames and return associated transforms dictionary.

        Args:
            frame_numbers (Iterable[int]): frames to render.
            allow_skips (bool, optional): if true, blender will not re-render and overwrite existing frames.
                This does not however apply to depth/normals/etc, which cannot be skipped. Defaults to True.
            dry_run (bool, optional): if true, nothing will be rendered at all. Defaults to False.
            update_fn (UpdateFn, optional): callback function to track render progress. Will first be called with ``total`` kwarg,
                indicating number of steps to be taken, then will be called with ``advance=1`` at every step. Closely mirrors the
                `rich.Progress API <https://rich.readthedocs.io/en/stable/reference/progress.html#rich.progress.Progress.update>`_.
                Defaults to None.

        Raises:
            RuntimeError: raised if trying to render frames beyond blender's limits.

        Returns:
            dict[str, Any]: transforms dictionary containing paths to rendered frames, camera poses and intrinsics.
        """
        # Ensure frame_numbers is a list to find extrema
        frame_numbers = list(frame_numbers)

        # Set total number of steps
        if update_fn is not None:
            update_fn(total=len(frame_numbers))

        # Max number of frames is 1,048,574 as of Blender 3.4
        # See: https://docs.blender.org/manual/en/latest/advanced/limits.html
        if (frame_end := max(frame_numbers)) >= 1_048_574:
            raise RuntimeError(
                f"Blender cannot currently render more than 1,048,574 frames, yet requested you "
                f"requested {max(frame_numbers)} frames to be rendered. For more please see: "
                f"https://docs.blender.org/manual/en/latest/advanced/limits.html"
            )
        if (frame_start := min(frame_numbers)) < 0:
            raise RuntimeError("Cannot render frames at negative indices. You can try shifting keyframes.")

        # Warn if requested frames lie outside animation range
        if self.use_animation and (frame_start < self.scene.frame_start or self.scene.frame_end < frame_end):
            self.log.warning(
                f"Current animation starts at frame #{self.scene.frame_start} and ends at "
                f"#{self.scene.frame_end} (with step={self.scene.frame_step}), but you requested "
                f"some frames between #{frame_start} and to #{frame_end} to be rendered.\n"
            )

        # If we request to render frames outside the nominal animation range,
        # blender will just wrap around and go back to that range. As a workaround,
        # extend the animation range to it's maximum, even if we do not exceed it,
        # it will be restored at the end.
        scene_original_range = self.scene.frame_start, self.scene.frame_end
        self.scene.frame_start, self.scene.frame_end = 0, 1_048_574

        # Store transforms as we go
        transforms = self.exposed_empty_transforms()

        # Capture frames!
        for frame_number in frame_numbers:
            # Tell blender to update camera position and all animations and render frame
            frame_data = self.exposed_render_frame(frame_number, allow_skips=allow_skips, dry_run=dry_run)
            transforms["frames"].append(frame_data)

            # Call any progress callbacks
            if update_fn is not None:
                update_fn(advance=1)

        # Restore animation range to original values
        self.scene.frame_start, self.scene.frame_end = scene_original_range
        return transforms

    @require_initialized_service
    def exposed_render_animation(
        self,
        frame_start: int | None = None,
        frame_end: int | None = None,
        frame_step: int | None = None,
        allow_skips=True,
        dry_run=False,
        update_fn: UpdateFn | None = None,
    ) -> dict[str, Any]:
        """Determines frame range to render, sets camera positions and orientations, and renders all frames in animation range.

        Note: All frame start/end/step arguments are absolute quantities, applied after any keyframe moves.
              If the animation is from (1-100) and you've scaled it by calling :meth:`move_keyframes(scale=2.0) <exposed_move_keyframes>`
              then calling :meth:`render_animation(frame_start=1, frame_end=100) <exposed_render_animation>` will only render half of the animation.
              By default the whole animation will render when no start/end and step values are set.

        Args:
            frame_start (int, optional): Starting index (inclusive) of frames to render as seen in blender. Defaults to None, meaning value from ``.blend`` file.
            frame_end (int, optional): Ending index (inclusive) of frames to render as seen in blender. Defaults to None, meaning value from ``.blend`` file.
            frame_step (int, optional): Skip every nth frame. Defaults to None, meaning value from ``.blend`` file.
            allow_skips (bool, optional): Same as :meth:`render_current_frame <exposed_render_current_frame>`.
            dry_run (bool, optional): Same as :meth:`render_current_frame <exposed_render_current_frame>`.
            update_fn (UpdateFn, optional): Same as :meth:`render_frames <exposed_render_frames>`.

        Raises:
            ValueError: raised if scene and camera are entirely static.

        Returns:
            dict[str, Any]: transforms dictionary containing paths to rendered frames, camera poses and intrinsics.
        """
        frame_start = self.scene.frame_start if frame_start is None else frame_start
        frame_end = self.scene.frame_end if frame_end is None else frame_end
        frame_step = self.scene.frame_step if frame_step is None else frame_step
        frame_range = range(frame_start, frame_end + 1, frame_step)

        if not self.use_animation:
            raise ValueError(
                "Animations are disabled, scene will be entirely static. "
                "To instead render a single frame, use `render_frame`."
            )
        elif all(p.animation_data is None for p in self.get_parents(self.camera)) and self.camera.animation_data is None:
            self.log.warning("Active camera nor it's parents are animated, camera will be static.")

        return self.exposed_render_frames(frame_range, allow_skips=allow_skips, dry_run=dry_run, update_fn=update_fn)

    @require_initialized_service
    def exposed_save_file(self, path: str | os.PathLike) -> None:
        """Save the opened blender file. This is useful for introspecting the state of the compositor/scene/etc.

        Args:
            path (str | os.PathLike): path where to save blendfile.

        Raises:
            ValueError: raised if file already exists.
        """
        if (path := Path(str(path)).resolve()) == Path(str(self.blend_file)).resolve():
            raise ValueError("Cannot overwrite currently loaded blend-file!")

        self.log.info(f"Saving scene to {path}...")
        path.parent.mkdir(exist_ok=True, parents=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(path))


class BlenderClient:
    """Client-side API to interact with blender and render novel views.

    The :class:`BlenderClient` is responsible for communicating with (and potentially spawning)
    separate :class:`BlenderServer`s that will actually perform the rendering via a :class:`BlenderService`.

    The client acts as a context manager, it will connect to it's server when the context is
    entered and cleanly disconnect and close the connection in case of errors or when exiting
    the with-block.

    Many useful methods to interact with blender are provided, such as
    :meth:`set_resolution <BlenderService.exposed_set_resolution>` or
    :meth:`render_animation <BlenderService.exposed_render_animation>`.
    These methods are dynamically generated when the client connects to
    the server. Available methods are directly inherited from :class:`BlenderService`
    (or whichever service the server is exposing), specifically any service method
    starting with ``exposed_`` will be accessible to the client at runtime.
    For example, ``BlenderClient.include_depths`` is a remote procedure call
    to :meth:`BlenderService.exposed_include_depths`.
    """

    def __init__(self, addr: tuple[str, int], timeout: float = 10.0) -> None:
        """Initialize a client with known address of server.
        Note: Using :meth:`auto_connect` or :meth:`spawn` is often more convenient.

        Args:
            addr (tuple[str, int]): Connection tuple containing the hostname and port
            timeout (float, optional): Maximum time in seconds the client will attempt
                to connect to the server for before an error is thrown. Only used when
                entering context manager. Defaults to 10 seconds.
        """
        self.addr: tuple[str, int] = addr
        self.conn: rpyc.Connection | None = None
        self.awaitable: rpyc.AsyncResult | None = None
        self.process: subprocess.Popen | None = None
        self.timeout: float = timeout

    @classmethod
    def auto_connect(cls, timeout: float = 10.0) -> Self:
        """Automatically connect to available server.

        Use :meth:`BlenderServer.discover` to find available server within ``timeout``.

        Note: This doesn't actually connect to the server instance, the connection happens
            when the context manager is entered. This simply creates a client instance with
            the connection settings (i.e: hostname, port) of an existing server. The connection
            might still fail when entering the with-block.

        Args:
            timeout (float, optional): try to discover server instance for ``timeout``
                (in seconds) before giving up. Defaults to 10.0 seconds.

        Raises:
            TimeoutError: raise if unable to discover server in ``timeout`` seconds.

        Returns:
            Self: client instance initialized with connection settings of existing server.
        """
        start = time.time()

        while True:
            if (time.time() - start) > timeout:
                raise TimeoutError("Unable to discover server in alloted time.")
            if conns := set(BlenderServer.discover()):
                break
            time.sleep(0.1)
        return cls(conns.pop())

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
        The spawned process is accessible through the client's ``process`` attribute.

        Args:
            timeout (float, optional): try to discover spawned instances for ``timeout``
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

        Returns:
            Self: the connected client
        """
        with BlenderServer.spawn(jobs=1, timeout=timeout, log_dir=log_dir, autoexec=autoexec, executable=executable) as (
            procs,
            conns,
        ):
            with cls(conns.pop()) as client:
                client.process = procs[0]
                yield client
                client.process = None

    @require_connected_client
    def render_animation_async(self, *args, **kwargs) -> rpyc.AsyncResult:
        """Asynchronously call :meth:`render_animation <BlenderService.exposed_render_animation>`
        and return an rpyc.AsyncResult.

        Args:
            *args: Same as :meth:`BlendService.exposed_render_animation`
            *kwargs: Same as :meth:`BlendService.exposed_render_animation`

        Returns:
            rpyc.AsyncResult: Result encapsulating the return value of ``render_animation``.
                After ``wait``ing for the render to finish, it can be accessed using
                the ``.value`` attribute.
        """
        self.conn = cast(rpyc.Connection, self.conn)
        render_animation_async = rpyc.async_(self.conn.root.render_animation)
        async_result = render_animation_async(*args, **kwargs)
        self.awaitable = async_result
        async_result.add_callback(lambda _: setattr(self, "awaitables", None))
        return async_result

    @require_connected_client
    def render_frames_async(self, *args, **kwargs) -> rpyc.AsyncResult:
        """Asynchronously call :meth:`render_frames <BlenderService.exposed_render_frames>`
        and return an rpyc.AsyncResult.

        Args:
            *args: Same as :meth:`BlendService.exposed_render_frames`
            *kwargs: Same as :meth:`BlendService.exposed_render_frames`

        Returns:
            rpyc.AsyncResult: Result encapsulating the return value of ``render_frames``.
                After ``wait``ing for the render to finish, it can be accessed using
                the ``.value`` attribute.
        """
        self.conn = cast(rpyc.Connection, self.conn)
        render_frames_async = rpyc.async_(self.conn.root.render_frames)
        async_result = render_frames_async(*args, **kwargs)
        self.awaitable = async_result
        async_result.add_callback(lambda _: setattr(self, "awaitables", None))
        return async_result

    def wait(self) -> None:
        """Block and await any async results."""
        if self.awaitable:
            self.awaitable.wait()

    def __enter__(self) -> Self:
        """Connect to the render server via a context manager.

        Raises:
            TimeoutError: raised if unable to connect to server in time.
        """
        # Loop until we connect or timeout
        start = time.time()

        while True:
            try:
                self.conn = rpyc.connect(*self.addr, config={"sync_request_timeout": -1, "allow_all_attrs": True})
                break
            except ConnectionRefusedError:
                pass

            if (time.time() - start) > self.timeout:
                raise TimeoutError("Unable to connect to server in alloted time.")
            time.sleep(0.1)

        # Setup default logger for client, otherwise warnings
        # in the render service won't propagate to the client
        logger = logging.getLogger(__name__ + str(self.addr))
        logger.setLevel(logging.INFO)
        self.with_logger(logger)
        return self

    def __getattr__(self, name: str) -> rpyc.BaseNetref:
        """Retrieve remote attribute if client is connected.
        This will be called when local attribute is not found.

        Args:
            name (str): Name of attribute.

        Raises:
            AttributeError: raised if attribute is not found.

        Returns:
            rpyc.BaseNetref: remote proxy object.
        """
        # Spoof all `exposed_` methods from the service
        if self.conn is not None and EXPOSED_PREFIX + name in dir(self.conn.root):
            return getattr(self.conn.root, EXPOSED_PREFIX + name)
        raise AttributeError

    def __exit__(
        self, type: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        """Disconnect to the render server via a context manager.

        Args:
            type (type[BaseException] | None): Type of exception that was caught, if any.
            value (BaseException | None): Value of exception if any.
            traceback (TracebackType | None): Traceback of exception if any.
        """
        if self.conn is not None:
            self.conn.close()


class BlenderClients(tuple):
    """Collection of :class:`BlenderClient` instances.

    Most methods in this class simply call the equivalent method of each client, that is,
    calling ``clients.set_resolution`` is equivalent to calling :meth:`set_resolution <BlenderService.exposed_set_resolution>`
    for each client in clients. Some special methods, namely the :meth:`render_frames` and :meth:`render_animation`
    methods will instead distribute the rendering load to all clients.

    Finally, entering each client's context-manager, and closing each client connection
    is ensured by using this class' context-manager.
    """

    def __new__(cls, *objs: Iterator[BlenderClient | tuple[str, int]]) -> Self:
        """Create a new instance from iterable of clients, or their connection settings.

        Args:
            *objs (Iterator[BlenderClient | tuple[str, int]]): :class:`BlenderClient` instances or their hostnames and ports.

        Raises:
            TypeError: raised when input objects are of incorrect type.
        """
        clients = [BlenderClient(o) if isinstance(o, tuple) else o for o in objs]
        if not all(isinstance(o, BlenderClient) for o in clients):
            raise TypeError("'BlenderClients' can only contain 'BlenderClient' instances or their hostnames and ports.")
        return super().__new__(cls, clients)

    def __init__(self, *objs) -> None:
        """Initialize collection of :class:`BlenderClient` from iterable of clients, or their connection settings.

        Args:
            *objs (Iterator[BlenderClient | tuple[str, int]]): :class:`BlenderClient` instances or their hostnames and ports.
        """
        # Note: At this point the tuple is already initialized because of __new__, i.e: objs == list(self)
        # Not sure why mypy can't resolve the following...
        self.stack: ExitStack = ExitStack()

    def _method_dispatch_factory(self, name: str, method: Callable) -> Callable:
        @functools.wraps(method)
        def inner(*args, **kwargs):
            # Call method for each client, collect results into tuple
            retval = tuple(getattr(client, name)(*args, **kwargs) for client in self)
            return None if all(i is None for i in retval) else retval

        return inner

    def __enter__(self) -> Self:
        """Connect all clients to their render servers via a context manager.

        Raises:
            TimeoutError: raised if unable to connect to servers in time.
        """
        self.stack.__enter__()
        for client in self:
            # Enter each client's context, connecting them all to servers
            self.stack.enter_context(client)

            # Dynamically generate methods that dispatch to all clients
            # TODO: We currently assume all clients use `BlenderService`.
            # TODO: Move this to a __getattr__ method like in BlenderClient!
            for method_name in dir(BlenderService):
                if method_name.startswith(EXPOSED_PREFIX):
                    name = method_name.removeprefix(EXPOSED_PREFIX)

                    if name not in dir(self):
                        method = getattr(BlenderService, method_name)
                        multicall = self._method_dispatch_factory(name, method)
                        setattr(self, name, multicall)
        return self

    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Disconnect from each render server via a context manager.

        Args:
            type (type[BaseException] | None): Type of exception that was caught, if any.
            value (BaseException | None): Value of exception if any.
            traceback (TracebackType | None): Traceback of exception if any.
        """
        self.stack.__exit__(type, value, traceback)

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
        The spawned processes are accessible through the client's ``process`` attribute.

        Args:
            jobs (int, optional): number of jobs to spawn. Defaults to 1.
            timeout (float, optional): try to discover spawned instances for ``timeout``
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

        Returns:
            Self: the connected clients
        """
        with BlenderServer.spawn(
            jobs=jobs, timeout=timeout, log_dir=log_dir, autoexec=autoexec, executable=executable
        ) as (procs, conns):
            with cls(*conns) as clients:
                for client, p in zip(clients, procs):
                    client.process = p

                yield clients

                for client in clients:
                    client.process = None

    @staticmethod
    @contextmanager
    def pool(
        jobs: int = 1,
        timeout: float = -1.0,
        log_dir: str | os.PathLike | None = None,
        autoexec: bool = False,
        executable: str | os.PathLike | None = None,
        conns: list[tuple[str, int]] | None = None,
    ) -> Iterator[multiprocess.Pool]:
        """Spawns a multiprocessing-like worker pool, each with their own :class:`BlenderClient` instance.
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
            Here we use ``multiprocess`` instead of the builtin multiprocessing library to take
            advantage of the more advanced dill serialization (as opposed to the standard pickling).

        Args:
            jobs (int, optional): number of jobs to spawn. Defaults to 1.
            timeout (float, optional): try to discover spawned instances for ``timeout``
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
            conns: List of connection tuples containing the hostnames and ports of existing servers.
                If specified, the pool will use these servers (and ``jobs`` and other spawn arguments will
                be ignored) instead of spawning new ones.

        Returns:
            multiprocess.Pool: A ``multiprocess.Pool`` instance which has had it's applicator methods
                (map/imap/starmap/etc) monkey-patched to inject a client instance as first argument.
        """
        # Note import here as this is a dependency only on the client-side
        import multiprocess
        import multiprocess.pool

        def inject_client(func, conns):
            # Note: Usually it's good practice to add `@functools.wraps(func)`
            # here, but it makes dill freak out with a rather cryptic
            # "disallowed for security reasons" error... Works fine otherwise.
            def inner(*args, **kwargs):
                conn = conns.get()

                try:
                    with BlenderClient(conn) as client:
                        retval = func(client, *args, **kwargs)
                finally:
                    conns.put(conn)
                return retval

            return inner

        def modify_applicator(applicator, conns):
            @functools.wraps(applicator)
            def inner(func, *args, **kwargs):
                func = inject_client(func, conns)
                return applicator(func, *args, **kwargs)

            return inner

        context_manager = (
            BlenderServer.spawn(jobs=jobs, timeout=timeout, log_dir=log_dir, autoexec=autoexec, executable=executable)
            if conns is None
            else nullcontext(enter_result=(None, conns))
        )

        with multiprocess.Manager() as manager:
            with context_manager as (_, conns):
                q = manager.Queue()

                for conn in conns:
                    q.put(conn)

                with multiprocess.Pool(len(conns)) as pool:
                    for name, method in inspect.getmembers(pool, predicate=inspect.ismethod):
                        params = list(inspect.signature(method).parameters.keys())

                        # Get all map/starmap/apply/etc variants
                        if not name.startswith("_") and next(iter(params), None) == "func":
                            setattr(pool, name, modify_applicator(method, q))
                    yield pool

    @require_connected_clients
    def common_animation_range(self) -> range:
        """Get animation range shared by all clients as range(start, end+1, step).

        Raises:
            RuntimeError: animation ranges for all clients are expected to be the same.

        Returns:
            range: Range of frames in animation.
        """
        start, end, step = self.common_animation_range_tuple()
        return range(start, end + 1, step)

    @require_connected_clients
    def common_animation_range_tuple(self) -> tuple[int, int, int]:
        """Get animation range shared by all clients as a tuple of (start, end, step).

        Raises:
            RuntimeError: animation ranges for all clients are expected to be the same.

        Returns:
            tuple[int, int, int]: Frame start, end, and step of animation.
        """
        if len(ranges := set(self.animation_range_tuple())) != 1:  # type: ignore
            raise RuntimeError("Found different animation ranges. All connected servers should be in the same state.")
        return ranges.pop()

    @require_connected_clients
    def render_frames(
        self,
        frame_numbers: Collection[int],
        allow_skips: bool = True,
        dry_run: bool = False,
        update_fn: UpdateFn | None = None,
    ) -> dict[str, Any]:
        """Render all requested frames by distributing workload across connected clients and return associated transforms dictionary.

        Warning:
            Assumes all clients are initialized in the same manner, that is, to the same blendfile, with the same animation range,
            render settings, etc.

        Args:
            frame_numbers (Collection[int]): frames to render.
            allow_skips (bool, optional): if true, blender will not re-render and overwrite existing frames.
                This does not however apply to depth/normals/etc, which cannot be skipped. Defaults to True.
            dry_run (bool, optional): if true, nothing will be rendered at all. Defaults to False.
            update_fn (UpdateFn, optional): callback function to track render progress. Will first be called with ``total`` kwarg,
                indicating number of steps to be taken, then will be called with ``advance=1`` at every step. Closely mirrors the
                `rich.Progress API <https://rich.readthedocs.io/en/stable/reference/progress.html#rich.progress.Progress.update>`_.
                Defaults to None.

        Raises:
            RuntimeError: raised if trying to render frames beyond blender's limits.

        Returns:
            dict[str, Any]: transforms dictionary containing paths to rendered frames, camera poses and intrinsics.
        """

        # Set total number of steps, disable updates of total from child processes
        def ignore_total(*args, total=None, **kwargs):
            if update_fn is not None:
                return update_fn(*args, **kwargs)

        if update_fn is not None:
            update_fn(total=len(frame_numbers))

        # Equivalent to more-itertools' distribute
        children = itertools.tee(frame_numbers, len(self))
        frame_chunks = [itertools.islice(it, index, None, len(self)) for index, it in enumerate(children)]

        transforms_dicts = [
            client.render_frames_async(frames, allow_skips=allow_skips, dry_run=dry_run, update_fn=ignore_total)
            for client, frames in zip(self, frame_chunks)
        ]
        self.wait()

        # Equivalent to more-itertools' interleave_longest
        _marker = object()
        frames = [
            frame
            for frame in itertools.chain.from_iterable(
                itertools.zip_longest(*[t.value["frames"] for t in transforms_dicts], fillvalue=_marker)
            )
            if frame is not _marker
        ]
        transforms: dict[str, Any] = transforms_dicts.pop().value
        transforms["frames"] = frames
        return transforms

    @require_connected_clients
    def render_animation(
        self,
        frame_start: int | None = None,
        frame_end: int | None = None,
        frame_step: int | None = None,
        allow_skips=True,
        dry_run=False,
        update_fn: UpdateFn | None = None,
    ) -> dict[str, Any]:
        """Determines frame range to render, sets camera positions and orientations, and renders all frames in animation range by distributing
        workload onto all connected clients.

        Note: All frame start/end/step arguments are absolute quantities, applied after any keyframe moves.
              If the animation is from (1-100) and you've scaled it by calling :meth:`move_keyframes(scale=2.0) <exposed_move_keyframes>`
              then calling :meth:`render_animation(frame_start=1, frame_end=100) <exposed_render_animation>` will only render half of the animation.
              By default the whole animation will render when no start/end and step values are set.

        Args:
            frame_start (int, optional): Starting index (inclusive) of frames to render as seen in blender. Defaults to None, meaning value from ``.blend`` file.
            frame_end (int, optional): Ending index (inclusive) of frames to render as seen in blender. Defaults to None, meaning value from ``.blend`` file.
            frame_step (int, optional): Skip every nth frame. Defaults to None, meaning value from ``.blend`` file.
            allow_skips (bool, optional): Same as :meth:`render_current_frame <exposed_render_current_frame>`.
            dry_run (bool, optional): Same as :meth:`render_current_frame <exposed_render_current_frame>`.
            update_fn (UpdateFn, optional): Same as :meth:`render_frames <exposed_render_frames>`.

        Raises:
            ValueError: raised if scene and camera are entirely static.

        Returns:
            dict[str, Any]: transforms dictionary containing paths to rendered frames, camera poses and intrinsics.
        """
        start, end, step = self.common_animation_range_tuple()
        frame_start = start if frame_start is None else frame_start
        frame_end = end if frame_end is None else frame_end
        frame_step = step if frame_step is None else frame_step
        frame_range = range(frame_start, frame_end + 1, frame_step)

        return self.render_frames(frame_range, allow_skips=allow_skips, dry_run=dry_run, update_fn=update_fn)

    @require_connected_clients
    def save_file(self, path: str | os.PathLike) -> None:
        """Save opened blender file. This is useful for introspecting the state of the compositor/scene/etc.

        Note: Only saves file once (from a single connected client), assumes all clients have
            been initialized in the same manner.

        Args:
            path (str | os.PathLike): path where to save blendfile.

        Raises:
            ValueError: raised if file already exists.
        """
        client, *_ = self
        client.save_file(path)

    def wait(self) -> None:
        """Wait for all clients at once."""
        awaitables = [client.awaitable for client in self]

        while awaitables:
            awaitables = [a for a in awaitables if a._waiting()]

            for awaitable in awaitables:
                # Here we query the property `awaitable.ready` which enables
                # the underlying connection to poll and serve any incoming events.
                # Roughly equivalent to the following (but does not rely on private API):
                #     awaitable._conn.serve(awaitable._ttl, waiting=awaitable._waiting)
                awaitable.ready


if __name__ == "__main__":
    if sys.version_info < (3, 9, 0):
        raise RuntimeError("Please use newer blender version with a python version of at least 3.9.")

    if bpy is None:
        sys.exit()

    # Get script specific arguments
    try:
        index = sys.argv.index("--") + 1
    except ValueError:
        index = len(sys.argv)

    parser = argparse.ArgumentParser("Startup a BlenderServer on a given port.")
    parser.add_argument("-p", "--port", type=int, default=0)
    args, unknown = parser.parse_known_args(sys.argv[index:])

    server = BlenderServer(port=args.port)
    server.start()
