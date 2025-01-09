import functools
import inspect
import itertools
import shlex
import signal
import site
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from multiprocessing import Process
from pathlib import Path

import numpy as np

try:
    # These are blender specific modules which aren't easily installed but
    # are loaded in when this script is ran from blender. You can install fake
    # versions of these (which only help with autocomplete really) like so:
    #   $ pip install fake-bpy-module-<blender version, i.e: 3.3>
    #
    # Note: If using blender as a module, the above doesn't apply.
    import addon_utils
    import bpy
    import mathutils
except ImportError:
    bpy = None
    mathutils = None
    addon_utils = None

try:
    sys.path.insert(0, site.USER_SITE)
    sys.path.insert(0, str(Path(__file__).parent.resolve()))

    import rpyc
    import rpyc.utils.registry
except ImportError as e:
    # Note: the same can be done for np, mathutils, etc but these
    # should already be installed by default inside of blender.
    try:
        if bpy is not None:
            print("Attempting to auto install dependencies into blender's runtime...")
            print(subprocess.check_output(shlex.split(f"{sys.executable} -m ensurepip"), universal_newlines=True))
            print(
                subprocess.check_output(
                    shlex.split(f"{sys.executable} -m pip install rpyc --target {site.USER_SITE}"),
                    universal_newlines=True,
                )
            )
            sys.exit()
        else:
            raise e
    except subprocess.CalledProcessError:
        print(
            "Some dependencies are needed to run this script. To install it so that "
            "it is accessible from blender, you need to pip install it "
            "into blender's python interpreter like so:\n"
        )
        print(f"$ {sys.executable} -m ensurepip")
        print(f"$ {sys.executable} -m pip install rpyc --target {site.USER_SITE}")

REGISTRY = None


class BlenderServer(rpyc.utils.server.Server):
    def __init__(self, hostname=None, port=0, service_klass=None, extra_config=None, **kwargs):
        if bpy is None:
            raise RuntimeError(f"{type(self).__name__} needs to be instantiated from within blender's python runtime.")
        if service_klass and not issubclass(service_klass, BlenderService):
            raise ValueError(f"Parameter 'service_klass' must be 'BlenderService' or subclass.")

        super().__init__(
            service_klass or BlenderService,
            hostname=hostname,
            port=port,
            protocol_config={"allow_all_attrs": True, "allow_setattr": True} | (extra_config or {}),
            auto_register=True,
            **kwargs,
        )
        print(f"INFO: Started listening on {self.host}:{self.port}")

    def _accept_method(self, sock):
        self._authenticate_and_serve_client(sock)

    @staticmethod
    @contextmanager
    def spawn(jobs=1, timeout=10, log_dir=None, autoexec=False, executable=None):
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
        autoexec = "--enable-autoexec" if autoexec else "--disable-autoexec"
        cmd = shlex.split(f"{executable or 'blender'} -b --python {__file__} {autoexec}")
        procs = []

        if log_dir:
            log_dir = Path(log_dir).expanduser().resolve()
            log_dir.mkdir(parents=True, exist_ok=True)

        with ExitStack() as stack:
            for i in range(jobs):
                if log_dir:
                    (log_dir / f"job{i:03}").mkdir(parents=True, exist_ok=True)
                    stdout = stack.enter_context(open(log_dir / f"job{i:03}" / "stdout.log", "w"))
                    stderr = stack.enter_context(open(log_dir / f"job{i:03}" / "stderr.log", "w"))
                else:
                    stdout, stderr = subprocess.DEVNULL, subprocess.DEVNULL
                proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, universal_newlines=True)
                procs.append(proc)
            stack.enter_context(terminate_jobs(procs))

            start = time.time()

            while True:
                if (time.time() - start) > timeout:
                    # Terminate all procs and close fds
                    stack.close()
                    raise TimeoutError("Unable to spawn and discover server(s) in alloted time.")
                if len(conns := set(BlenderServer.discover()) - set(existing)) == jobs:
                    break
                time.sleep(0.1)

            yield (procs, conns)

    @staticmethod
    def spawn_registry():
        global REGISTRY

        def launch_registry():
            try:
                registry = rpyc.utils.registry.UDPRegistryServer()
                registry.start()
            except OSError:
                # Note: Address is likely already in use, meaning there's
                #   already a spawned registry in another thread/process which
                #   we should be able to use. No need to re-spawn one then.
                pass

        if not REGISTRY or not REGISTRY[0].is_alive():
            registry = Process(target=launch_registry, daemon=True)
            client = rpyc.utils.registry.UDPRegistryClient()
            registry.start()
            REGISTRY = (registry, client)
        return REGISTRY

    @staticmethod
    def discover():
        _, client = BlenderServer.spawn_registry()
        return client.discover("BLENDER")


class BlenderClient:
    """Main API to generate a dataset of different views of a blend file, this
    blender client is responsible for communicating with (and potentially spawning)
    separate threads that will actually perform the rendering.

    Args:
        addr: Connection tuple containing the hostname and port
    """

    def __init__(self, addr):
        self.addr = addr
        self.conn = None
        self.awaitable = None
        self.process = None

    def require_connected(func):
        @functools.wraps(func)
        def _decorator(self, *args, **kwargs):
            if not self.conn:
                raise RuntimeError(
                    f"'BlenderClient' must be connected to a server instance before calling '{func.__name__}'"
                )
            return func(self, *args, **kwargs)

        return _decorator

    @classmethod
    @contextmanager
    def spawn(cls, timeout=10, log_dir=None, autoexec=False, executable=None):
        with BlenderServer.spawn(jobs=1, timeout=timeout, log_dir=log_dir, autoexec=autoexec, executable=executable) as (
            procs,
            conns,
        ):
            with cls(conns.pop()) as client:
                client.process = procs[0]
                yield client
                client.process = None

    @require_connected
    def render_animation_async(self, *args, **kwargs):
        render_animation_async = rpyc.async_(self.conn.root.render_animation)
        async_result = render_animation_async(*args, **kwargs)
        self.awaitable = async_result
        async_result.add_callback(lambda _: setattr(self, "awaitables", None))
        return async_result

    @require_connected
    def render_frames_async(self, *args, **kwargs):
        render_frames_async = rpyc.async_(self.conn.root.render_frames)
        async_result = render_frames_async(*args, **kwargs)
        self.awaitable = async_result
        async_result.add_callback(lambda _: setattr(self, "awaitables", None))
        return async_result

    def wait(self):
        if self.awaitable:
            self.awaitable.wait()

    def __enter__(self):
        self.conn = rpyc.connect(*self.addr, config={"sync_request_timeout": -1, "allow_pickle": True})

        for method_name in dir(BlenderService):
            if method_name.startswith("exposed_"):
                name = method_name.replace("exposed_", "")
                method = getattr(self.conn.root, method_name)
                setattr(self, name, method)
        return self

    def __exit__(self, type, value, traceback):
        self.conn.close()


class BlenderClients(tuple):
    """Collection of `BlenderClient` instances.

    Most methods in this class simply call the equivalent method of each client, that is,
    calling `clients.set_resolution` is equivalent to calling `set_resolution` for each
    client in clients. Some special methods, namely the `render_frames*` methods will
    instead distribute the rendering load to all clients. Finally, entering each client's
    context-manager, and closing each client connection (and by extension killing the
    server process if they were spawned) is ensured by using this class' context-manager.
    """

    def __new__(cls, *objs):
        objs = [BlenderClient(o) if isinstance(o, tuple) else o for o in objs]
        if not all(isinstance(o, BlenderClient) for o in objs):
            raise TypeError("'BlenderClients' can only contain 'BlenderClient' instances or their hostnames and ports.")
        return super().__new__(cls, objs)

    def __init__(self, *objs):
        # Note: At this point the tuple is already
        #       initialized, i.e: objs == list(self)
        for method_name in dir(BlenderService):
            if method_name.startswith("exposed_"):
                name = method_name.replace("exposed_", "")

                if name not in dir(self):
                    method = getattr(BlenderService, method_name)
                    multicall = self._method_dispatch_factory(name, method)
                    setattr(self, name, multicall)
        self.stack = ExitStack()

    def _method_dispatch_factory(self, name, method):
        @functools.wraps(method)
        def inner(*args, **kwargs):
            return tuple(getattr(client, name)(*args, **kwargs) for client in self)

        return inner

    def require_all_connected(func):
        @functools.wraps(func)
        def _decorator(self, *args, **kwargs):
            if not all(c.conn for c in self):
                raise RuntimeError(
                    f"All client instances in 'BlenderClients' must be connected before calling '{func.__name__}'"
                )
            return func(self, *args, **kwargs)

        return _decorator

    @classmethod
    @contextmanager
    def spawn(cls, jobs=1, timeout=10, log_dir=None, autoexec=False, executable=None):
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
    def pool(jobs=1, timeout=10, log_dir=None, autoexec=False, executable=None):
        """Spawns a multiprocessing-like worker pool, each with their own `BlenderClient` instance.
        The first argument to the function supplied to pool.map/imap/starmap and their async variants will
        be automagically passed a client instance as their first argument that they can use for rendering.

        Example Usage:

            def render(client, blend_file):
                root = Path("renders") / Path(blend_file).stem
                client.initialize(blend_file, root)
                client.render_animation()

            if __name__ == "__main__":
                with BlenderClients.pool(2) as pool:
                    pool.map(render, ["monkey.blend", "cube.blend", "metaballs.blend"])

        Note: Here we use `multiprocess` instead of the builtin multiprocessing library to take
            advantage of the more advanced dill serialization (as opposed to the standard pickling).

        Args:
            Same as `BlenderServer.spawn`

        Returns:
            A `multiprocess.Pool` instance which has had it's applicator methods (map/imap/starmap/etc)
            monkey-patched to inject a client instance as first argument.
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

                with BlenderClient(conn) as client:
                    retval = func(client, *args, **kwargs)

                conns.put(conn)
                return retval

            return inner

        def modify_applicator(applicator, conns):
            @functools.wraps(applicator)
            def inner(func, *args, **kwargs):
                func = inject_client(func, conns)
                return applicator(func, *args, **kwargs)

            return inner

        with multiprocess.Manager() as manager:
            with BlenderServer.spawn(
                jobs=jobs, timeout=timeout, log_dir=log_dir, autoexec=autoexec, executable=executable
            ) as (_, conns):
                q = manager.Queue()

                for conn in conns:
                    q.put(conn)

                with multiprocess.Pool(jobs) as pool:
                    for name, method in inspect.getmembers(pool, predicate=inspect.ismethod):
                        params = list(inspect.signature(method).parameters.keys())

                        # Get all map/starmap/apply/etc variants
                        if not name.startswith("_") and next(iter(params), None) == "func":
                            setattr(pool, name, modify_applicator(method, q))
                    yield pool

    @require_all_connected
    def common_animation_range(self):
        if len(ranges := set(self.animation_range_tuple())) != 1:
            raise RuntimeError("Found different animation ranges. All connected servers should be in the same state.")
        return range(*ranges.pop())

    @require_all_connected
    def render_frames(self, frame_numbers, allow_skips=True, dry_run=False, update_fn=None):
        # Set total number of steps, disable updates of total from child processes
        def ignore_total(*args, total=None, **kwargs):
            if update_fn is not None:
                return update_fn(*args, **kwargs)

        if update_fn is not None:
            update_fn(total=len(frame_numbers))

        # Equivalent to more-itertools' distribute
        children = itertools.tee(frame_numbers, len(self))
        frame_chunks = [itertools.islice(it, index, None, len(self)) for index, it in enumerate(children)]

        transforms = [
            client.render_frames_async(frames, allow_skips=allow_skips, dry_run=dry_run, update_fn=ignore_total)
            for client, frames in zip(self, frame_chunks)
        ]
        self.wait()

        # Equivalent to more-itertools' interleave_longest
        _marker = object()
        frames = [
            frame
            for frame in itertools.chain.from_iterable(
                itertools.zip_longest(*[t.value["frames"] for t in transforms], fillvalue=_marker)
            )
            if frame is not _marker
        ]
        transforms = transforms.pop().value
        transforms["frames"] = frames
        return transforms

    @require_all_connected
    def render_animation(
        self, frame_start=None, frame_end=None, frame_step=None, allow_skips=True, dry_run=False, update_fn=None
    ):
        if len(ranges := set(self.animation_range_tuple())) != 1:
            raise RuntimeError("Found different animation ranges. All connected servers should be in the same state.")

        start, end, step = ranges.pop()
        frame_start = start if frame_start is None else frame_start
        frame_end = end if frame_end is None else frame_end
        frame_step = step if frame_step is None else frame_step
        frame_range = range(frame_start, frame_end + 1, frame_step)

        return self.render_frames(frame_range, allow_skips=allow_skips, dry_run=dry_run, update_fn=update_fn)

    def wait(self):
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

    def __enter__(self):
        self.stack.__enter__()
        for client in self:
            self.stack.enter_context(client)
        return self

    def __exit__(self, type, value, traceback):
        self.stack.__exit__(type, value, traceback)


class BlenderService(rpyc.Service):
    def __init__(self):
        if bpy is None:
            raise RuntimeError(f"{type(self).__name__} needs to be instantiated from within blender's python runtime.")
        self.render_update_fn = None
        self.initialized = False
        self._conn = None

    def require_initialized(func):
        @functools.wraps(func)
        def _decorator(self, *args, **kwargs):
            if not self.initialized:
                raise RuntimeError(f"'BlenderService' must be initialized before calling '{func.__name__}'")
            return func(self, *args, **kwargs)

        return _decorator

    def validate_camera_moved(func):
        @functools.wraps(func)
        def _decorator(self, *args, **kwargs):
            prev_matrix = np.array(self.camera.matrix_world.copy())
            retval = func(self, *args, **kwargs)
            post_matrix = np.array(self.camera.matrix_world.copy())

            if np.allclose(prev_matrix, post_matrix):
                raise RuntimeError("Camera has not moved as intended, perhaps it is still bound by parent or animation?")
            return retval

        return _decorator

    def on_connect(self, conn):
        # TODO: Proper logging
        print(f"INFO: Successfully connected to BlenderClient instance.")
        self._conn = conn

    def on_disconnect(self, conn):
        # De-initialize service by restoring blender to it's startup state,
        # ensuring we clear any cached attrs (otherwise objects will be stale),
        # and resetting any instance variables we previously initialized.
        bpy.ops.wm.read_factory_settings()
        self.clear_cached_properties()
        self.initialized = False
        self._conn = None
        print(f"INFO: Successfully disconnected from BlenderClient instance.")

    def clear_cached_properties(self):
        # Based on: https://stackoverflow.com/a/71579485
        for name in dir(type(self)):
            if isinstance(getattr(type(self), name), functools.cached_property):
                vars(self).pop(name, None)

    def exposed_initialize(self, blend_file, root_path):
        if self.initialized:
            self.on_disconnect(None)

        # Load blendfile
        self.blend_file = blend_file
        bpy.ops.wm.open_mainfile(filepath=str(blend_file))
        print(f"INFO: Successfully loaded {blend_file}")

        # Ensure root paths exist, set default vars
        self.root_path = Path(str(root_path)).resolve()
        self.root_path.mkdir(parents=True, exist_ok=True)
        (self.root_path / "frames").mkdir(parents=True, exist_ok=True)

        # Init various variables to track state
        self.depth_path = None
        self.normal_path = None
        self.flow_path = None
        self.segmentation_path = None
        self.initialized = True
        self.unbind_camera = False
        self.use_animation = True
        self.pre_render_callbacks = []
        self.post_render_callbacks = []

        # Ensure we are using the compositor, and node tree.
        self.scene.use_nodes = True
        self.scene.render.use_compositing = True

        # Set default render settings
        self.scene.render.use_persistent_data = True
        self.scene.render.film_transparent = False

        # Warn if extra file output pipelines are found
        for n in getattr(self.tree, "nodes", []):
            if isinstance(n, bpy.types.CompositorNodeOutputFile):
                print(f"WARNING: Found output node {n}")

        # Catalogue any animations that are already disabled, otherwise
        # disabling and re-enabling animations would enable them.
        self.disabled_fcurves = set(
            [fcurve for action in bpy.data.actions for fcurve in (action.fcurves or []) if fcurve.mute]
        )

    @property
    @require_initialized
    def scene(self):
        # Ensures self.scene is always fresh
        return bpy.context.scene

    @property
    @require_initialized
    def tree(self):
        # Ensures self.tree is always fresh
        return self.scene.node_tree

    @functools.cached_property
    @require_initialized
    def render_layers(self):
        for node in self.tree.nodes:
            if node.type == "R_LAYERS":
                return node
        return self.tree.nodes.new("CompositorNodeRLayers")

    @property
    @require_initialized
    def view_layer(self):
        # Ensures self.view_layer is always fresh
        if not bpy.context.view_layer:
            raise ValueError("Expected at least one view layer, cannot render without it. Please add one manually.")
        return bpy.context.view_layer

    @functools.cached_property
    @require_initialized
    def camera(self):
        # Make sure there's a camera
        cameras = [ob for ob in self.scene.objects if ob.type == "CAMERA"]
        if not cameras:
            raise RuntimeError("No camera found, please add one manually.")
        elif len(cameras) > 1 and self.scene.camera:
            print(f"Multiple cameras found. Using active camera named: '{self.scene.camera}'.")
            return self.scene.camera
        else:
            print(f"No active camera was found. Using camera named: '{cameras[0]}'.")
            return cameras[0]

    @require_initialized
    def prerender_normal_pass_update(self, normal_group):
        """Updates the default values corresponding to the camera pose in the normals node group.
        This needs to be called before the render pass if normals are enabled."""
        mat = np.linalg.inv(self.camera.matrix_world)

        # Note: Blender's normals node actually returns the negative dot product (which may be a bug)
        # so we negate that here. See: https://projects.blender.org/blender/blender/issues/132770
        # TODO: Negation of dot product might become version specific once the bug is fixed.
        normal_group.node_tree.nodes["RotRow1"].outputs[0].default_value = -mat[0, :-1]
        normal_group.node_tree.nodes["RotRow2"].outputs[0].default_value = -mat[1, :-1]
        normal_group.node_tree.nodes["RotRow3"].outputs[0].default_value = -mat[2, :-1]

    @require_initialized
    def exposed_empty_transforms(self):
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
        transforms["c"] = 3
        transforms["w"] = self.scene.render.resolution_x
        transforms["h"] = self.scene.render.resolution_y
        transforms["fl_x"] = float(1 / 2 * self.scene.render.resolution_x / np.tan(1 / 2 * self.camera.data.angle))
        transforms["fl_y"] = float(1 / 2 * self.scene.render.resolution_y / np.tan(1 / 2 * self.camera.data.angle))
        transforms["cx"] = 1 / 2 * self.scene.render.resolution_x + transforms["shift_x"]
        transforms["cy"] = 1 / 2 * self.scene.render.resolution_y + transforms["shift_y"]
        transforms["intrinsics"] = self.exposed_camera_intrinsics().tolist()
        return transforms

    @require_initialized
    def get_parents(self, obj):
        """Recursively retrieves parent objects of a given object in Blender

        Args:
            obj: Object to find parent of.

        :return:
            List of parent objects of obj.
        """
        if getattr(obj, "parent", None):
            return [obj.parent] + self.get_parents(obj.parent)
        return []

    @require_initialized
    def exposed_animation_range(self):
        return range(self.scene.frame_start, self.scene.frame_end + 1, self.scene.frame_step)

    @require_initialized
    def exposed_animation_range_tuple(self):
        return (self.scene.frame_start, self.scene.frame_end, self.scene.frame_step)

    @require_initialized
    def exposed_include_depths(self, debug=True):
        """Sets up Blender compositor to include depth map in rendered images."""
        self.view_layer.use_pass_z = True
        (self.root_path / "depths").mkdir(parents=True, exist_ok=True)

        if debug:
            debug_depth_path = self.tree.nodes.new(type="CompositorNodeOutputFile")
            debug_depth_path.label = "Debug Depth Output"
            debug_depth_path.base_path = str(self.root_path / "depths")
            debug_depth_path.file_slots[0].path = f"debug_depth_{'#'*6}"

            normalize = self.tree.nodes.new("CompositorNodeNormalize")
            self.tree.links.new(self.render_layers.outputs["Depth"], normalize.inputs[0])
            self.tree.links.new(normalize.outputs[0], debug_depth_path.inputs[0])

        self.depth_path = self.tree.nodes.new(type="CompositorNodeOutputFile")
        self.depth_path.label = "Depth Output"
        self.depth_path.format.file_format = "OPEN_EXR"
        self.tree.links.new(self.render_layers.outputs["Depth"], self.depth_path.inputs[0])
        self.depth_path.base_path = str(self.root_path / "depths")
        self.depth_path.file_slots[0].path = f"depth_{'#'*6}"

    @require_initialized
    def exposed_include_normals(self, debug=True):
        """Sets up Blender compositer to include normal map in rendered images."""
        self.view_layer.use_pass_normal = True
        (self.root_path / "normals").mkdir(parents=True, exist_ok=True)

        # The node group `normaldebug` transforms normals from the global
        # coordinate frame to the camera's, and also colors normals as RGB
        from nodes import normaldebug

        normal_group = self.tree.nodes.new("CompositorNodeGroup")
        normal_group.label = "NormalDebug"
        normal_group.node_tree = normaldebug
        self.tree.links.new(self.render_layers.outputs["Normal"], normal_group.inputs[0])

        if debug:
            debug_normal_path = self.tree.nodes.new(type="CompositorNodeOutputFile")
            debug_normal_path.base_path = str(self.root_path / "normals")
            debug_normal_path.label = "Normals Debug Output"
            debug_normal_path.file_slots[0].path = f"debug_normal_{'#'*6}"

            # Important! Set the view settimgs to raw otherwise result is tonemapped
            debug_normal_path.format.color_management = "OVERRIDE"
            debug_normal_path.format.view_settings.view_transform = "Raw"

            self.tree.links.new(normal_group.outputs["RGBA"], debug_normal_path.inputs[0])

        self.normal_path = self.tree.nodes.new(type="CompositorNodeOutputFile")
        self.normal_path.label = "Normal Output"
        self.normal_path.format.file_format = "OPEN_EXR"
        self.tree.links.new(normal_group.outputs["Vector"], self.normal_path.inputs[0])
        self.normal_path.base_path = str(self.root_path / "normals")
        self.normal_path.file_slots[0].path = f"normal_{'#'*6}"
        self.pre_render_callbacks.append(functools.partial(self.prerender_normal_pass_update, normal_group))

    @require_initialized
    def exposed_include_flows(self, direction="fwd", debug=True):
        """Sets up Blender compositor to include optical flow in rendered images.

        Args:
            direction: One of 'forward', 'backward' or 'both'. Direction of flow, only used
                when debug is true, otherwise both forward and backward flows are saved.
            debug: If true, also save debug visualizations of flow.
        """
        if direction.lower() not in ("forward", "backward", "both"):
            raise ValueError(f"Direction argument should be one of forward, backward or both, got {direction}.")
        if self.scene.render.use_motion_blur:
            raise RuntimeError("Cannot compute optical flow if motion blur is enabled.")

        self.view_layer.use_pass_vector = True
        (self.root_path / "flows").mkdir(parents=True, exist_ok=True)

        if debug:
            from nodes import flowdebug

            # Seperate forward and backward flows (with a seperate color not vector node)
            split_flow = self.tree.nodes.new(type="CompositorNodeSeparateColor")
            self.tree.links.new(self.render_layers.outputs["Vector"], split_flow.inputs["Image"])

            # Create output node
            debug_flow_path = self.tree.nodes.new(type="CompositorNodeOutputFile")
            debug_flow_path.base_path = str(self.root_path / "flows")
            debug_flow_path.label = "Flow Debug Output"
            debug_flow_path.file_slots.clear()

            # Instantiate flow debug node group(s) and connect them
            if direction.lower() in ("forward", "both"):
                flow_group = self.tree.nodes.new("CompositorNodeGroup")
                flow_group.label = "Forward FlowDebug"
                flow_group.node_tree = flowdebug

                self.tree.links.new(split_flow.outputs["Red"], flow_group.inputs["x"])
                self.tree.links.new(split_flow.outputs["Green"], flow_group.inputs["y"])

                slot = debug_flow_path.file_slots.new(f"debug_fwd_flow_{'#'*6}")
                self.tree.links.new(flow_group.outputs["Image"], slot)
            if direction.lower() in ("backward", "both"):
                flow_group = self.tree.nodes.new("CompositorNodeGroup")
                flow_group.label = "Backward FlowDebug"
                flow_group.node_tree = flowdebug

                self.tree.links.new(split_flow.outputs["Blue"], flow_group.inputs["x"])
                self.tree.links.new(split_flow.outputs["Alpha"], flow_group.inputs["y"])

                slot = debug_flow_path.file_slots.new(f"debug_bwd_flow_{'#'*6}")
                self.tree.links.new(flow_group.outputs["Image"], slot)

        # Save flows as EXRs
        self.flow_path = self.tree.nodes.new(type="CompositorNodeOutputFile")
        self.flow_path.label = "Flow Debug Output"
        self.flow_path.format.file_format = "OPEN_EXR"
        self.tree.links.new(self.render_layers.outputs["Vector"], self.flow_path.inputs["Image"])
        self.flow_path.base_path = str(self.root_path / "flows")
        self.flow_path.file_slots[0].path = f"flow_{'#'*6}"

    @require_initialized
    def exposed_include_segmentations(self, shuffle=True, debug=True):
        """Sets up Blender compositor to include segmentation maps in rendered images.
        
        The debug visualization simply assigns a color to each object ID by mapping the 
        objects ID value to a hue using a HSV node with saturation=1 and value=1 (except 
        for the background which will have a value of 0 to ensure it is black).
        """
        # TODO: Enable assignement of custom IDs for certain objects via a dictionary. 

        if self.scene.render.engine.upper() != "CYCLES":
            raise RuntimeError("Cannot produce segmentation maps when not using CYCLES.")
        
        self.view_layer.use_pass_object_index = True
        (self.root_path / "segmentations").mkdir(parents=True, exist_ok=True)

        # Assign IDs to every object (background will be 0)
        indices = np.arange(len(bpy.data.objects))

        if shuffle:
            np.random.shuffle(indices)

        for i, obj in zip(indices, bpy.data.objects): 
            obj.pass_index = i+1

        if debug:
            from nodes import segmentationdebug

            seg_group = self.tree.nodes.new("CompositorNodeGroup")
            seg_group.label = "SegmentationDebug"
            seg_group.node_tree = segmentationdebug
            seg_group.node_tree.nodes["NormalizeIdx"].inputs["From Max"].default_value = len(bpy.data.objects)

            debug_seg_path = self.tree.nodes.new(type="CompositorNodeOutputFile")
            debug_seg_path.base_path = str(self.root_path / "segmentations")
            debug_seg_path.label = "Segmentations Debug Output"
            debug_seg_path.file_slots[0].path = f"debug_segmentation_{'#'*6}"

            self.tree.links.new(self.render_layers.outputs["IndexOB"], seg_group.inputs["Value"])
            self.tree.links.new(seg_group.outputs["Image"], debug_seg_path.inputs[0])
        
        self.segmentation_path = self.tree.nodes.new(type="CompositorNodeOutputFile")
        self.segmentation_path.label = "Segmentation Output"
        self.segmentation_path.format.file_format = "OPEN_EXR"
        self.tree.links.new(self.render_layers.outputs["IndexOB"], self.segmentation_path.inputs[0])
        self.segmentation_path.base_path = str(self.root_path / "segmentations")
        self.segmentation_path.file_slots[0].path = f"segmentation_{'#'*6}"        

    @require_initialized
    def exposed_load_addons(self, *addons):
        """Load blender addons by name (case-insensitive)"""
        for addon in addons:
            addon = addon.strip().lower()
            addon_module = addon_utils.enable(addon, default_set=True)
            print(f"INFO: Loaded addon {addon}: {addon_module}")

    @require_initialized
    def exposed_set_resolution(self, height=None, width=None):
        """Set frame resolution (height, width) in pixels"""
        if isinstance(height, (tuple, list)):
            if width is not None:
                raise ValueError(
                    "Cannot understand desired resolution, either pass a (h, w) tuple, or use keyword arguments."
                )
            height, width = height

        if height:
            self.scene.render.resolution_y = height
        if width:
            self.scene.render.resolution_x = width
        self.scene.render.resolution_percentage = 100

    @require_initialized
    def exposed_image_settings(self, file_format="PNG", bitdepth=8):
        self.scene.render.image_settings.file_format = file_format.upper()
        self.scene.render.image_settings.color_depth = str(bitdepth)
        self.scene.render.image_settings.color_mode = "RGB"

    @require_initialized
    def exposed_use_motion_blur(self, enable):
        self.scene.render.use_motion_blur = enable

    @require_initialized
    def exposed_use_animations(self, enable):
        for action in bpy.data.actions:
            for fcurve in action.fcurves or []:
                if fcurve not in self.disabled_fcurves:
                    fcurve.mute = not enable
        self.use_animation = enable

    @require_initialized
    def exposed_cycles_settings(
        self,
        device_type=None,
        use_cpu=None,
        adaptive_threshold=None,
        use_denoising=None,
    ):
        """Enables/activates cycles render devices and settings.

        Note: A default arguments of `None` means do not change setting.

        Args:
            device_type: Name of device to use, one of "cpu", "cuda", "optix", "metal"...
            use_cpu: Boolean flag to enable CPUs alongside GPU devices.
            adaptive_threshold: Set noise threshold upon which to stop taking samples.
            use_denoising: If enabled, a denoising pass will be used.

        :return:
            List of activated devices.
        """
        if self.scene.render.engine.upper() != "CYCLES":
            print(
                f"WARNING: Using {self.scene.render.engine.upper()} rendering engine, "
                f"with default OpenGL rendering device(s)."
            )
            return []

        if use_denoising is not None:
            self.scene.cycles.use_denoising = use_denoising

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
                print("INFO: Activated device", device.name, device.type)
                activated_devices.append(device.name)
                device.use = True
            cycles_preferences.compute_device_type = "NONE" if device_type.upper() == "CPU" else device_type.upper()
            self.scene.cycles.device = "CPU" if device_type.upper() == "CPU" else "GPU"
            return activated_devices
        return []

    @require_initialized
    def exposed_unbind_camera(self):
        """Remove constraints, animations and parents from main camera.

        Note: In order to undo this, you'll need to re-initialize the instance.
        """
        for c in self.camera.constraints:
            self.camera.constraints.remove(c)
        self.camera.animation_data_clear()
        self.unbind_camera = True
        self.camera.parent = None

    @require_initialized
    def exposed_move_keyframes(self, scale=1.0, shift=0.0):
        """Adjusts keyframes in Blender animations, keypoints are first scaled then shifted.

        Args:
            scale: Factor used to rescale keyframe positions along x-axis. Defaults to 1.0.
            shift: Factor used to shift keyframe positions along x-axis. Defaults to 0.0.
        """
        # TODO: This method can be slow if there's a lot of keyframes
        #   See: https://blender.stackexchange.com/questions/111644
        if scale == 1.0 and shift == 0.0:
            return

        # No idea why, but if we don't break this out into separate
        # variables the value we store is incorrect, often off by one.
        start = round(self.scene.frame_start * scale + shift)
        end = round(self.scene.frame_end * scale + shift)

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

    @require_initialized
    def exposed_set_current_frame(self, frame_number):
        self.scene.frame_set(frame_number)

    @require_initialized
    def exposed_camera_extrinsics(self):
        pose = np.array(self.camera.matrix_world)
        pose[:3, :3] /= np.linalg.norm(pose[:3, :3], axis=0)
        return pose

    @require_initialized
    def exposed_camera_intrinsics(self):
        """Calculates camera intrinsics matrix for active camera in Blender,
        which defines how 3D points are projected onto 2D.

        :return:
            Camera intrinsics matrix based on camera properties.
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

    @require_initialized
    @validate_camera_moved
    def exposed_position_camera(self, location=None, rotation=None, look_at=None):
        """
        Positions and orients camera in Blender scene according to specified parameters.

        Note: Only one of `look_at` or `rotation` can be set at once.

        Args:
            location: Location to place camera in 3D space. Defaults to none.
            rotation: Rotation matrix for camera. Defaults to none.
            look_at: Location to point camera. Defaults to none.
        """
        # Move camera to position, orient it towards object
        if not (look_at is not None) ^ (rotation is not None):
            raise ValueError("Only one of `look_at` or `rotation` can be set.")

        if location is not None:
            self.camera.location = mathutils.Vector(location)

        if look_at is not None:
            # point the camera's '-Z' towards `look_at` and use its 'Y' as up
            direction = mathutils.Vector(look_at) - self.camera.location
            rot_quat = direction.to_track_quat("-Z", "Y")
            rotation = rot_quat.to_matrix()

        if rotation is not None:
            location = self.camera.location.copy()
            self.camera.matrix_world = mathutils.Matrix(rotation).to_4x4()
            self.camera.location = location
        self.view_layer.update()

    @require_initialized
    @validate_camera_moved
    def exposed_rotate_camera(self, angle):
        """
        Rotate camera around it's optical axis.

        Args:
            angle: Amount to rotate by (in radians).
        """
        location = self.camera.location.copy()
        right, up, _ = self.camera.matrix_world.to_3x3().transposed()
        look_at_direction = mathutils.Vector(np.cross(up, right))
        rotation = mathutils.Matrix.Rotation(angle, 3, look_at_direction)
        rotation = rotation @ self.camera.matrix_world.to_3x3()
        self.camera.matrix_world = rotation.to_4x4()
        self.camera.location = location
        self.view_layer.update()

    @require_initialized
    def exposed_render_current_frame(self, allow_skips=True, dry_run=False):
        """
        Generates a single frame in Blender at the current camera location,
        return the file paths for that frame, potentially including depth and normals.

        Args:
            allow_skips: if true, blender will not re-render and overwrite existing frames.
                This does not however apply to depth/normals, which cannot be skipped.
            dry_run: if true, nothing will be rendered at all.

        :return:
            dictionary containing paths to rendered frames for this index and camera pose.
        """
        # TODO: Implement skipping for depth/normals/flow?

        # Assumes the camera position, frame number and all other params have been set
        index = self.scene.frame_current
        self.scene.render.filepath = str(self.root_path / "frames" / f"frame_{index:06}")
        paths = {"file_path": Path(f"frames/frame_{index:06}").with_suffix(self.scene.render.file_extension)}

        if self.depth_path is not None:
            paths["depth_file_path"] = Path(f"depths/depth_{index:06}.exr")
        if self.normal_path is not None:
            paths["normal_file_path"] = Path(f"normals/normal_{index:06}.exr")
        if self.flow_path is not None:
            paths["flow_file_path"] = Path(f"flows/flow_{index:06}.exr")
        if self.segmentation_path is not None:
            paths["segmentation_file_path"] = Path(f"segmentations/segmentation_{index:06}.exr")

        for callback in self.pre_render_callbacks:
            callback()

        if not dry_run:
            # Render frame(s), skip the render iff all files exist and `allow_skips`
            if not allow_skips or any(not Path(self.root_path / p).exists() for p in paths.values()):
                # If `write_still` is false, depth & normals can be written but rgb will be skipped
                skip_frame = Path(self.root_path / paths["file_path"]).exists() and allow_skips
                bpy.ops.render.render(write_still=not skip_frame)

        for callback in self.post_render_callbacks:
            callback()

        # Returns paths that were written
        return {
            **{k: str(p) for k, p in paths.items()},
            "transform_matrix": self.exposed_camera_extrinsics().tolist(),
        }

    @require_initialized
    def exposed_render_frame(self, frame_number, allow_skips=True, dry_run=False):
        """Same as first setting current frame then rendering it."""
        self.exposed_set_current_frame(frame_number)
        return self.exposed_render_current_frame(allow_skips=allow_skips, dry_run=dry_run)

    @require_initialized
    def exposed_render_frames(self, frame_numbers, allow_skips=True, dry_run=False, update_fn=None):
        """Render all requested frames and return associated transforms dictionary"""
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
            print(
                f"WARNING: Current animation starts at frame #{self.scene.frame_start} and ends at "
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
                update_fn()

        # Restore animation range to original values
        self.scene.frame_start, self.scene.frame_end = scene_original_range
        return transforms

    @require_initialized
    def exposed_render_animation(
        self, frame_start=None, frame_end=None, frame_step=None, allow_skips=True, dry_run=False, update_fn=None
    ):
        """
        This is the core frame generation process. Determines frame range to render,
        sets camera positions and orientations,
        and renders all frames.

        Note: All frame start/end/step arguments are absolute quantities, applied after any keyframe moves.
              If the animation is from (1-100) and you've scaled it by calling `move_keyframes(scale=2.0)`
              then calling `render_animation(frame_start=1, frame_end=100)` will only render half of the animation.
              By default the whole animation will render when no start/end and step values are set.

        Args:
            frame_start: Starting index (inclusive) of frames to render as seen in blender. Defaults to value from `.blend` file.
            frame_end: Ending index (inclusive) of frames to render as seen in blender. Defaults to value from `.blend` file.
            frame_step: Skip every nth frame. Defaults to value from `.blend` file.
            allow_skips: Same as `(exposed_)render_frame_here`.
            dry_run: Same as `(exposed_)render_frame_here`.
            update_fn: Callback which expects no arguments, that will get called after each frame has rendered. Useful for tracking progress.
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
            print("WARNING: Active camera nor it's parents are animated, camera will be static.")

        return self.exposed_render_frames(frame_range, allow_skips=allow_skips, dry_run=dry_run, update_fn=update_fn)

    @require_initialized
    def exposed_save_file(self, path):
        """Save opened blender file. This is useful for introspecting the state of the compositor/scene/etc."""
        bpy.ops.wm.save_as_mainfile(filepath=str(path))


# This script has only been tested using Blender 3.3.1 (hash b292cfe5a936 built 2022-10-05 00:14:35) and above.
if __name__ == "__main__":
    if sys.version_info < (3, 9, 0):
        raise RuntimeError("Please use newer blender version with a python version of at least 3.9.")

    server = BlenderServer(port=0)
    server.start()
