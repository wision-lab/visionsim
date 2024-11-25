import argparse
import ast
import functools
import importlib
import itertools
import json
import os
import shlex
import signal
import site
import subprocess
import sys
import time
from contextlib import ExitStack
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

    import rpyc
    import rpyc.utils.registry
    from rich.progress import track
    from scipy.interpolate import CubicSpline, interp1d
except ImportError:
    # Note: the same can be done for np, mathutils, etc but these
    # should already be installed by default.
    print(
        "\nScipy and Rich are needed to run this script. To install it so that "
        "it is accessible from blender, you need to pip install it "
        "into blender's python interpreter like so:\n"
    )
    print(f"${sys.executable} -m ensurepip")
    print(f"${sys.executable} -m pip install scipy rich rpyc --target {site.USER_SITE}")
    sys.exit()

usage = (
    "This script needs to be called via blender like so: \n"
    "$ blender scene.blend --background --python render.py -- <script arguments>\n\n"
    "Or by using the spsim cli: `spsim -h blender.render`"
)

REGISTRY = None


class LogRedirect:
    """Capture and redirect stdout and stderr of non-python program to log files

    We use this here as Blender logs everything to console, slowing it down and
    creating a lot of (usually) unnecessary noise.

    Args:
        root_path = root path to create log files for stdout and stderr.
    """

    # Adapted from: https://stackoverflow.com/questions/66858529
    def __init__(self, root_path=None):
        if not root_path:
            self.logpath_out = os.devnull
            self.logpath_err = os.devnull
        else:
            self.root_path = Path(root_path).resolve().with_suffix("")
            self.logpath_out = str(self.root_path) + "_stdout.log"
            self.logpath_err = str(self.root_path) + "_stderr.log"
            self.root_path.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout.flush()
        sys.stderr.flush()

    def __enter__(self):
        self.logfile_out = os.open(str(self.logpath_out), os.O_WRONLY | os.O_TRUNC | os.O_CREAT)
        self.logfile_err = os.open(str(self.logpath_err), os.O_WRONLY | os.O_TRUNC | os.O_CREAT)
        self.orig_stdout = os.dup(1)
        self.orig_stderr = os.dup(2)
        self.new_stdout = os.dup(1)
        self.new_stderr = os.dup(2)
        os.dup2(self.logfile_out, 1)
        os.dup2(self.logfile_err, 2)
        sys.stdout = os.fdopen(self.new_stdout, "w")
        sys.stderr = os.fdopen(self.new_stderr, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.flush()
        sys.stderr.flush()

        os.dup2(self.orig_stdout, 1)
        os.dup2(self.orig_stderr, 2)
        os.close(self.orig_stdout)
        os.close(self.orig_stderr)

        os.close(self.logfile_out)
        os.close(self.logfile_err)


class Spline:
    """

    Args:
        spline_points: Array of points that define spline curve. Expected in N x 3, where N is number of points.
        periodic: Boolean indicating whether spline should be made periodic. Defaults to true.
        unit_speed: Boolean indicating whether to re-parameterize for speed. Defaults to false.
        samples: Number of samples to use for spline. Defaults to 1000.
        kind: Cubic or linear spline. Determines type of spline interpolation Defaults to "cubic".
        **kwargs
    """

    def __init__(
        self,
        spline_points,
        periodic=True,
        unit_speed=False,
        samples=1000,
        kind="cubic",
        **kwargs,
    ):
        spline_points = np.array(spline_points) if not isinstance(spline_points, np.ndarray) else spline_points
        n, c, *_ = spline_points.shape

        if spline_points.ndim != 2 and c != 3:
            raise ValueError(f"Expected `spline_points` to have dimensions Nx3, instead got {spline_points.shape}.")

        if n == 1:
            # If only one point, it's static. Return spline that only maps to that point.
            t = np.array([0, 1])
            points = np.tile(spline_points, (2, 1))
            spline = interp1d(t, points, kind="linear", axis=0, fill_value="extrapolate")
        else:
            if periodic:
                # Make spline cyclical, i.e f(0)=f(1) and f'(0)=f'(1).
                t = np.linspace(-1, 2, n * 3, endpoint=False)
                points = np.tile(spline_points, (3, 1))
            else:
                t = np.linspace(0, 1, n)
                points = spline_points
            # RY/ use either cubic spline or linear spline from scipy
            if kind.lower() == "cubic":
                spline = CubicSpline(t, points, axis=0)
            elif kind.lower() == "linear":
                spline = interp1d(t, points, kind="linear", axis=0, fill_value="extrapolate")
            else:
                raise ValueError(f"Unrecognized value for `kind`. Expected either 'cubic' or 'linear', got '{kind}'.")

        self.samples = samples
        self.points = spline_points
        self.t = np.linspace(0, 1, n, endpoint=not periodic)
        self.unit_speed = unit_speed
        self.t_reparam = None
        self.unit_spline = None
        self.unit_spline_d = None
        self.unit_spline_dd = None
        self.spline = spline
        self.periodic = periodic
        self.kind = kind.lower()
        self.n = n

        if self.unit_speed:
            self._arclength_reparameterize()

    @staticmethod
    def smooth_spline(spline, t, samples=5, interval=0.01):
        """
        Create smooth version of provided spline.

        Args:
            spline: Spline curve to be smoothed.
            t: Time when spline is evaluated.
            samples: Number of samples used for smoothing. Defaults to 5.
            interval: Interval used for between samplings. Defaults to 0.01.
        """
        return np.mean(
            [spline(t + offset) for offset in (np.arange(samples) - samples // 2) * interval],
            axis=0,
        )

    def _arclength_reparameterize(self):
        """Perform arclength re-parameterization to achieve unit speed. Ensure equal distance over equal time intervals"""
        # Arc-length re-parameterization
        if self.unit_spline is None:
            if self.n != 1:
                # Sample densely to estimate running length
                t = np.linspace(0, 1, self.samples, endpoint=not self.periodic)
                dp = np.gradient(self.spline(t), axis=0)[1:]

                # Ensure length(t=0) is zero
                dl = np.concatenate([[0], np.sqrt((dp**2).sum(axis=1))])
                length = np.cumsum(dl)

                # Invert curve, l -> t and use the output as new parameter t
                self.t_reparam = interp1d(length, t, kind="linear", axis=0, fill_value="extrapolate")
                t_ = self.t_reparam(t * length.max())
                p_ = self.spline(t_)

                # Add in original spline points to ensure unit spline passes through them too
                t_reparam_inv = interp1d(t, length, kind="linear", axis=0, fill_value="extrapolate")
                t = np.append(t, t_reparam_inv(self.t) / length.max())
                p_ = np.vstack([p_, self.points])

                if self.periodic:
                    # Make spline cyclical, i.e f(0)=f(1) and f'(0)=f'(1).
                    t = np.concatenate([t - 1, t, t + 1])
                    p_ = np.tile(p_, (3, 1))

                # Ensure points are ordered with no duplicates
                t, idx = np.unique(t, return_index=True)
                p_ = p_[idx]
                idx = np.argsort(t)
                t, p_ = t[idx], p_[idx]

                if self.kind == "cubic":
                    self.unit_spline = CubicSpline(t, p_)
                    self.unit_spline_d = self.unit_spline.derivative(1)
                    self.unit_spline_dd = self.unit_spline.derivative(2)
                else:
                    self.unit_spline = interp1d(t, p_, kind="linear", axis=0, fill_value="extrapolate")
            else:
                self.unit_spline = self.spline

    def tnb_frame(self, t, smooth=False):
        """Compute the Frenet-Serret Frame given an *arclength* parameterized spline

        Args:
            t: Arclength parameter.
            smooth: Boolean indicating whether to smooth derivatives of of spline. Defaults to false.
        """
        if self.n > 2 and self.kind == "cubic":
            self._arclength_reparameterize()
            t = np.atleast_1d(t)

            if smooth:
                T = self.smooth_spline(self.unit_spline_d, t)
                N = self.smooth_spline(self.unit_spline_dd, t)
            else:
                T = self.unit_spline_d(t)
                N = self.unit_spline_dd(t)

            N = N / np.sqrt((N**2).sum(axis=1))[..., None]
            B = np.cross(T, N)
            return T, N, B
        raise RuntimeError("Cannot compute TNB frame for a linear spline or one with less than 3 points.")

    def show(self, ax=None, color=None, samples=1000, step=10):
        """
        Visualize spline curves in 3d.

        Args:
            ax:  Optional matplotlib axis to place plot in. Defaults to none.
            color: Color of plot. Defaults to none.
            samples: Number of samples to use for spline. Defaults to 1000
            step: Number of points to skip when plotting. Defaults to 10.
        """
        import matplotlib.pyplot as plt

        ax = plt.figure(figsize=(8, 8)).add_subplot(projection="3d") if ax is None else ax
        t = np.linspace(0, 1, samples)
        x, y, z = self(t).T

        if self.n == 1:
            ax.scatter(x, y, z)
        else:
            color = self.unit_speed if color is None else color
            if not color:
                ax.plot(x, y, z)
            else:
                for i in range(len(t) - 1):
                    if step and (i // step) % 2 == 0:
                        continue
                    ax.plot(
                        x[i : i + 2],
                        y[i : i + 2],
                        z[i : i + 2],
                        color=plt.cm.hsv(i / len(t)),
                    )
        return ax

    def show_tnb(self, t=0.5, length=1, **kwargs):
        """Visualizes Frenet Serret Frame t a specific t value using show method
        Args:
        t: T value to use along spline.
        length: Length of arrow.
        """
        ax = self.show(**kwargs)
        self._arclength_reparameterize()
        x, y, z = self.unit_spline(t)
        T, N, B = [i.squeeze() for i in self.tnb_frame(t)]
        ax.quiver(x, y, z, *T.squeeze(), length=length, normalize=True, color="r")
        ax.quiver(x, y, z, *N.squeeze(), length=length, normalize=True, color="g")
        ax.quiver(x, y, z, *B.squeeze(), length=length, normalize=True, color="b")
        return ax

    def __call__(self, t):
        if self.periodic:
            t, _ = np.modf(t)
        if self.unit_speed:
            self._arclength_reparameterize()
            return self.unit_spline(t)
        return self.spline(t)


class BlenderServer(rpyc.utils.server.OneShotServer):
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

    @staticmethod
    def spawn(jobs=1):
        # Blender has trouble spawning threads within itself, ..........
        cmd = shlex.split(f"blender -b --python {__file__}")
        return [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(jobs)]

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

    """

    def __init__(self, addr):
        self.addr = addr
        self.conn = None
        self.awaitable = None

    @classmethod
    def spawn(cls, timeout=10):
        BlenderServer.spawn_registry()
        existing = BlenderServer.discover()
        (worker,) = BlenderServer.spawn(jobs=1)
        start = time.time()

        while True:
            if (time.time() - start) > timeout:
                worker.terminate()
                raise TimeoutError("Unable to spawn and discover server in alloted time.")
            if len(conns := set(BlenderServer.discover()) - set(existing)) == 1:
                break
            time.sleep(0.1)

        return cls(conns.pop())

    def render_animation_async(self, *args, **kwargs):
        if not self.conn:
            raise RuntimeError(f"'BlenderClient' must be initialized before calling 'render_animation_async'")
        render_animation_async = rpyc.async_(self.conn.root.render_animation)
        async_result = render_animation_async(*args, **kwargs)
        self.awaitable = async_result
        async_result.add_callback(lambda _: setattr(self, "awaitables", None))
        return async_result

    def wait(self):
        if self.awaitable:
            self.awaitable.wait()

    def __enter__(self):
        self.conn = rpyc.connect(*self.addr, config={"sync_request_timeout": -1,  "allow_pickle": True})

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

    Most methods in this class simply call the equivalent method
    of each client, that is, calling `clients.set_resolution` is
    equivalent to calling `set_resolution` for each client in clients.
    Some special methods, namely the `render_frames*` methods will
    instead distribute the rendering load to all clients.
    Finally, entering each client's context-manager, and closing each
    client connection (and by extension killing the server process)
    is ensured by using this class' context-manager.
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
            if method_name.startswith("exposed_") and "render" not in method_name:
                name = method_name.replace("exposed_", "")
                method = getattr(BlenderService, method_name)
                multicall = self._method_dispatch_factory(name, method)
                setattr(self, name, multicall)
        self.stack = ExitStack()

    def _method_dispatch_factory(self, name, method):
        @functools.wraps(method)
        def inner(*args, **kwargs):
            for client in self:
                getattr(client, name)(*args, **kwargs)

        return inner
    
    @classmethod
    def spawn(cls, jobs=1, timeout=10):
        BlenderServer.spawn_registry()
        existing = BlenderServer.discover()
        workers = BlenderServer.spawn(jobs=jobs)
        start = time.time()

        while True:
            if (time.time() - start) > timeout:
                for worker in workers:
                    worker.terminate()
                raise TimeoutError("Unable to spawn and discover server in alloted time.")
            if len(conns := set(BlenderServer.discover()) - set(existing)) == jobs:
                break
            time.sleep(0.1)

        return cls(*conns)

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
        bpy.ops.wm.open_mainfile(filepath=blend_file)
        print(f"INFO: Successfully loaded {blend_file}")

        # Ensure root paths exist, set default vars
        self.root_path = Path(str(root_path)).resolve()
        self.root_path.mkdir(parents=True, exist_ok=True)
        (self.root_path / "frames").mkdir(parents=True, exist_ok=True)

        # Init various variables to track state
        self.depth_path, self.normals_path = None, None
        self.initialized = True
        self.unbind_camera = False
        self.use_animation = True

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
        self.disabled_fcurves = set([
            fcurve
            for action in bpy.data.actions
            for fcurve in (action.fcurves or [])
            if fcurve.mute
        ])

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
        transforms["fl_x"] = 1 / 2 * self.scene.render.resolution_x / np.tan(1 / 2 * self.camera.data.angle)
        transforms["fl_y"] = 1 / 2 * self.scene.render.resolution_y / np.tan(1 / 2 * self.camera.data.angle)
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
    def exposed_include_depths(self):
        """Sets up Blender compositor to include depth map in rendered images."""
        self.view_layer.use_pass_z = True
        (self.root_path / "depths").mkdir(parents=True, exist_ok=True)
        self.depth_path = self.tree.nodes.new(type="CompositorNodeOutputFile")
        self.depth_path.label = "Depth Output"
        self.depth_path.format.file_format = "OPEN_EXR"
        self.tree.links.new(self.render_layers.outputs["Depth"], self.depth_path.inputs[0])
        self.depth_path.base_path = str(self.root_path / "depths")
        self.depth_path.file_slots[0].path = f"depth_{'#'*6}"

    @require_initialized
    def exposed_include_normals(self):
        """Sets up Blender compositer to include normal map in rendered images."""
        self.view_layer.use_pass_normal = True
        (self.root_path / "normals").mkdir(parents=True, exist_ok=True)
        self.normals_path = self.tree.nodes.new(type="CompositorNodeOutputFile")
        self.normals_path.label = "Normal Output"
        self.tree.links.new(self.render_layers.outputs["Normal"], self.normals_path.inputs[0])
        self.normals_path.base_path = str(self.root_path / "normals")
        self.normals_path.file_slots[0].path = f"normal_{'#'*6}"

    @require_initialized
    def exposed_set_resolution(self, resolution):
        """Set frame resolution (height, width) in pixels"""
        self.scene.render.resolution_y, self.scene.render.resolution_x = resolution
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
        device_type="optix",
        use_cpu=False,
        adaptive_threshold=0.1,
        use_denoising=True,
    ):
        """Enables/activates cycles render devices and settings.

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

        self.scene.cycles.use_denoising = use_denoising
        self.scene.cycles.adaptive_threshold = adaptive_threshold

        preferences = bpy.context.preferences
        cycles_preferences = preferences.addons["cycles"].preferences
        cycles_preferences.refresh_devices()

        if not cycles_preferences.devices:
            raise RuntimeError("No devices found!")

        for device in cycles_preferences.devices:
            device.use = False

        activated_devices = []
        devices = filter(lambda d: d.type.upper() == device_type.upper(), cycles_preferences.devices)

        for device in itertools.chain(devices, filter(lambda d: d.type == "CPU" and use_cpu, devices)):
            print("INFO: Activated device", device.name, device.type)
            activated_devices.append(device.name)
            device.use = True
        cycles_preferences.compute_device_type = "NONE" if device_type.upper() == "CPU" else device_type
        self.scene.cycles.device = "CPU" if device_type.upper() == "CPU" else "GPU"
        return activated_devices

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
    def exposed_render_current_frame(self, allow_skips=True):
        """
        Generates a single frame in Blender at the current camera location,
        return the file paths for that frame, potentially including depth and normals.

        Args:
            allow_skips: if true, blender will not re-render and overwrite existing frames.
                This does not however apply to depth/normals, which cannot be skipped.

        :return:
            dictionary containing paths to rendered frames for this index and camera pose.
        """
        # Assumes the camera position, frame number and all other params have been set
        index = self.scene.frame_current
        self.scene.render.filepath = str(self.root_path / "frames" / f"frame_{index:06}")
        paths = {"file_path": Path(f"frames/frame_{index:06}").with_suffix(self.scene.render.file_extension)}

        if self.depth_path is not None:
            paths["depth_file_path"] = Path(f"depths/depth_{index:06}.exr")
        if self.normals_path is not None:
            paths["normals_file_path"] = Path(f"normals/normal_{index:06}").with_suffix(self.scene.render.file_extension)

        # Render frame(s), skip the render iff all files exist and `allow_skips`
        if not allow_skips or any(not Path(self.root_path / p).exists() for p in paths.values()):
            # If `write_still` is false, depth & normals can be written but rgb will be skipped
            skip_frame = Path(self.root_path / paths["file_path"]).exists() and allow_skips
            bpy.ops.render.render(write_still=not skip_frame)

        # Returns paths that were written
        return {
            **{k: str(p) for k, p in paths.items()},
            "transform_matrix": self.exposed_camera_extrinsics().tolist(),
        }
    
    @require_initialized
    def exposed_render_frame(self, frame_number, allow_skips=True):
        """Same as first setting current frame then rendering it."""
        self.exposed_set_current_frame(frame_number)
        return self.exposed_render_current_frame(allow_skips=allow_skips)
    
    @require_initialized
    def exposed_render_frames(self, frame_numbers, allow_skips=True, update_fn=None):
        """Render all requested frames and return associated transforms dictionary"""
        # Ensure frame_numbers is a list to find extrema
        frame_numbers = list(frame_numbers)

        # Max number of frames is 1,048,574 as of Blender 3.4
        # See: https://docs.blender.org/manual/en/latest/advanced/limits.html
        if max(frame_numbers) >= 1_048_574:
            raise RuntimeError(
                f"Blender cannot currently render more than 1,048,574 frames, yet requested you "
                f"requested {max(frame_numbers)} frames to be rendered. For more please see: "
                f"https://docs.blender.org/manual/en/latest/advanced/limits.html"
            )
        if min(frame_numbers) < 0:
            raise RuntimeError("Cannot render frames at negative indices. You can try shifting keyframes.")

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
            frame_data = self.exposed_render_frame(frame_number, allow_skips=allow_skips)
            transforms["frames"].append(frame_data)

            # Call any progress callbacks
            if update_fn is not None:
                update_fn()

        # Restore animation range to original values
        self.scene.frame_start, self.scene.frame_end = scene_original_range
        return transforms

    @require_initialized
    def exposed_render_animation(
        self, frame_start=None, frame_end=None, frame_step=None, allow_skips=True, update_fn=None
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
            update_fn: Callback which expects no arguments, that will get called after each frame has rendered. Useful for tracking progress.
        """
        if not self.use_animation:
            raise ValueError(
                "Animations are disabled, scene will be entirely static. "
                "To instead render a single frame, use `render_frame`."
            )
        elif all(p.animation_data is None for p in self.get_parents(self.camera)) and self.camera.animation_data is None:
            print("WARNING: Active camera nor it's parents are animated, camera will be static.")

        frame_start = self.scene.frame_start if frame_start is None else frame_start
        frame_end = self.scene.frame_end if frame_end is None else frame_end
        frame_step = self.scene.frame_step if frame_step is None else frame_step
        frame_range = range(frame_start, frame_end + 1, frame_step)

        # Warn if requested frames lie outside animation range
        if self.use_animation and (frame_start < self.scene.frame_start or self.scene.frame_end < frame_end):
            print(
                f"WARNING: Current animation starts at frame #{self.scene.frame_start} and ends at "
                f"#{self.scene.frame_end} (with step={self.scene.frame_step}), but you requested frames "
                f"#{frame_start} to #{frame_end} (step={frame_step}) to be rendered.\n"
            )

        return self.exposed_render_frames(frame_range, allow_skips=allow_skips, update_fn=update_fn)

    @require_initialized
    def exposed_save_file(self, path):
        """Save opened blender file. This is useful for introspecting the state of the compositor/scene/etc."""
        bpy.ops.wm.save_as_mainfile(filepath=path)


class _BlenderDatasetGenerator:
    """Generate a dataset of different views of a blend file.

    Args:
        root_path: Root where dataset will be saved.
        blend_file: Optional Blender file to use. Defaults to none and assumes file is already open.
        resolution: Width and height of frames. Defaults to (800, 800).
        depth: Boolean flag to enable depth. Defaults to false.
        normals: Boolean flag to enable normals. Defaults to false.
        file_format: File format of images to be saved. defaults to PNG.
        color_depth: Bits per channel for color depth. Defaults to 8.
        device: Device to use to render. Defaults to "optix".
        render: Boolean flag to render. Defaults to true.
        unbind_camera: Boolean flag specifying whether camera should be unbound and moved explicitly. Defaults to true.
        device_idx: Specifying subset of devices when rendering with multiple GPUS. Defaults to all.
        allow_skips: Boolean flag indicating whether to skip rendering frames that already exist. Defaults to true.
        use_animation: Boolean flag indicating whether animations should beu sed for camera movement. Defaults to true.
        keyframe_multiplier: Scales animation speed. Defaults to 1.0.
        use_motion_blur: Controls motion blur in rendered frames. Defaults to true.
        alpha_color: Color for background alpha blending. Defaults to none.
    """

    def __init__(
        self,
        root_path,
        blend_file=None,
        resolution=(800, 800),
        depth=False,
        normals=False,
        file_format="PNG",
        color_depth=8,
        device="optix",
        render=True,
        unbind_camera=True,
        device_idxs=slice(None),
        allow_skips=True,
        use_animation=True,
        keyframe_multiplier=1.0,
        use_motion_blur=True,
        alpha_color=None,
        adaptive_threshold=0.1,
        **kwargs,
    ):
        # Load blender file
        if not bpy.data.filepath:
            if not blend_file:
                raise ValueError("No blender file was specified!")
            bpy.ops.wm.open_mainfile(filepath=blend_file)
            print(f"INFO: Successfully loaded {blend_file}")
        elif blend_file:
            raise ValueError("A blender file already opened, cannot also specify `blend_file` param.")

        self.root_path = Path(root_path).resolve()
        self.root_path.mkdir(parents=True, exist_ok=True)
        (self.root_path / "frames").mkdir(parents=True, exist_ok=True)
        self.height, self.width = resolution
        self.use_animation = use_animation
        self.unbind_camera = unbind_camera
        self.file_format = file_format
        self.allow_skips = allow_skips
        self.alpha_color = alpha_color
        self.render = render

        # Ensure we are using the compositor, and node tree.
        self.scene = bpy.context.scene
        self.scene.use_nodes = True
        self.scene.render.use_compositing = True
        self.tree = self.scene.node_tree

        # Set frame resolution
        self.scene.render.resolution_y = self.height
        self.scene.render.resolution_x = self.width
        self.scene.render.resolution_percentage = 100

        # Render Optimizations and Settings
        if self.scene.render.engine.upper() == "CYCLES":
            self.scene.cycles.use_denoising = True
            self.scene.cycles.adaptive_threshold = adaptive_threshold
        else:
            print(
                f"WARNING: Using {self.scene.render.engine.upper()} rendering engine, "
                f"with default OpenGL rendering device(s)."
            )
        self.scene.render.use_persistent_data = True
        self.scene.render.use_motion_blur = use_motion_blur
        self.scene.render.motion_blur_shutter /= keyframe_multiplier

        # Image settings
        self.scene.render.image_settings.file_format = file_format.upper()
        self.scene.render.image_settings.color_depth = str(color_depth)
        self.scene.render.image_settings.color_mode = "RGB" if self.alpha_color else "RGBA"

        # Make sure there's a camera
        cameras = [ob for ob in self.scene.objects if ob.type == "CAMERA"]
        if not cameras:
            raise RuntimeError("No camera found, please add one manually.")
            # TODO: Add default camera?
            # camera_data = bpy.data.cameras.new(name="Camera")
            # self.camera = bpy.data.objects.new("Camera", camera_data)
            # self.scene.collection.objects.link(self.camera)
            # self.scene.camera = self.camera
        elif len(cameras) > 1 and self.scene.camera:
            self.camera = self.scene.camera
            print(f"Multiple cameras found. Using active camera named: '{self.camera.name}'.")
        else:
            self.camera = cameras[0]
            print(f"No active camera was found. Using camera named: '{self.camera.name}'.")

        self.move_keyframes(scale=keyframe_multiplier)
        self.keyframe_multiplier = keyframe_multiplier

        if unbind_camera:
            # Assume we are using provided spline for camera movements
            # Remove extra file output pipelines
            for n in getattr(self.tree, "nodes", []):
                if isinstance(n, bpy.types.CompositorNodeOutputFile):
                    self.tree.nodes.remove(n)

            # Remove constraints, animations and parents
            for c in self.camera.constraints:
                self.camera.constraints.remove(c)
            self.camera.animation_data_clear()
            self.camera.parent = None
        else:
            # Use existing blender animation for movements
            if not use_animation:
                raise ValueError("Both `unbind_camera` and `use_animation` are false, scene will be entirely static!")
            elif (
                all(p.animation_data is None for p in self.get_parents(self.camera))
                and self.camera.animation_data is None
            ):
                print("WARNING: Active camera nor it's parents are animated, camera will be static.")

        # Setup alpha blending if needed...
        if self.alpha_color:
            # Set clear background
            self.scene.render.dither_intensity = 0.0
            self.scene.render.film_transparent = True

            # Bypass everything, and only alpha blend
            # This is a bit fragile atm, as it relies on the node to have correct names...
            alpha_compositor = self.tree.nodes.new(type="CompositorNodeAlphaOver")
            alpha_compositor.inputs[1].default_value = list(self.alpha_color) + [1.0] * (4 - len(self.alpha_color))
            self.tree.links.new(
                self.tree.nodes["Render Layers"].outputs["Image"],
                alpha_compositor.inputs[2],
            )
            self.tree.links.new(
                self.tree.nodes["Render Layers"].outputs["Alpha"],
                alpha_compositor.inputs[0],
            )
            self.tree.links.new(
                alpha_compositor.outputs[0],
                self.tree.nodes["Composite"].inputs["Image"],
            )
        else:
            self.scene.render.film_transparent = False

        # Setup for getting normals and depth
        self.depth, self.normals = depth, normals
        self.depth_path = None
        self.normals_path = None
        self.render_layers = None

        if depth or normals:
            # Add passes for additionally dumping depth and normals.
            if len(keys := list(self.scene.view_layers.keys())) < 1:
                raise ValueError("Expected at least one view layer, cannot render without it. Please add one manually.")

            self.scene.view_layers[keys[0]].use_pass_normal = normals
            self.scene.view_layers[keys[0]].use_pass_z = depth
            self.render_layers = self.tree.nodes.new("CompositorNodeRLayers")

            if depth:
                self.depth_path = self.include_depth()
                (self.root_path / "depths").mkdir(parents=True, exist_ok=True)

            if normals:
                self.normals_path = self.include_normals()
                (self.root_path / "normals").mkdir(parents=True, exist_ok=True)

        self.enable_devices(device.upper(), indices=device_idxs)

    def move_keyframes(self, scale=1.0, shift=0.0):
        """Adjusts keyframes in Blender animations.

        Args:
            scale: Factor used to rescale keyframe positions along x-axis. Defaults to 1.0.
            shift: Factor used to shift keyframe positions along x-axis. Defaults to 0.0.

        """
        # TODO: This method can be slow if there's a lot of keyframes
        #   See: https://blender.stackexchange.com/questions/111644
        if scale == 1.0 and shift == 0.0:
            return
        if self.use_animation:
            for action in bpy.data.actions:
                for fcurve in action.fcurves or []:
                    for kfp in fcurve.keyframe_points or []:
                        # Note: Updating `co_ui` does not move the handles properly!!!
                        kfp.co.x = kfp.co.x * scale + shift
                        kfp.handle_left.x = kfp.handle_left.x * scale + shift
                        kfp.handle_right.x = kfp.handle_right.x * scale + shift
                        kfp.period *= scale

    def include_depth(self):
        """
        Sets up Blender compositor to include depth map in rendered images.
        """
        depth_file_output = self.tree.nodes.new(type="CompositorNodeOutputFile")
        depth_file_output.label = "Depth Output"
        depth_file_output.format.file_format = "OPEN_EXR"
        self.tree.links.new(self.render_layers.outputs["Depth"], depth_file_output.inputs[0])
        depth_file_output.base_path = str(self.root_path)
        return depth_file_output

    def include_normals(self):
        """
        Sets up Blender compositer to include normal map in rendered images.
        """
        normal_file_output = self.tree.nodes.new(type="CompositorNodeOutputFile")
        normal_file_output.label = "Normal Output"
        self.tree.links.new(self.render_layers.outputs["Normal"], normal_file_output.inputs[0])
        normal_file_output.base_path = str(self.root_path)
        return normal_file_output

    @staticmethod
    def look_at(obj_camera, point):
        """Orients camera in Blender to look at point.

        Args:
            obj_camera: The camera to orient.
            point: Point in 3D space.
        """
        # Note: Make sure too call `bpy.context.view_layer.update()` after!
        # See: https://blender.stackexchange.com/questions/5210

        point = mathutils.Vector(point) if not isinstance(point, mathutils.Vector) else point
        direction = point - obj_camera.location
        # point the cameras '-Z' and use its 'Y' as up
        rot_quat = direction.to_track_quat("-Z", "Y")

        # assume we're using euler rotation
        obj_camera.rotation_euler = rot_quat.to_euler()

    @staticmethod
    def enable_devices(device_type, indices=slice(None), use_cpus=False):
        """Enables/activates rendering devices in Blender.

        Args:
            device_type: Name of device to use. "none", "cuda", "optix", "metal".
            indices: Specify which devices to activate. Defaults to all devices.
            use_cpus: Boolean flag to enable CPUs alongside GPU devices.

        :return:
            List of activated devices.
        """
        # Modified from: https://blender.stackexchange.com/questions/156503
        if bpy.context.scene.render.engine.upper() != "CYCLES":
            return []

        preferences = bpy.context.preferences
        cycles_preferences = preferences.addons["cycles"].preferences
        cycles_preferences.refresh_devices()
        devices = cycles_preferences.devices

        if not devices:
            raise RuntimeError("No devices found!")
        if device_type.lower() not in ("none", "cuda", "optix", "metal"):
            raise ValueError("Unrecognized device type!")

        for device in devices:
            device.use = False

        activated_devices = []
        device_type_ = "CPU" if device_type.lower() == "none" else device_type
        devices_ = filter(lambda d: d.type == device_type_, devices)
        devices_ = np.array(list(devices_), dtype=object)[indices]

        for device in itertools.chain(devices_, filter(lambda d: d.type == "CPU" and use_cpus, devices)):
            print("INFO: Activated gpu", device.name, device.type)
            activated_devices.append(device.name)
            device.use = True
        cycles_preferences.compute_device_type = device_type
        bpy.context.scene.cycles.device = "CPU" if device_type == "CPU" else "GPU"
        return activated_devices

    @staticmethod
    def parse_json_str(string):
        """Parse a string that is assumed to either be valid json or the path to a valid json file

        Args:
            string: String to parse or string file path

        :return:
            If input is not a string, passes through. If can be parsed, returns parsed JSON data. else, loads and parses.
        """
        if not isinstance(string, str):
            # Pass through for default args
            return string
        try:
            return json.loads(string)
        except json.JSONDecodeError:
            path = Path(string).expanduser().resolve()
            with open(str(path), "r") as f:
                return json.load(f)

    @staticmethod
    def get_parents(obj):
        """Retrieves parent objects of a given object in Blender

        Args:
            obj: Object to find parent of.

        :return:
            List of parent objects of obj.
        """
        if getattr(obj, "parent", None):
            return [obj.parent] + _BlenderDatasetGenerator.get_parents(obj.parent)
        return []

    @staticmethod
    def get_camera_intrinsics():
        """Calculates camera intrinsics matrix for active camera in Blender,
        which defines how 3D points are projected onto 2D.

        :return:
            Camera intrinsics matrix based on camera properties.
        """

        # Based on: https://mcarletti.github.io/articles/blenderintrinsicparams/
        scene = bpy.context.scene
        scale = scene.render.resolution_percentage / 100
        width = scene.render.resolution_x * scale  # px
        height = scene.render.resolution_y * scale  # px
        camera_data = scene.camera.data

        aspect_ratio = width / height
        K = np.zeros((3, 3), dtype=np.float32)
        K[0][0] = width / 2.0 / np.tan(camera_data.angle / 2)
        K[1][1] = height / 2.0 / np.tan(camera_data.angle / 2) * aspect_ratio
        K[0][2] = width / 2.0
        K[1][2] = height / 2.0
        K[2][2] = 1.0
        K.transpose()
        return K

    def position_camera(self, camera_location, target_location=None, camera_rotation=None):
        """
        Positions and orients camera in Blender scene according to specified parameters.

        Args:
            camera_location: Location to place camera in 3D space.
            target_location: Location to point camera. Defaults to none.
            camera_rotation: Rotation matrix for camera. Defaults to none.

        """
        # Move camera to position, orient it towards object
        if not (target_location is not None) ^ (camera_rotation is not None):
            raise ValueError("Only one of `target_location` or `camera_rotation` can be set.")

        if target_location is not None:
            self.camera.location = mathutils.Vector(camera_location)
            self.look_at(self.camera, target_location)
        else:
            matrix_world = np.eye(4)
            matrix_world[:3, :3] = camera_rotation
            matrix_world[:3, -1] = camera_location
            self.camera.matrix_world = mathutils.Matrix(matrix_world)
        bpy.context.view_layer.update()

    def generate_single(self, index):
        """
        Generates a single frame in Blender and manages the file paths for that frame, including depth and normals.

        Args:
            index: index of frame to generate.

        :return:
            paths dictionary containing paths to rendered frames for this index.

        """
        # Assumes the camera position, frame number and all other params have been set
        # Set paths to save images
        paths = {"file_path": Path(f"frames/frame_{index:06}").with_suffix(self.scene.render.file_extension)}
        self.scene.render.filepath = str(self.root_path / "frames" / f"frame_{index:06}")
        if self.depth:
            paths["depth_file_path"] = Path(f"depths/depth_{index:06}.exr")
            self.depth_path.file_slots[0].path = f"depths/depth_{'#'*6}"
        if self.normals:
            paths["normals_file_path"] = Path(f"normals/normal_{index:06}").with_suffix(self.scene.render.file_extension)
            self.normals_path.file_slots[0].path = f"normals/normal_{'#'*6}"
        exists_all = all(Path(self.root_path / p).exists() for p in paths.values())

        # Render frame(s), skip the render iff all files exist and `allow_skips`
        if self.render:
            if not (exists_all and self.allow_skips):
                # If `write_still` is false, depth & normals can be written but rgb will be skipped
                skip_frame = Path(self.root_path / paths["file_path"]).exists() and self.allow_skips
                bpy.ops.render.render(write_still=not skip_frame)

        # Returns paths that were written
        return paths

    def generate_views_along_spline(
        self,
        *,
        location_points,
        viewing_points=np.zeros((1, 3)),
        frame_start=None,
        frame_end=None,
        frame_step=None,
        num_frames=None,
        tnb=False,
        **kwargs,
    ):
        """
        This is the core frame generation process. Determines frame range to render,
        sets camera positions and orientations,
        and renders all frames.

        Args:
            location_points: Locations on spline path camera should follow during rendering.
            viewing_points: Specifies points camera should be directed at during rendering. Defaults to (0,0,0).
            frame_start: Starting index of frames to render as seen in blender. Defaults to value from `.blend` file.
            frame_end: Ending index of frames to render as seen in blender. Defaults to value from `.blend` file.
            frame_step: Skip every nth frame. This is applied after scaling by `keyframe-multiplier`. Defaults to value from `.blend` file.
            num_frames: Total number of frames to be rendered, applied after scaling by `keyframe-multiplier`. Cannot be used with `frame-end`. Default: None.
            tnb: Boolean flag indicating camera orientation should be based on TNB frame. Defaults to false.

        """
        # Remap requested frame range according to keyframe multiplier
        # Cases:
        #   1) frame/start/end and step are set:
        #      - remap range(start, end, step) w/ keyframe multiplier and render
        #   2) Nothing is set, render default animation w/ kf multiplier
        #   3) start and num-frames is set, render start*mul to start*mul + num-frames with step size
        if num_frames is not None and frame_end is not None:
            raise ValueError("Cannot use both `num-frames` and `frame-end` simultaneously.")

        frame_start = int((self.scene.frame_start if frame_start is None else frame_start) * self.keyframe_multiplier)
        frame_step = self.scene.frame_step if frame_step is None else frame_step

        if frame_end is None:
            if num_frames is None:
                frame_end = int(self.scene.frame_end * self.keyframe_multiplier)
            else:
                frame_end = frame_start + num_frames
        frame_range = range(frame_start, frame_end, frame_step)

        # Warn if generated frames lie outside animation range
        if self.use_animation and (
            (frame_start / self.keyframe_multiplier < self.scene.frame_start)
            or (self.scene.frame_end < frame_end / self.keyframe_multiplier)
        ):
            print(
                f"WARNING: Current animation starts at frame #{self.scene.frame_start} and ends at "
                f"#{self.scene.frame_end} (with step={self.scene.frame_step}), but you requested frames "
                f"#{frame_start / self.keyframe_multiplier} to #{frame_end / self.keyframe_multiplier} "
                f"(step={frame_step}) to be rendered.\n"
            )

        # Apply animation slowdown/speedup factor and shift keyframes so they start at zero
        if self.use_animation and self.keyframe_multiplier != 1.0:
            shift = frame_start
            new_frame_range = range(0, frame_end - shift, frame_step)
            self.move_keyframes(shift=-shift)
        else:
            new_frame_range = frame_range
            shift = 0

        # Max number of frames is 1,048,574 as of Blender 3.4
        # See: https://docs.blender.org/manual/en/latest/advanced/limits.html
        if new_frame_range.stop - new_frame_range.start >= 1_048_574:
            raise RuntimeError(
                f"Blender cannot currently render more than 1,048,574 frames, yet requested you "
                f"requested {new_frame_range.stop - new_frame_range.start} frames to be rendered. For more please see: "
                f"https://docs.blender.org/manual/en/latest/advanced/limits.html"
            )

        # If we request to render frames outside the nominal animation range,
        # blender will just wrap around and go back to that range. As a workaround,
        # extend the animation range to it's maximum, even if we do not exceed it,
        # it will be restored at the end.
        scene_original_range = self.scene.frame_start, self.scene.frame_end
        self.scene.frame_start, self.scene.frame_end = 0, 1_048_574

        # Store transforms as we go
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
        transforms["w"] = self.width
        transforms["h"] = self.height
        transforms["c"] = 3 if self.alpha_color else 4  # RGB vs RGBA
        transforms["fl_x"] = 1 / 2 * self.width / np.tan(1 / 2 * self.camera.data.angle)
        transforms["fl_y"] = 1 / 2 * self.height / np.tan(1 / 2 * self.camera.data.angle)
        transforms["cx"] = 1 / 2 * self.width + transforms["shift_x"]
        transforms["cy"] = 1 / 2 * self.height + transforms["shift_y"]
        transforms["intrinsics"] = self.get_camera_intrinsics().tolist()
        interrupted = False

        # Sanitize arguments if they come from CLI
        location_points = np.array(self.parse_json_str(location_points))
        viewing_points = np.array(self.parse_json_str(viewing_points))

        # Generate splines (if needed)
        ts = np.linspace(0, 1, len(new_frame_range), endpoint=False)
        location_points = Spline(location_points, **kwargs) if self.unbind_camera else None
        viewing_points = Spline(viewing_points, **kwargs) if self.unbind_camera else None

        # Handle CTRL+C
        def sigint_handler(*args):
            nonlocal interrupted
            interrupted = True

        signal.signal(signal.SIGINT, sigint_handler)

        # Capture frames!
        for i, (t, frame_number) in track(
            enumerate(zip(ts, new_frame_range)),
            description="Generating frames... ",
            total=len(ts),
        ):
            if interrupted:
                break
            if self.unbind_camera:
                # If camera is unbound, then we are following explicit path,
                # otherwise, assume the camera will be moved by an animation
                if tnb:
                    T, N, B = location_points.tnb_frame(t, smooth=False)
                    rot = mathutils.Matrix(np.stack([B, N, -T]).squeeze().T)
                    self.position_camera(location_points(t), camera_rotation=rot)
                else:
                    self.position_camera(location_points(t), target_location=viewing_points(t))

            self.scene.frame_set(frame_number if self.use_animation else 0)
            paths = self.generate_single(frame_number)

            # Get camera pose and normalize rotation matrix axes
            pose = np.array(self.camera.matrix_world)
            pose[:3, :3] /= np.linalg.norm(pose[:3, :3], axis=0)
            frame_data = {
                **{k: str(p) for k, p in paths.items()},
                "transform_matrix": pose.tolist(),
            }
            transforms["frames"].append(frame_data)

        # Restore animation range if modified, and shift back keyframes
        self.scene.frame_start, self.scene.frame_end = scene_original_range
        self.move_keyframes(shift=shift)

        with open(str(self.root_path / "transforms.json"), "w") as f:
            json.dump(transforms, f, indent=2)


def _parser_config():
    """Define all arguments for the default Argparse CLI

    :return:
        Dictionary containing configuration for argument parser.
    """
    if sys.version_info < (3, 9, 0):
        boolean_action = None
    else:
        boolean_action = argparse.BooleanOptionalAction

    parser_conf = {
        "parser": dict(
            prog=(
                "Render views of a .blend file while moving camera along a spline or animated trajectory\n\n"
                "Example: \n"
                "  spsim blender.render <file.blend> <output-path> --num-frames=100 --width=800 --height=800"
            ),
            description=usage,
        ),
        "arguments": [
            dict(name="root_path", type=str, help="location at which to save dataset"),
            dict(
                name="--blend-file",
                type=str,
                default="",
                help="path to blender file to use",
            ),
            # Defaults are set below in `_render_views` to allow for some validation checks
            dict(
                name="--num-frames",
                type=int,
                default=None,
                help="number of frame to capture, this argument is affected by `keyframe-multiplier`. default: 100",
            ),
            dict(
                name="--frame-start",
                type=int,
                default=0,
                help="frame number to start capture at (inclusive), default: 0",
            ),
            dict(
                name="--frame-end",
                type=int,
                default=None,
                help="frame number to stop capture at (exclusive), default: 100",
            ),
            dict(
                name="--frame-step",
                type=int,
                default=1,
                help="step with which to capture frames, default: 1",
            ),
            dict(
                name="--location-points",
                type=str,
                help=(
                    "points defining the spline the camera follows. Expected to be json-str or path "
                    "to json file. Default is circular obit at Z=1 with radius=5."
                ),
                default="",
            ),
            dict(
                name="--viewing-points",
                type=str,
                help=(
                    "points defining the spline the camera looks at. Expected to be json-str or path "
                    "to json file. Default is static origin."
                ),
                default="",
            ),
            dict(
                name="--height",
                type=int,
                default=800,
                help="height of frame to capture, default: 800",
            ),
            dict(
                name="--width",
                type=int,
                default=800,
                help="width of frame to capture, default: 800",
            ),
            dict(
                name="--bit-depth",
                type=int,
                default=8,
                help="bit depth for frames, usually 8 for pngs, default: 8",
            ),
            dict(
                name="--device",
                type=str,
                default="optix",
                choices=["none", "cuda", "optix", "metal"],
                help="which device type to use, one of none (meaning cpu), cuda, optix. default: 'optix'",
            ),
            dict(
                name="--device-idxs",
                type=str,
                default="all",
                help="which devices to use. Ex: '[0,2]'. Default: 'all'",
            ),
            dict(
                name="--render",
                type=bool,
                action=boolean_action,
                default=True,
                help="whether or not render frames, default: True",
            ),
            dict(
                name="--tnb",
                type=bool,
                action=boolean_action,
                default=False,
                help="if true, ignore viewing points and use trajectory's TNB frame, default: False",
            ),
            dict(
                name="--unbind-camera",
                type=bool,
                action=boolean_action,
                default=True,
                help=(
                    "free the camera from it's parents, any constraints and animations it may have. Ensures it "
                    "uses the world's coordinate frame and the provided camera trajectory. default: True"
                ),
            ),
            dict(
                name="--use-animation",
                type=bool,
                action=boolean_action,
                default=True,
                help="allow any animations to play out, if false, scene will be static. default: True",
            ),
            dict(
                name="--use-motion-blur",
                type=bool,
                action=boolean_action,
                default=True,
                help="enable realistic motion blur, default: True",
            ),
            dict(
                name="--keyframe-multiplier",
                type=float,
                default=1.0,
                help="slow down animations by this factor, default: 1.0 (no slowdown)",
            ),
            dict(
                name="--periodic",
                type=bool,
                action=boolean_action,
                default=True,
                help="whether or not to make the splines periodic, default: True",
            ),
            dict(
                name="--allow-skips",
                type=bool,
                action=boolean_action,
                default=True,
                help="whether or not to skip rendering a frame if it already exists, default: True",
            ),
            dict(
                name="--unit-speed",
                type=bool,
                action=boolean_action,
                default=False,
                help="whether or not to reparameterize splines so that movement is of constant speed, default: False",
            ),
            dict(
                name="--depth",
                type=bool,
                action=boolean_action,
                default=False,
                help="whether or not to capture depth images, default: False",
            ),
            dict(
                name="--normals",
                type=bool,
                action=boolean_action,
                default=False,
                help="whether or not to capture normals images, default: False",
            ),
            dict(
                name="--file-format",
                type=str,
                default="PNG",
                help=(
                    "frame file format to use. Depth is always 'OPEN_EXR' thus is "
                    "unaffected by this setting, default: PNG"
                ),
            ),
            dict(
                name="--log-file",
                type=str,
                default=None,
                help="where to save log to, default: None (no log is saved)",
            ),
            dict(
                name="--addons",
                type=str,
                default=None,
                help="list of extra addons to enable, default: None",
            ),
            dict(
                name="--alpha-color",
                type=str,
                default=None,
                help="background color as specified by a RGB list in [0-1] range, if specified, renders will be "
                "composited with this color and be RGB instead of RGBA. default: None (no override)",
            ),
            dict(
                name="--adaptive-threshold",
                type=float,
                default=0.1,
                help="Noise threshold of rendered images, for higher quality frames make this threshold smaller. "
                "The default value is intentionally a little high to speed up renders. default: 0.1",
            ),
        ],
    }
    return parser_conf


def _get_parser():
    conf = _parser_config()
    parser = argparse.ArgumentParser(**conf["parser"])
    for argument in conf["arguments"]:
        parser.add_argument(argument.pop("name"), **argument)
    return parser


def _render_views(args):
    """
    With parsed command line arguments, configures rendering settings and renders views using Blender.

    Args:
        args: Contains command line arguments parsed.
    """
    if not args.unbind_camera and (args.location_points or args.viewing_points):
        raise ValueError(
            "Camera cannot be bound to parents and follow provided path. Either remove "
            "'location_points'/'viewing_points' arguments to enable camera to follow its "
            "animation (if any) or remove unbind camera."
        )

    if not args.location_points:
        theta = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        args.location_points = np.stack([5 * np.cos(theta), 5 * np.sin(theta), np.ones_like(theta)]).T

    if not args.viewing_points:
        args.viewing_points = np.zeros((1, 3))

    if not args.device_idxs or args.device_idxs.lower() == "all":
        args.device_idxs = slice(None)
    else:
        try:
            args.device_idxs = ast.literal_eval(args.device_idxs)
            args.device_idxs = np.atleast_1d(args.device_idxs)
        except SyntaxError:
            raise ValueError(
                f"Failed to parse `device-idxs` value {args.device_idxs}. Please "
                f"ensure this is a list of integers (i.e: [0, 1]) or 'all'/''."
            )

    if args.alpha_color:
        try:
            args.alpha_color = tuple(ast.literal_eval(args.alpha_color))
            if args.alpha_color and len(args.alpha_color) not in (3, 4):
                raise SyntaxError
        except SyntaxError:
            raise ValueError(
                f"Failed to parse `alpha-color` value {args.alpha_color}. Please "
                f"ensure this is a list of floats (i.e: [1.0,0.0,0.0] for red)"
            )

    if args.addons:
        for addon in args.addons.split(","):
            addon = addon.strip().lower()
            addon_module = addon_utils.enable(addon, default_set=True)
            print(f"INFO: Loaded addon {addon}: {addon_module}")
        print()

    if args.num_frames is not None and args.frame_end is not None:
        raise ValueError("Cannot use both `num-frames` and `frame-end` simultaneously.")

    frame_range = range(
        args.frame_start,
        args.frame_end or (args.frame_start + (args.num_frames or 100) * args.frame_step),
        args.frame_step,
    )

    with LogRedirect(args.log_file):
        bds = _BlenderDatasetGenerator(
            **(
                vars(args)
                | dict(
                    resolution=(args.height, args.width),
                    file_format=args.file_format.strip(".").upper(),
                    color_depth=args.bit_depth,
                )
            )
        )
        bds.generate_views_along_spline(frame_range=frame_range, **vars(args))


# This script has only been tested using Blender 3.3.1 (hash b292cfe5a936 built 2022-10-05 00:14:35) and above.
if __name__ == "__main__":
    if sys.version_info < (3, 9, 0):
        raise RuntimeError("Please use newer blender version with a python version of at least 3.9.")

    server = BlenderServer(port=0)
    server.start()

    # if bpy is None:
    #     print(usage)
    #     sys.exit()

    # # Get script specific arguments
    # try:
    #     index = sys.argv.index("--") + 1
    # except ValueError:
    #     index = len(sys.argv)

    # # Parse args and ignore any additional arguments
    # # enabling upstream CLI more flexibility.
    # parser = _get_parser()
    # args, extra_args = parser.parse_known_args(sys.argv[index:])
    # _render_views(args)
