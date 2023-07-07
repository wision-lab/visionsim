import argparse
import ast
import itertools
import json
import os
import sys
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
    print(f"$ {sys.exec_prefix}/bin/python* -m ensurepip")
    print(f"$ {sys.exec_prefix}/bin/python* -m pip install scipy rich")
    sys.exit()

usage = (
    "This script needs to be called via blender like so: \n"
    "$ blender scene.blend --background --python render.py -- <script arguments>\n\n"
    "Or by using the spsim cli: `spsim -h blender.render`"
)


class LogRedirect:
    """Capture and redirect stdout and stderr of non-python program to log files

    We use this here as Blender is logs everything to console, slowing it down and
    creating a lot of (usually) unnecessary noise.
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
    def __init__(self, spline_points, periodic=True, unit_speed=False, samples=1000, kind="cubic", **kwargs):
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
        return np.mean([spline(t + offset) for offset in (np.arange(samples) - samples // 2) * interval], axis=0)

    def _arclength_reparameterize(self):
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
        """Compute the Frenet-Serret Frame given an *arclength* parameterized spline"""
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
                    ax.plot(x[i : i + 2], y[i : i + 2], z[i : i + 2], color=plt.cm.hsv(i / len(t)))
        return ax

    def show_tnb(self, t=0.5, length=1, **kwargs):
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


class BlenderDatasetGenerator:
    """Generate a dataset of different views of a blend file.

    (Very) loosely inspired and adapted from UPstartDeveloper's code here:
    https://github.com/UPstartDeveloper/nerf/blob/generate-blender-dataset-json/360_view.py
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
        bgcolor=None,
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
        self.render = render

        # Set frame resolution
        self.scene = bpy.context.scene
        self.tree = self.scene.node_tree
        self.scene.render.resolution_y = self.height
        self.scene.render.resolution_x = self.width
        self.scene.render.resolution_percentage = 100

        # Set bg color if provided, disables any environment maps
        if bgcolor:
            self.scene.use_nodes = False
            self.scene.world.color = bgcolor
        else:
            self.scene.use_nodes = True

        # Set clear background
        self.scene.render.dither_intensity = 0.0
        self.scene.render.film_transparent = True

        # Render Optimizations and Settings
        self.scene.render.engine = "CYCLES"
        self.scene.cycles.use_denoising = True
        self.scene.cycles.adaptive_threshold = 0.1
        # self.scene.cycles.debug_use_spatial_splits = True
        self.scene.render.use_persistent_data = True
        self.scene.render.use_motion_blur = use_motion_blur
        self.scene.render.motion_blur_shutter /= keyframe_multiplier

        # Image settings
        self.scene.render.image_settings.file_format = file_format.upper()
        self.scene.render.image_settings.color_depth = str(color_depth)
        self.scene.render.image_settings.color_mode = "RGBA"

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
            print(f"No active camera was found. Using active camera named: '{self.camera.name}'.")

        self.move_keyframes(scale=keyframe_multiplier)
        self.keyframe_multiplier = keyframe_multiplier

        if unbind_camera:
            # Assume we are using provided spline for camera movements
            # Remove extra file output pipelines
            for n in self.tree.nodes:
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

        # Setup for getting normals and depth
        self.depth, self.normals = depth, normals
        self.depth_path = None
        self.normals_path = None
        self.render_layers = None

        if depth or normals:
            # Add passes for additionally dumping albedo and normals.
            if len(keys := list(self.scene.view_layers.keys())) != 1:
                raise ValueError(f"Expected only one key, either 'RenderLayer' or 'ViewLayer', but got {keys}.")
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
        # TODO: This method can be slow if there's a lot of keyframes
        #   See: https://blender.stackexchange.com/questions/111644
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
        depth_file_output = self.tree.nodes.new(type="CompositorNodeOutputFile")
        depth_file_output.label = "Depth Output"
        depth_file_output.format.file_format = "OPEN_EXR"
        self.tree.links.new(self.render_layers.outputs["Depth"], depth_file_output.inputs[0])
        depth_file_output.base_path = ""
        return depth_file_output

    def include_normals(self):
        normal_file_output = self.tree.nodes.new(type="CompositorNodeOutputFile")
        normal_file_output.label = "Normal Output"
        self.tree.links.new(self.render_layers.outputs["Normal"], normal_file_output.inputs[0])
        normal_file_output.base_path = ""
        return normal_file_output

    @staticmethod
    def look_at(obj_camera, point):
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
        # Modified from: https://blender.stackexchange.com/questions/156503
        preferences = bpy.context.preferences
        cycles_preferences = preferences.addons["cycles"].preferences
        cycles_preferences.refresh_devices()
        devices = cycles_preferences.devices

        if not devices:
            raise RuntimeError("No devices found!")
        if device_type.lower() not in ("none", "cuda", "optix"):
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
        """Parse a string that is assumed to either be valid json or the path to a valid json file"""
        if not isinstance(string, str):
            # Pass through for default args
            return string
        try:
            return json.loads(string)
        except json.JSONDecodeError:
            path = Path(string).resolve()
            with open(str(path), "r") as f:
                return json.load(f)

    @staticmethod
    def get_parents(obj):
        if getattr(obj, "parent", None):
            return [obj.parent] + BlenderDatasetGenerator.get_parents(obj.parent)
        return []

    @staticmethod
    def get_camera_intrinsics():
        # Based on: https://mcarletti.github.io/articles/blenderintrinsicparams/
        scene = bpy.context.scene
        scale = scene.render.resolution_percentage / 100
        width = scene.render.resolution_x * scale  # px
        height = scene.render.resolution_y * scale  # px
        camdata = scene.camera.data

        aspect_ratio = width / height
        K = np.zeros((3, 3), dtype=np.float32)
        K[0][0] = width / 2.0 / np.tan(camdata.angle / 2)
        K[1][1] = height / 2.0 / np.tan(camdata.angle / 2) * aspect_ratio
        K[0][2] = width / 2.0
        K[1][2] = height / 2.0
        K[2][2] = 1.0
        K.transpose()
        return K

    def position_camera(self, camera_location, target_location=None, camera_rotation=None):
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
        # Assumes the camera position, frame number and all other params have been set
        # Set paths to save images
        paths = [Path(f"frames/frame_{index:06}").with_suffix(self.scene.render.file_extension)]
        self.scene.render.filepath = str(self.root_path / "frames" / f"frame_{index:06}")
        if self.depth:
            paths.append(Path(f"depths/depth_{index:06}.exr"))
            self.depth_path.file_slots[0].path = str(self.root_path / "depths" / f"depth_{'#'*6}")
        if self.normals:
            paths.append(Path(f"normals/normal_{index:06}").with_suffix(self.scene.render.file_extension))
            self.normals_path.file_slots[0].path = str(self.root_path / "normals" / f"normal_{'#'*6}")
        exists_all = all(Path(self.root_path / p).exists() for p in paths)

        # Render frame(s), skip the render iff all files exist and `allow_skips`
        if self.render:
            if not (exists_all and self.allow_skips):
                bpy.ops.render.render(write_still=True)

        # Returns paths that were written
        return paths

    def generate_views_along_spline(
        self, *, location_points, viewing_points=np.zeros((1, 3)), frame_range=range(100), tnb=False, **kwargs
    ):
        if self.use_animation:
            # Warn if generated frames lie outside animation range
            if frame_range.start < self.scene.frame_start or self.scene.frame_end < frame_range.stop:
                print(
                    f"WARNING: Current animation starts at frame #{self.scene.frame_start} and ends at "
                    f"#{self.scene.frame_end} (with step={self.scene.frame_step}), but you requested frames "
                    f"#{frame_range.start} to #{frame_range.stop} (step={frame_range.step}) to be rendered.\n"
                )

            # Apply animation slowdown/speedup factor and shift frames so the start at zero
            if self.keyframe_multiplier != 1.0:
                shift = int(frame_range.start * self.keyframe_multiplier)
                new_frame_range = range(
                    0,
                    int(frame_range.stop * self.keyframe_multiplier) - shift,
                    frame_range.step,
                )
                self.move_keyframes(shift=-shift)
                print(
                    f"INFO: Requested frames were {frame_range}, but these were remapped to "
                    f"{range(new_frame_range.start+shift, new_frame_range.stop+shift, frame_range.step)} "
                    f"due to a `keyframe_multiplier` value of {self.keyframe_multiplier}, and finally "
                    f"shifted to start at frame zero. The final frames are now {new_frame_range}."
                )
            else:
                shift = 0
                new_frame_range = frame_range

            if new_frame_range.stop - new_frame_range.start >= 1_048_574:
                raise RuntimeError(
                    f"Blender cannot currently render more than 1,048,574 frames, yet requested you "
                    f"requested {len(new_frame_range)} frames to be rendered. For more please see: "
                    f"https://docs.blender.org/manual/en/latest/advanced/limits.html"
                )

            # If we request to render frames outside the nominal animation range,
            # blender will just wrap around and go back to that range. As a workaround,
            # extend the animation range if we exceed it.
            # Max number of frames is 1,048,574 as of Blender 3.4
            # See: https://docs.blender.org/manual/en/latest/advanced/limits.html
            scene_original_range = self.scene.frame_start, self.scene.frame_end
            self.scene.frame_start, self.scene.frame_end = 0, 1_048_574
        else:
            new_frame_range = frame_range
            scene_original_range, shift = None, 0

        # Store transforms as we go
        transforms = {
            "camera": {
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
            },
            "frames": [],
        }
        # Note: This might be a blender bug, but when height==width,
        #   angle_x != angle_y, so here we just use angle.
        transforms["camera"]["fx"] = 1 / 2 * self.width / np.tan(1 / 2 * self.camera.data.angle)
        transforms["camera"]["fy"] = 1 / 2 * self.height / np.tan(1 / 2 * self.camera.data.angle)
        transforms["camera"]["cx"] = 1 / 2 * self.width + transforms["camera"]["shift_x"]
        transforms["camera"]["cy"] = 1 / 2 * self.height + transforms["camera"]["shift_y"]
        transforms["camera"]["intrinsics"] = self.get_camera_intrinsics().tolist()

        # Sanitize arguments if they come from CLI
        location_points = np.array(self.parse_json_str(location_points))
        viewing_points = np.array(self.parse_json_str(viewing_points))

        # Generate splines (if needed)
        ts = np.linspace(0, 1, len(new_frame_range), endpoint=False)
        location_points = Spline(location_points, **kwargs) if self.unbind_camera else None
        viewing_points = Spline(viewing_points, **kwargs) if self.unbind_camera else None

        # Capture frames!
        for i, (t, frame_number) in track(
            enumerate(zip(ts, new_frame_range)), description="Generating frames... ", total=len(ts)
        ):
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
            paths = self.generate_single(frame_number + shift if self.use_animation else i)

            frame_data = {
                "file_paths": [str(p) for p in paths],
                "transform_matrix": np.array(self.camera.matrix_world).tolist(),
            }
            transforms["frames"].append(frame_data)

        # Restore animation range if modified, and shift back keyframes
        if scene_original_range:
            self.scene.frame_start, self.scene.frame_end = scene_original_range
            self.move_keyframes(shift=shift)

        with open(str(self.root_path / "transforms.json"), "w") as f:
            json.dump(transforms, f, indent=2)


def parser_config():
    """Define all arguments for the default Argparse CLI"""
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
            dict(name="root-path", type=str, help="location at which to save dataset"),
            dict(name="--blend-file", type=str, default="", help="path to blender file to use"),
            # Defaults are set below in `_render_views` to allow for some validation checks
            dict(name="--num-frames", type=int, default=None, help="number of frame to capture, default: 100"),
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
            dict(name="--frame-step", type=int, default=1, help="step with which to capture frames, default: 1"),
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
            dict(name="--height", type=int, default=800, help="height of frame to capture, default: 800"),
            dict(name="--width", type=int, default=800, help="width of frame to capture, default: 800"),
            dict(name="--bit-depth", type=int, default=8, help="bit depth for frames, usually 8 for pngs, default: 8"),
            dict(
                name="--device",
                type=str,
                default="optix",
                choices=["none", "cuda", "optix"],
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
                help="whether or not to reparametrize splines so that movement is of constant speed, default: False",
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
                name="--log-file", type=str, default=None, help="where to save log to, default: None (no log is saved)"
            ),
            dict(name="--addons", type=str, default=None, help="list of extra addons to enable, default: None"),
            dict(
                name="--bgcolor",
                type=str,
                default=None,
                help="background color as specified by a RGB list in [0-1] range, default: None (no override)",
            ),
        ],
    }
    return parser_conf


def get_parser():
    conf = parser_config()
    parser = argparse.ArgumentParser(**conf["parser"])
    for argument in conf["arguments"]:
        parser.add_argument(argument.pop("name"), **argument)
    return parser


def _render_views(args):
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

    if args.bgcolor:
        try:
            args.bgcolor = tuple(ast.literal_eval(args.bgcolor))
        except SyntaxError:
            raise ValueError(
                f"Failed to parse `bgcolor` value {args.bgcolor}. Please "
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
        bds = BlenderDatasetGenerator(
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


# This script has only been tested using Blender 3.3.1 (hash b292cfe5a936 built 2022-10-05 00:14:35)
if __name__ == "__main__":
    if sys.version_info < (3, 9, 0):
        raise RuntimeError("Please use newer blender version with a python version of at least 3.9.")

    if bpy is None:
        print(usage)
        sys.exit()

    # Get script specific arguments
    try:
        index = sys.argv.index("--") + 1
    except ValueError:
        index = len(sys.argv)

    # Parse args and ignore any additional arguments
    # enabling upstream CLI more flexibility.
    parser = get_parser()
    args, extra_args = parser.parse_known_args(sys.argv[index:])
    _render_views(args)
