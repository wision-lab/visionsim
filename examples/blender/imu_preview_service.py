"""Example of a :class:`BlenderService` subclass that draws camera trajectories.

The subclass adds three methods on top of the stock service: it turns a set of
trajectories into frustum-and-trail markers inside whatever scene the service was
initialized with, and renders them over a still of the room. All of the drawing lives
here, in the extension, rather than in the core package, since it is only ever used by
this preview.

The markers come from a small asset: one cube carrying a geometry nodes group that
draws a camera frustum and a trail. The group takes the camera object as an input
socket and accumulates its world position once per frame, so a marker follows whatever
camera it is wired to for the whole animation. The group ships as a Tree Clipper
export (``examples/blender/mesh_camera.json``) and is imported from there, so the
marker geometry is text that can be diffed and reopened in blender.

Run it from blender, as the docs on
:doc:`extending the simulator <../../../docs/source/tutorials/extending>` describe::

    blender --background --python-use-system-env --python examples/blender/imu_preview_service.py

``examples/blender/imu_preview.py`` is the client side: it computes the trajectories
from the ultramodern ground truth, calls
:meth:`add_trajectories <ImuPreviewService.exposed_add_trajectories>` and renders the
overview of the result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import bpy
import mathutils
import numpy as np

from visionsim.simulate.blender import BlenderServer, BlenderService

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

ASSET = Path(__file__).with_name("mesh_camera.json")

MARKER_OBJECT: str = "MeshCamera"
"""Name of the marker object, and of the mesh it carries, the frustum is drawn on."""

CAMERA_SOCKET: str = "Camera"
"""Name of the geometry nodes input socket that the marker follows."""

MATERIAL_SOCKET: str = "Material"
"""Name of the geometry nodes input socket that overrides the marker's material."""

GROUND_TRUTH_MATERIAL: str = "GT-Cam"
"""Material of the ground-truth marker, which is opaque and drawn in bright blue."""

TRAJECTORY_MATERIAL: str = "Sim-Cam"
"""Material of the estimated markers, which is translucent so the trails stay readable."""

COLLECTION_NAME: str = "IMU Preview"
"""Collection the cameras and markers are linked to."""

OVERVIEW_CAMERA_NAME: str = "static-camera"
"""Name of the camera the preview is rendered from.

The scene is built to be rendered from the outside, so the preview is rendered from the
overview camera it already carries rather than from one fitted to the trajectories. A
scene without a camera of this name cannot be used.
"""

MARKER_VERTICES: tuple[tuple[float, float, float], ...] = (
    (-1.0, -1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, 1.0, 1.0),
    (1.0, -1.0, -1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, -1.0),
    (1.0, 1.0, 1.0),
)
"""Cube the marker's geometry nodes group is applied to, as :func:`bpy.ops.mesh.primitive_cube_add` makes it.

The group builds the frustum and the trail itself, so the object underneath is only
there to carry the modifier.
"""

MARKER_FACES: tuple[tuple[int, ...], ...] = (
    (0, 1, 3, 2),
    (4, 6, 7, 5),
    (0, 4, 5, 1),
    (2, 3, 7, 6),
    (0, 2, 6, 4),
    (1, 5, 7, 3),
)
"""Faces of :data:`MARKER_VERTICES`, a plain cube."""

MARKER_MATERIALS: dict[str, dict[str, float | tuple[float, float, float, float]]] = {
    GROUND_TRUTH_MATERIAL: {"base_color": (0.0044, 0.0, 0.799, 1.0), "roughness": 1.0},
    TRAJECTORY_MATERIAL: {"base_color": (0.0034, 0.0, 0.3, 1.0), "roughness": 0.5, "alpha": 0.3},
}
"""Materials the markers are drawn with, by the name :func:`_add_marker` overrides them with.

Two flat principled shaders, so they live here rather than in the asset.
"""


class ImuPreviewService(BlenderService):
    """Render service that can draw camera trajectories into the loaded scene."""

    def exposed_add_trajectories(self, trajectories, ground_truth, start_frame=1):
        """Draw a frustum-and-trail marker per trajectory, plus one for the ground truth.

        Call this after :meth:`initialize <visionsim.simulate.blender.BlenderService.exposed_initialize>`,
        which loads the scene the markers are added to. The scene's animation range is
        set to the frames the trajectories are keyed on, and the scene's own overview
        camera becomes the active one.

        Args:
            trajectories (Sequence[Sequence[Sequence[float]]]): One ``(N, 4, 4)`` block
                of camera-to-world matrices per trajectory to draw, in blender's world
                convention. Matrices have to arrive as nested lists: numpy arrays are
                not serialized by the RPC layer, they arrive as unusable proxies.
            ground_truth (Sequence[Sequence[float]]): The ``(N, 4, 4)`` matrices the
                trajectories are compared against, drawn in the ground-truth material.
            start_frame (int, optional): Frame the first pose is keyed on. Defaults to 1.

        Returns:
            dict[str, object]: Names of the created cameras, markers and overview camera,
            and the ``[start, stop]`` frame range to render.
        """
        return self._draw_trajectories(trajectories, ground_truth, start_frame=start_frame)

    def exposed_render_backdrop(self, path):
        """Render the scene once without the preview, as the image to render over.

        The room does not change over the animation and the markers are the only
        moving part, so rendering it once costs one render of the scene rather than
        one per frame.

        Args:
            path: File the still is written to.
        """
        collection = self._preview_collection()
        hidden = collection.hide_render
        collection.hide_render = True
        try:
            self.scene.render.image_settings.file_format = "PNG"
            self.scene.render.filepath = str(path)
            # The parent's single frame path (`exposed_render_current_frame`) writes into
            # `root_path`, one file per registered output, and records metadata for each,
            # none of which a one-off still has, so the render call is made directly.
            bpy.ops.render.render(write_still=True)
        finally:
            collection.hide_render = hidden

    def exposed_render_over_backdrop(self, path):
        """Render the markers over the still at ``path``, as the world background.

        Everything outside the preview collection is hidden and the still is mapped to
        the camera's view as the world, so the renderer itself draws the markers over
        it and the frames come out finished. The room only has to be rendered once,
        for the still.

        Args:
            path: Image of the scene to render the markers over.
        """
        preview = self._preview_collection()
        names = {obj.name for obj in preview.objects}
        for obj in self.scene.objects:
            obj.hide_render = obj.name not in names
        # The film has to be opaque: with `film_transparent` on, the world is not rendered
        # and the markers would have no backdrop behind them.
        self.scene.render.film_transparent = False

        world = self.scene.world or bpy.data.worlds.new(f"{COLLECTION_NAME} Backdrop")
        self.scene.world = world
        world.use_nodes = True
        # The still replaces the world's own shader as its surface, so every ray that
        # misses a marker - the whole frame but the markers - returns the room the still
        # was rendered from.
        tree = world.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputWorld")
        background = tree.nodes.new("ShaderNodeBackground")
        coordinates = tree.nodes.new("ShaderNodeTexCoord")
        still = tree.nodes.new("ShaderNodeTexImage")
        still.image = bpy.data.images.load(str(path))
        # Window coordinates put the still on the camera frame as seen, and EXTEND keeps the
        # edge pixels from repeating outside it.
        still.extension = "EXTEND"
        tree.links.new(coordinates.outputs["Window"], still.inputs["Vector"])
        tree.links.new(still.outputs["Color"], background.inputs["Color"])
        tree.links.new(background.outputs["Background"], output.inputs["Surface"])

    def _draw_trajectories(
        self,
        trajectories: Sequence[npt.ArrayLike],
        ground_truth: npt.ArrayLike,
        start_frame: int = 1,
    ) -> dict[str, object]:
        """Draw a frustum-and-trail marker per trajectory, plus one for the ground truth.

        Every trajectory is keyed on consecutive frames starting at ``start_frame``, which
        becomes the scene's animation range: the trails accumulate as the render plays, so
        the frames have to be rendered in order for them to be drawn.

        Args:
            trajectories: Camera-to-world matrices in blender's world convention, one
                ``(N, 4, 4)`` array per trajectory.
            ground_truth: The same, for the trajectory the estimates are compared to.
            start_frame: First frame of the preview.

        Returns:
            The dict :meth:`exposed_add_trajectories` hands back to the client.

        Raises:
            ValueError: If the trajectories or the ground truth are not ``(N, 4, 4)``
                arrays, or if they do not all hold the same number of poses.
        """
        ground_truth = _as_poses(ground_truth, "ground_truth")
        trajectories = [_as_poses(poses, f"trajectories[{i}]") for i, poses in enumerate(trajectories)]
        for i, poses in enumerate(trajectories):
            if len(poses) != len(ground_truth):
                raise ValueError(
                    f"trajectories[{i}] holds {len(poses)} poses, ground_truth holds {len(ground_truth)}, "
                    "all trajectories must be keyed on the same frames."
                )

        collection = self._preview_collection()
        template = _build_template(ASSET, collection)

        overview = self._overview_camera()

        cameras: list[bpy.types.Object] = []
        markers: list[bpy.types.Object] = []
        for i, poses in enumerate(trajectories):
            camera = _add_camera(collection, f"Simulated-Camera-{i}", poses, start_frame)
            cameras.append(camera)
            markers.append(_add_marker(collection, template, camera, f"MeshCamera-{i}", TRAJECTORY_MATERIAL))

        camera = _add_camera(collection, "Ground-Truth-Camera", ground_truth, start_frame)
        cameras.append(camera)
        markers.append(_add_marker(collection, template, camera, "MeshCamera-Ground-Truth", GROUND_TRUTH_MATERIAL))

        # The template only exists to be copied, so it is removed once every marker is in place.
        bpy.data.objects.remove(template, do_unlink=True)

        frame_range = (start_frame, start_frame + len(ground_truth) - 1)
        self.scene.frame_start, self.scene.frame_end = frame_range

        return {
            "cameras": [camera.name for camera in cameras],
            "markers": [marker.name for marker in markers],
            "overview": overview.name,
            "frame_range": list(frame_range),
        }

    def _preview_collection(self) -> bpy.types.Collection:
        """Get the collection the preview lives in, creating and linking it if needed."""
        collection = bpy.data.collections.get(COLLECTION_NAME)
        if collection is None:
            collection = bpy.data.collections.new(COLLECTION_NAME)
            self.scene.collection.children.link(collection)
        return collection

    def _overview_camera(self) -> bpy.types.Object:
        """Return the scene's own overview camera, made the active camera.

        Returns:
            The camera named :data:`OVERVIEW_CAMERA_NAME`, now set as ``scene.camera``.

        Raises:
            ValueError: If the scene holds no camera by that name.
        """
        camera = self.scene.objects.get(OVERVIEW_CAMERA_NAME)
        if camera is None or camera.type != "CAMERA":
            raise ValueError(f"the scene holds no camera named {OVERVIEW_CAMERA_NAME!r} to render the preview from")
        self.scene.camera = camera
        return camera


def _as_poses(poses: npt.ArrayLike, name: str) -> npt.NDArray:
    """Return ``poses`` as an ``(N, 4, 4)`` float array.

    Args:
        poses: Camera-to-world matrices, as an array or a nested sequence, which is
            what a numpy array turns into when it crosses the RPC boundary.
        name: Name of the argument, used in the error message.

    Returns:
        The matrices as a float array of shape ``(N, 4, 4)``.

    Raises:
        ValueError: If ``poses`` does not hold ``(N, 4, 4)`` matrices.
    """
    array = np.asarray(poses, dtype=float)
    if array.ndim != 3 or array.shape[1:] != (4, 4):
        raise ValueError(f"{name} must be an (N, 4, 4) array of pose matrices, got shape {array.shape}")
    return array


def _build_template(asset: str | Path, collection: bpy.types.Collection) -> bpy.types.Object:
    """Build the marker object the copies are made from, from the asset's node group.

    The asset holds the geometry nodes group as a Tree Clipper export, so the part of
    the marker that is worth versioning is a JSON file that can be diffed and opened in
    blender again, rather than a blend file. Everything around the group - the cube it
    is applied to and the two materials - is small enough to live here.

    Args:
        asset: Tree Clipper export holding the marker's geometry nodes group.
        collection: Collection to link the marker object to.

    Returns:
        The marker object, still to be copied once per trajectory.
    """
    _marker_materials()

    mesh = bpy.data.meshes.new(MARKER_OBJECT)
    mesh.from_pydata(MARKER_VERTICES, [], MARKER_FACES)
    mesh.update()

    template = bpy.data.objects.new(MARKER_OBJECT, mesh)
    collection.objects.link(template)

    modifier: bpy.types.NodesModifier = template.modifiers.new("GeometryNodes", "NODES")
    modifier.node_group = _import_node_group(asset)
    return template


def _import_node_group(asset: str | Path) -> bpy.types.NodeTree:
    """Import the geometry nodes group a Tree Clipper export holds.

    Args:
        asset: Tree Clipper export to import.

    Returns:
        The imported node group, named as the export named it.

    Raises:
        RuntimeError: If tree-clipper is not installed in blender's runtime.
        ValueError: If the export holds no node group, or imports with warnings.
    """
    try:
        from tree_clipper.import_nodes import ImportIntermediate, ImportParameters
        from tree_clipper.specific_handlers import BUILT_IN_IMPORTER
    except ImportError as exc:
        raise RuntimeError(
            f"{Path(asset).name} is a Tree Clipper export, so reading it needs the tree-clipper package "
            "in blender's runtime: pip install tree-clipper"
        ) from exc

    intermediate = ImportIntermediate(file_path=Path(asset))
    report = intermediate.import_all(ImportParameters(specific_handlers=BUILT_IN_IMPORTER, debug_prints=False))
    if report.warnings:
        raise ValueError(f"{Path(asset).name} imported with warnings: {report.warnings}")
    return bpy.data.node_groups[report.last_getter().name]


def _marker_materials() -> None:
    """Create the materials the markers are drawn with, leaving loaded ones alone."""
    for name, values in MARKER_MATERIALS.items():
        material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        if material.node_tree is None:
            material.use_nodes = True

        bsdf = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
        bsdf.inputs["Base Color"].default_value = values["base_color"]
        bsdf.inputs["Roughness"].default_value = values["roughness"]
        bsdf.inputs["Alpha"].default_value = values.get("alpha", 1.0)


def _add_camera(
    collection: bpy.types.Collection,
    name: str,
    poses: npt.NDArray,
    start_frame: int,
) -> bpy.types.Object:
    """Add a camera keyed on ``poses``, one pose per frame from ``start_frame``."""
    camera = bpy.data.objects.new(name, bpy.data.cameras.new(name))
    collection.objects.link(camera)

    for offset, pose in enumerate(poses):
        camera.matrix_world = mathutils.Matrix(pose.tolist())
        frame = start_frame + offset
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)
        camera.keyframe_insert(data_path="scale", frame=frame)
    return camera


def _add_marker(
    collection: bpy.types.Collection,
    template: bpy.types.Object,
    camera: bpy.types.Object,
    name: str,
    material: str,
) -> bpy.types.Object:
    """Copy ``template`` into a marker that follows ``camera``.

    Args:
        collection: Collection to link the marker to.
        template: Object to copy, whose geometry nodes group draws the marker.
        camera: Camera the marker is wired to, and whose pose the frustum takes.
        name: Name of the new marker.
        material: Name of the material overriding the marker's own.

    Returns:
        The new marker object.

    Raises:
        KeyError: If no material named ``material`` is loaded.
    """
    marker = template.copy()
    marker.name = name
    collection.objects.link(marker)

    modifier = next(m for m in marker.modifiers if m.type == "NODES")
    _set_socket(modifier, CAMERA_SOCKET, camera)
    _set_socket(modifier, MATERIAL_SOCKET, bpy.data.materials[material])
    return marker


def _set_socket(modifier: bpy.types.NodesModifier, socket_name: str, value: object) -> None:
    """Assign ``value`` to the named input socket of a geometry nodes modifier.

    Args:
        modifier: Modifier whose node group takes the socket.
        socket_name: Name of the input socket, as shown in the node group interface.
        value: Object or material to assign.

    Raises:
        KeyError: If the node group has no input socket by that name.
    """
    node_group = modifier.node_group
    # The shared node utils only build group interfaces (`new_socket`); none assigns a value
    # to an existing modifier input, so the lookup is done here.
    for item in node_group.interface.items_tree:
        if item.item_type == "SOCKET" and item.in_out == "INPUT" and item.name == socket_name:
            if hasattr(modifier, "properties"):
                # Blender 5.0 moved the socket values off the modifier's ID properties.
                modifier.properties.inputs[item.identifier]["value"] = value
            else:
                modifier[item.identifier] = value
            return

    raise KeyError(f"geometry nodes group {node_group.name!r} has no {socket_name!r} input socket")


if __name__ == "__main__":
    try:
        index = sys.argv.index("--") + 1
    except ValueError:
        index = len(sys.argv)

    parser = argparse.ArgumentParser("Start a BlenderServer exposing ImuPreviewService.")
    parser.add_argument("-p", "--port", type=int, default=0)
    args, unknown = parser.parse_known_args(sys.argv[index:])

    server = BlenderServer(service=ImuPreviewService, port=args.port)
    server.start()
