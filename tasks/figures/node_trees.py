"""Export and arrange the node trees the documentation shows, as Tree Clipper JSON.

This is a :class:`~visionsim.simulate.blender.BlenderService` subclass, spawned by the figure
tasks through :meth:`BlenderClient.spawn
<visionsim.simulate.blender.BlenderClient.spawn>` with the
``tasks/figures/node_trees.py:NodeTreeService`` file target.

The figure tasks initialize it on a blend file and call :meth:`export_tree
<NodeTreeService.exposed_export_tree>` for the pass whose tree they want, while ``inv
figures.arrange-node-trees`` hands it the committed exports back to have :meth:`arrange_tree
<NodeTreeService.exposed_arrange_tree>` lay them out.

The tree the service builds has its nodes stacked on one another, while the viewer draws
them where the export says, so an export has to be arranged afterwards: imported, run
through the `Node Arrange <https://github.com/Leonardo-Pike-Excell/node-arrange>`_
extension, and written back over itself. Either use needs Tree Clipper in that blender's
python, arranging needs that extension too, and blender has to be 5.0 or newer - 4.x keeps
a scene's compositor tree out of ``bpy.data.node_groups``, where Tree Clipper looks its
roots up by name.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import bpy  # type: ignore

from visionsim.simulate.blender import BlenderService

NODE_ARRANGE = "bl_ext.user_default.node_arrange"


def _require_supported_blender() -> None:
    """Name the blender these need, and how to point the task at another one."""
    if bpy.app.version >= (5, 0, 0):
        return

    raise RuntimeError(
        f"Exporting and arranging these node trees needs blender >= 5.0, this one is {bpy.app.version_string}. "
        "Pass --executable=/path/to/blender (or set VSIM_BLENDER) to use another installation."
    )


def _require_tree_clipper() -> None:
    """Explain how to get Tree Clipper into blender's python, rather than installing it."""
    if importlib.util.find_spec("tree_clipper") is not None:
        return

    raise RuntimeError(
        "Tree Clipper is not installed in this blender's python. Install it with:\n"
        f'  {bpy.app.binary_path} --background --python-expr "import subprocess, sys; '
        "subprocess.run([sys.executable, '-m', 'pip', 'install', 'tree-clipper'], check=True)\""
    )


def _require_node_arrange() -> None:
    """Explain how to get the extension arranging needs into blender, rather than installing it."""
    if importlib.util.find_spec(NODE_ARRANGE) is not None:
        return

    raise RuntimeError(
        "The Node Arrange extension is not installed in this blender. Install it with:\n"
        f"  {bpy.app.binary_path} --command extension install node_arrange --enable"
    )


def _export(name: str, output: Path) -> int:
    """Serialize the ``name`` node tree to ``output``, and return how many trees it holds.

    Trees its group nodes point at become further ``node_trees`` entries, and pointers into
    the scene are described by the name of what they point at: that is what makes the file
    importable again.
    """
    from tree_clipper.export_nodes import ExportIntermediate, ExportParameters
    from tree_clipper.specific_handlers import BUILT_IN_EXPORTER

    intermediate = ExportIntermediate(
        ExportParameters(
            is_material=False,
            name=name,
            specific_handlers=BUILT_IN_EXPORTER,
            export_sub_trees=True,
            debug_prints=False,
            write_from_roots=False,
        )
    )
    while intermediate.step():
        pass
    intermediate.set_external(
        (external_id, (pointee := external.pointed_to_by.get_pointee()) and pointee.name)
        for external_id, external in intermediate.get_external().items()
    )
    intermediate.export_to_file(file_path=output, compress=False, json_indent=4)
    return len(intermediate.data["node_trees"])


def _scene(intermediate, external) -> bpy.types.Scene:
    """Build the scene a tree's render layers node points at, from the export's account of it.

    Tree Clipper only imports that pointer when handed a scene matching the one the tree was
    exported from - the same engine, and the same view layers down to their pass settings -
    and the export carries enough to rebuild it, which is less to ask for than the blend file
    it came out of.
    """
    info = intermediate.data["scenes"][str(external["scene_id"])]
    scene = bpy.context.scene
    scene.name = external["description"]
    scene.render.engine = info["render"]["engine"]
    wanted = {layer["name"] for layer in info["view_layers"]}

    for layer in info["view_layers"]:
        view_layer = scene.view_layers.get(layer["name"]) or scene.view_layers.new(layer["name"])
        _copy_settings(view_layer, layer)
        _copy_settings(view_layer.cycles, layer["cycles"])
        view_layer.pass_cryptomatte_depth = layer["pass_cryptomatte_depth"]

    # View layers the export does not describe are not part of the scene it came from.
    for view_layer in list(scene.view_layers):
        if view_layer.name not in wanted:
            scene.view_layers.remove(view_layer)

    return scene


def _copy_settings(target, info: dict) -> None:
    """Set the booleans ``info`` names on ``target``; read-only ones cannot be set.

    Both the export and the importer read these back: the scene's settings are written into
    the file, and a view layer's enabled passes are what give its render layers node its
    sockets, so they have to match the scene the tree came from.
    """
    for prop in target.bl_rna.properties:
        if prop.type == "BOOLEAN" and prop.identifier in info and not prop.is_readonly:
            setattr(target, prop.identifier, info[prop.identifier])


def _import(path: Path) -> list[str]:
    """Import the export at ``path`` and name the trees it brought in, the last being the root."""
    from tree_clipper.import_nodes import ImportIntermediate, ImportParameters
    from tree_clipper.specific_handlers import BUILT_IN_IMPORTER

    intermediate = ImportIntermediate(file_path=path)
    names = [tree["data"]["name"] for tree in intermediate.data["node_trees"]]
    # shortcut: a pointee that is not a Scene imports as None, so such a pointer comes
    # back described by nothing; upgrade when an export the docs draw points at one.
    intermediate.set_external(
        (int(external_id), _scene(intermediate, external) if external["fixed_type_name"] == "Scene" else None)
        for external_id, external in intermediate.get_external().items()
    )
    intermediate.import_all(ImportParameters(specific_handlers=BUILT_IN_IMPORTER, debug_prints=False))
    return names


def _arrange(tree: bpy.types.NodeTree) -> None:
    """Arrange ``tree`` the way the node editor's Arrange Selected operator does.

    That operator reads the tree it is arranging, and the sizes it spaces the nodes by, out
    of the node editor it is invoked in; nothing is driving a background blender's editor,
    so both are handed to the layout here instead.
    """
    from bl_ext.user_default.node_arrange import config, utils
    from bl_ext.user_default.node_arrange.arrange import graph, realize, sugiyama, x_coords
    from bl_ext.user_default.node_arrange.arrange.sugiyama import sugiyama_layout
    from mathutils import Vector

    # Each of these was bound into a module when the extension imported it, so a patch has
    # to be applied wherever that module looks the name up, not only where it is defined.
    for module in (graph, realize, sugiyama):
        module.get_ntree = lambda: tree

    # An undrawn node reports no size through ``dimensions``, and knows its own instead.
    # Reroutes have no size of their own, so they keep the extension's.
    dimensioned = utils.dimensions

    def dimensions(node: bpy.types.Node) -> Vector:
        return dimensioned(node) if node.bl_idname == "NodeReroute" else Vector((node.width, node.height))

    utils.dimensions = dimensions
    graph.dimensions = dimensions

    # Sockets route the links between ranks, and only a drawn node has them placed, so a
    # socket sits where its node does.
    graph.get_socket_y = lambda socket: utils.abs_loc(socket.node).y

    # Frame spacing comes off a drawn editor as well, and the extension's own formula
    # divides by the interface scale, which a never-drawn editor reports as zero.
    def frame_padding() -> float:
        preferences = bpy.context.preferences.system
        widget_unit = int(18 * preferences.ui_scale) + 2 * preferences.pixel_size
        return 1.5 * widget_unit / (preferences.ui_scale or 1)

    graph.frame_padding = frame_padding
    x_coords.frame_padding = frame_padding

    config.selected = list(tree.nodes)
    config.SETTINGS = bpy.context.scene.na_settings
    config.MARGIN = Vector(config.SETTINGS.margin).freeze()

    # The crossing reduction draws random tie-breaks, so pin the seed: arranging one
    # export twice has to write the same file twice.
    random.seed(0)

    try:
        sugiyama_layout(tree)
    finally:
        config.reset()


class NodeTreeService(BlenderService):
    """Render service that writes the trees it builds, and lays the exports of them out."""

    def exposed_export_tree(self, figure: str, output_dir: str) -> str:
        """Write the tree the service currently holds to ``<output_dir>/<figure>.json``.

        Returns:
            A line naming the file written and how many trees it holds.
        """
        _require_supported_blender()
        _require_tree_clipper()
        if not self.tree.nodes:
            raise RuntimeError(f"{figure}: the service holds an empty compositor tree.")

        # The service leaves its file output nodes pointing below ``root_path``, and the path
        # to this machine's copy of the dataset would end up in the figure: shorten it before
        # the export, so the round trip sees the same, shortened, paths.
        if self.root_path is not None:
            for node in self.tree.nodes:
                if isinstance(node, bpy.types.CompositorNodeOutputFile):
                    node.directory = Path(node.directory).relative_to(self.root_path.parent).as_posix()

        output = Path(output_dir) / f"{figure}.json"
        return f"{output.name} <- {_export(self.tree.name, output=output)} trees"

    def exposed_arrange_tree(self, path: str) -> str:
        """Arrange the trees held by the export at ``path`` and write it back over it.

        Every tree is arranged, including the ones a tree's group nodes point at: the viewer
        can be stepped into them, so they have to read as well.

        Returns:
            A line naming the file written and how many trees it holds.
        """
        _require_supported_blender()
        _require_tree_clipper()
        _require_node_arrange()

        # The trees of one export carry the names the trees of the next do, and a blend
        # holding both would export the wrong one of them, so each is arranged on its own.
        self.reset()
        bpy.ops.preferences.addon_enable(module=NODE_ARRANGE)

        export = Path(path)
        names = _import(export)
        for tree in bpy.data.node_groups:
            _arrange(tree)

        _export(names[-1], output=export)
        return f"{export.name} <- {len(names)} trees, {len(bpy.data.node_groups)} arranged"
