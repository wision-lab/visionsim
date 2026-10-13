"""Export the node trees the documentation shows, as Tree Clipper JSON.

This is a :class:`~visionsim.simulate.blender.BlenderService` subclass, so it runs in
blender like any other render service and is driven over the same client/server split:
the client initializes it on a blend file, asks for the ground-truth pass whose tree it
wants, and calls :meth:`export_tree <NodeTreeExportService.exposed_export_tree>`. It is
spawned the way :doc:`the extending tutorial <../../../docs/source/tutorials/extending>`
describes for subclasses::

    blender --background --enable-autoexec --python-use-system-env \\
        --python tasks/figures/export_node_trees.py -- --port <port>

Tree Clipper has to be installed in that blender's python, which the export explains if
it is missing, and blender has to be 5.0 or newer - 4.x keeps a scene's compositor tree
out of ``bpy.data.node_groups``, where Tree Clipper looks its roots up by name.

An export carries every node's ``location``, ``width`` and ``height``, which is the
layout the viewer draws, so the figures are arranged by hand rather than by this task:
import the file in blender, lay the tree out in the node editor, export it again and
commit that. What this task writes instead is the tree as the service created it, with
the nodes still stacked on one another.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import bpy  # type: ignore

from visionsim.simulate.blender import BlenderServer, BlenderService


def _require_supported_blender() -> None:
    """Explain what a blender older than 5.0 cannot do, rather than failing obscurely."""
    if bpy.app.version >= (5, 0, 0):
        return

    raise RuntimeError(
        f"Exporting the compositor node trees needs blender >= 5.0, this one is {bpy.app.version_string}. "
        "Blender 4.x keeps a scene's compositor node tree out of bpy.data.node_groups, where Tree Clipper "
        "looks its root up by name, and refuses to add a render layers node to any other node tree, so "
        "neither the export nor its round trip is possible there. Pass --executable=/path/to/blender (or "
        "set VSIM_BLENDER) to use another installation."
    )


def _require_tree_clipper() -> None:
    """Explain how to get Tree Clipper into blender's python, rather than installing it."""
    if importlib.util.find_spec("tree_clipper") is not None:
        return

    raise RuntimeError(
        "Tree Clipper is not installed in this blender's python. Install it into that interpreter with:\n"
        f'  {bpy.app.binary_path} --background --python-expr "import subprocess, sys; '
        "subprocess.run([sys.executable, '-m', 'pip', 'install', 'tree-clipper'], check=True)\""
    )


def _shorten_output_paths(tree: bpy.types.NodeTree, root: Path | None) -> None:
    """Make the file output nodes' paths readable in the exported graph.

    The service writes its renders below the ``root_path`` it is initialized with,
    which would put this machine's absolute paths into the figure; the path below the
    dataset directory is all that reads. Done before the export, so the round trip
    sees the same, shortened, paths.
    """
    if root is None:
        return

    for node in tree.nodes:
        if isinstance(node, bpy.types.CompositorNodeOutputFile):
            node.directory = Path(node.directory).relative_to(root.parent).as_posix()


def _export(name: str, output: Path) -> dict[str, Any]:
    """Serialize the ``name`` node tree, write it to ``output``, and return the serialization.

    Node trees that the tree's group nodes point at are exported as further entries in
    ``node_trees``, and pointers into the scene are described by the name of what they
    point at, which is what makes the file importable again.
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
        (external_id, external.pointed_to_by.get_pointee().name)
        for external_id, external in intermediate.get_external().items()
    )
    intermediate.export_to_file(file_path=output, compress=False, json_indent=4)
    return json.loads(intermediate.export_to_str(compress=False, json_indent=4))


class NodeTreeExportService(BlenderService):
    """Render service that writes the compositor trees it builds as Tree Clipper exports."""

    def exposed_export_tree(self, figure: str, output_dir: str) -> str:
        """Write the tree the service currently holds.

        Args:
            figure: File to write, without the ``.json`` suffix.
            output_dir: Directory to write it to.

        Returns:
            A line naming the file written and how many trees it holds.
        """
        _require_supported_blender()
        _require_tree_clipper()
        if not self.tree.nodes:
            raise RuntimeError(f"{figure}: the service holds an empty compositor tree.")

        # The service leaves its file output nodes pointing at this machine's absolute
        # paths, which would end up in the figure.
        _shorten_output_paths(self.tree, self.root_path)

        output = Path(output_dir) / f"{figure}.json"
        exported = _export(self.tree.name, output=output)
        return f"{output.name} <- {len(exported['node_trees'])} trees"


if __name__ == "__main__":
    try:
        index = sys.argv.index("--") + 1
    except ValueError:
        index = len(sys.argv)

    parser = argparse.ArgumentParser("Start a BlenderServer exposing NodeTreeExportService.")
    parser.add_argument("-p", "--port", type=int, default=0)
    arguments, _ = parser.parse_known_args(sys.argv[index:])

    BlenderServer(service=NodeTreeExportService, port=arguments.port).start()
