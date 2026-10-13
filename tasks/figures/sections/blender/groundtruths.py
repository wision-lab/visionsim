"""Ground-truth compositor figures: ``docs/source/sections/blender/groundtruths.rst``.

The page shows each kind of ground truth as an interactive graph of the compositor tree
:class:`~visionsim.simulate.blender.BlenderService` sets up for it, exported from the
quick-start scene the rest of the documentation renders its ground truth from.
"""

from __future__ import annotations

from typing import Any

from ..._page import Node, cached, export_node_trees, page_task, static

# One figure per pass: its name, the client call that builds the pass, and that call's
# arguments. Each is built on a freshly initialized service, so the figures do not
# accumulate one another's nodes. The preview node groups (NormalDebug, FlowDebug, the
# index colorizer) come out inside the pass that builds them rather than on their own.
PASSES: tuple[tuple[str, str | None, dict[str, Any]], ...] = (
    ("Composites-TopLevel", None, {}),
    ("RGB-TopLevel", "include_frames", {}),
    ("Depth-TopLevel", "include_depths", {"preview": True}),
    ("Normal-TopLevel", "include_normals", {"preview": True}),
    ("Flow-TopLevel", "include_flows", {"preview": True}),
    ("Segmentation-TopLevel", "include_segmentations", {"preview": True}),
)


def _export(executable: str | None, force: bool = False) -> None:
    """Build the page's ground-truth trees on the quick-start scene and export them.

    Args:
        executable: Blender to export with, which has to be 5.0 or newer and carry the
            Node Arrange extension.
        force: Ignored; the export overwrites unconditionally.
    """
    export_node_trees(executable, cached("ultramodern.blend"), static("blender/nodes"), "ultramodern-gt", PASSES)


NODES = (
    Node(
        name="ground-truth-node-trees",
        files=tuple(static(f"blender/nodes/{figure}.json") for figure, _, _ in PASSES),
        is_figure=True,
        recipe=_export,
    ),
)

build = page_task(NODES, "groundtruths", "Regenerate the ground-truth node tree figures.")
