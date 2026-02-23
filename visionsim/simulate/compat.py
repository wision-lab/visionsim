from __future__ import annotations

import os
from pathlib import Path

import bpy  # type: ignore


def file_output_node(
    tree: bpy.types.CompositorNodeTree,
    directory: str | os.PathLike,
    slot_names: tuple[str | tuple[str, str]] = ("####",),
    label: str = "File Output",
    media_type: str = "IMAGE",
    preview: bool = False,
    color_mode: str = "RGB",
) -> tuple[
    bpy.types.CompositorNodeOutputFile,
    list[bpy.types.NodeSocket],
    list[bpy.types.NodeOutputFileSlotFile | bpy.types.NodeCompositorFileOutputItem],
]:
    # See: https://developer.blender.org/docs/release_notes/5.0/python_api/#nodes
    slot_info = [slot if isinstance(slot, tuple) else (slot, "RGBA") for slot in slot_names]
    node = tree.nodes.new(type="CompositorNodeOutputFile")
    node.label = label

    if bpy.app.version >= (5, 0, 0):
        node.directory = str(directory)
        node.format.media_type = media_type
        node.file_output_items.clear()
        node.file_name = ""

        slots = [node.file_output_items.new(socket_type, slot) for slot, socket_type in slot_info]
    else:
        node.base_path = str(directory)
        node.file_slots.clear()

        slots = [node.file_slots.new(slot) for slot, _ in slot_info]

    # Apply presets if the this output node is a preview node
    if preview:
        node.format.file_format = "PNG"
        node.format.compression = 90
        node.format.color_depth = "8"
        node.format.color_mode = color_mode

        for slot in slots:
            slot.name = str(Path(slot.name).with_suffix(".png"))

        # Important! Set the view settings to raw otherwise result is tonemapped
        if bpy.app.version >= (3, 2, 0):
            node.format.color_management = "OVERRIDE"
        node.format.view_settings.view_transform = "Raw"
        node.format.view_settings.look = "None"
        node.format.view_settings.gamma = 0
        node.format.view_settings.exposure = 1
        node.format.view_settings.use_curve_mapping = False

        if bpy.app.version >= (4, 3, 0):
            node.format.view_settings.use_white_balance = False

    # Trim out empty 'extra' slot
    return node, node.inputs[: len(slot_info)], slots
