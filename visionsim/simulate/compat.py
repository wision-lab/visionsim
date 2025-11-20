import bpy  # type: ignore


def file_output_node(
    tree, directory, slots: list[str | tuple[str, str]], label: str = "File Output", media_type: str = "IMAGE"
):
    # See: https://developer.blender.org/docs/release_notes/5.0/python_api/#nodes
    slot_info = [slot if isinstance(slot, tuple) else (slot, "RGBA") for slot in slots]
    node = tree.nodes.new(type="CompositorNodeOutputFile")
    node.label = label

    if bpy.app.version >= (5, 0, 0):
        node.directory = str(directory)
        node.format.media_type = media_type
        node.file_output_items.clear()
        node.file_name = ""

        for slot, socket_type in slot_info:
            node.file_output_items.new(socket_type, slot)
    else:
        node.base_path = str(directory)
        node.file_slots.clear()

        for slot, _ in slot_info:
            node.file_slots.new(slot)

    # Trim out empty 'extra' slot
    return node, node.inputs[: len(slot_info)]
