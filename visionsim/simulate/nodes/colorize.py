# These were made in Blender and exported here as code
# using https://github.com/BrendanParmer/NodeToPython
# and then manually cleaned up/formatted.

# NOTE: This needs to be imported by blender to work properly.

import bpy  # type: ignore

from .common import MAPRANGE_NODE, MATH_NODE, new_socket, set_clamp


# initialize ColorizeIndices node group
def colorize_indices_node_group():
    colorize_indices = bpy.data.node_groups.new(type="CompositorNodeTree", name="ColorizeIndices")

    if bpy.app.version >= (4, 3, 0):
        colorize_indices.default_group_node_width = 140

    # colorize_indices interface
    # Socket Image
    new_socket(colorize_indices, name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")

    # Socket Value
    new_socket(colorize_indices, name="Value", in_out="INPUT", socket_type="NodeSocketFloat")

    # initialize colorize_indices nodes
    # node Group Output
    group_output = colorize_indices.nodes.new("NodeGroupOutput")
    group_output.name = "Group Output"
    group_output.is_active_output = True

    # node Group Input
    group_input = colorize_indices.nodes.new("NodeGroupInput")
    group_input.name = "Group Input"

    # node Combine Color
    if bpy.app.version >= (3, 3, 0):
        combine_color = colorize_indices.nodes.new("CompositorNodeCombineColor")
        combine_color.mode = "HSV"
    else:
        combine_color = colorize_indices.nodes.new("CompositorNodeCombHSVA")
    combine_color.name = "Combine Color"
    # Saturation
    combine_color.inputs[1].default_value = 1.0
    # Alpha
    combine_color.inputs[3].default_value = 1.0

    # node NormalizeIdx
    normalizeidx = colorize_indices.nodes.new(MAPRANGE_NODE)
    normalizeidx.name = "NormalizeIdx"
    set_clamp(normalizeidx, False)
    # From Min
    normalizeidx.inputs[1].default_value = 0.0
    # From Max
    normalizeidx.inputs[2].default_value = 6.0
    # To Min
    normalizeidx.inputs[3].default_value = 0.0
    # To Max
    normalizeidx.inputs[4].default_value = 1.0

    # node Math
    math = colorize_indices.nodes.new(MATH_NODE)
    math.name = "Math"
    math.operation = "ADD"
    set_clamp(math, True)
    # Value_001
    math.inputs[1].default_value = 0.0

    # Set locations
    group_output.location = (314.0, 0.0)
    group_input.location = (-281.0, 0.0)
    combine_color.location = (124.0, 0.0)
    normalizeidx.location = (-78.5, 118.19999694824219)
    math.location = (-78.5, -118.19999694824219)

    # initialize colorize_indices links
    # normalizeidx.Value -> combine_color.Red
    colorize_indices.links.new(normalizeidx.outputs[0], combine_color.inputs[0])
    # math.Value -> combine_color.Blue
    colorize_indices.links.new(math.outputs[0], combine_color.inputs[2])
    # group_input.Value -> normalizeidx.Value
    colorize_indices.links.new(group_input.outputs[0], normalizeidx.inputs[0])
    # group_input.Value -> math.Value
    colorize_indices.links.new(group_input.outputs[0], math.inputs[0])
    # combine_color.Image -> group_output.Image
    colorize_indices.links.new(combine_color.outputs[0], group_output.inputs[0])
    return colorize_indices
