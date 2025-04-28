# These were made in Blender and exported here as code
# using https://github.com/BrendanParmer/NodeToPython
# and then manually cleaned up/formatted.

# NOTE: This needs to be imported by blender to work properly.

import bpy  # type: ignore


# initialize SegmentationDebug node group
def segmentationdebug_node_group():
    segmentationdebug = bpy.data.node_groups.new(type="CompositorNodeTree", name="SegmentationDebug")

    segmentationdebug.color_tag = "NONE"
    segmentationdebug.description = ""
    segmentationdebug.default_group_node_width = 140

    # segmentationdebug interface
    # Socket Image
    image_socket = segmentationdebug.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    image_socket.attribute_domain = "POINT"

    # Socket Value
    value_socket = segmentationdebug.interface.new_socket(name="Value", in_out="INPUT", socket_type="NodeSocketFloat")
    value_socket.subtype = "NONE"
    value_socket.attribute_domain = "POINT"

    # initialize segmentationdebug nodes
    # node Group Output
    group_output = segmentationdebug.nodes.new("NodeGroupOutput")
    group_output.name = "Group Output"
    group_output.is_active_output = True

    # node Group Input
    group_input = segmentationdebug.nodes.new("NodeGroupInput")
    group_input.name = "Group Input"

    # node Combine Color
    combine_color = segmentationdebug.nodes.new("CompositorNodeCombineColor")
    combine_color.name = "Combine Color"
    combine_color.mode = "HSV"
    combine_color.ycc_mode = "ITUBT709"
    # Saturation
    combine_color.inputs[1].default_value = 1.0
    # Alpha
    combine_color.inputs[3].default_value = 1.0

    # node NormalizeIdx
    normalizeidx = segmentationdebug.nodes.new("CompositorNodeMapRange")
    normalizeidx.name = "NormalizeIdx"
    normalizeidx.use_clamp = False
    # From Min
    normalizeidx.inputs[1].default_value = 0.0
    # From Max
    normalizeidx.inputs[2].default_value = 6.0
    # To Min
    normalizeidx.inputs[3].default_value = 0.0
    # To Max
    normalizeidx.inputs[4].default_value = 1.0

    # node Math
    math = segmentationdebug.nodes.new("CompositorNodeMath")
    math.name = "Math"
    math.operation = "ADD"
    math.use_clamp = True
    # Value_001
    math.inputs[1].default_value = 0.0

    # Set locations
    group_output.location = (314.0, 0.0)
    group_input.location = (-281.0, 0.0)
    combine_color.location = (124.0, 0.0)
    normalizeidx.location = (-78.5, 118.19999694824219)
    math.location = (-78.5, -118.19999694824219)

    # initialize segmentationdebug links
    # normalizeidx.Value -> combine_color.Red
    segmentationdebug.links.new(normalizeidx.outputs[0], combine_color.inputs[0])
    # math.Value -> combine_color.Blue
    segmentationdebug.links.new(math.outputs[0], combine_color.inputs[2])
    # group_input.Value -> normalizeidx.Value
    segmentationdebug.links.new(group_input.outputs[0], normalizeidx.inputs[0])
    # group_input.Value -> math.Value
    segmentationdebug.links.new(group_input.outputs[0], math.inputs[0])
    # combine_color.Image -> group_output.Image
    segmentationdebug.links.new(combine_color.outputs[0], group_output.inputs[0])
    return segmentationdebug
