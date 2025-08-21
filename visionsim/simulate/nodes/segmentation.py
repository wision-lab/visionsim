# These were made in Blender and exported here as code
# using https://github.com/BrendanParmer/NodeToPython
# and then manually cleaned up/formatted.

# NOTE: This needs to be imported by blender to work properly.

import bpy  # type: ignore

from .common import new_socket


# initialize SegmentationDebug node group
def segmentationdebug_node_group():
    segmentationdebug = bpy.data.node_groups.new(type="CompositorNodeTree", name="SegmentationDebug")

    if bpy.app.version >= (4, 3, 0):
        segmentationdebug.default_group_node_width = 140

    # segmentationdebug interface
    # Socket Image
    new_socket(segmentationdebug, name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")

    # Socket Value
    new_socket(segmentationdebug, name="Value", in_out="INPUT", socket_type="NodeSocketFloat")

    # initialize segmentationdebug nodes
    # node Group Output
    group_output = segmentationdebug.nodes.new("NodeGroupOutput")
    group_output.name = "Group Output"
    group_output.is_active_output = True

    # node Group Input
    group_input = segmentationdebug.nodes.new("NodeGroupInput")
    group_input.name = "Group Input"

    # node Combine Color
    if bpy.app.version >= (3, 3, 0):
        combine_color = segmentationdebug.nodes.new("CompositorNodeCombineColor")
        combine_color.mode = "HSV"
    else:
        combine_color = segmentationdebug.nodes.new("CompositorNodeCombHSVA")
    combine_color.name = "Combine Color"
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
