# These were made in Blender and exported here as code
# using https://github.com/BrendanParmer/NodeToPython
# and then manually cleaned up/formatted.

# NOTE: This needs to be imported by blender to work properly.

import bpy  # type: ignore

from .common import new_socket


# initialize Cartesian2Polar node group
def cartesian2polar_node_group():
    cartesian2polar = bpy.data.node_groups.new(type="CompositorNodeTree", name="Cartesian2Polar")

    if bpy.app.version >= (4, 3, 0):
        cartesian2polar.default_group_node_width = 140

    # cartesian2polar interface
    # Socket r
    new_socket(cartesian2polar, name="r", in_out="OUTPUT", socket_type="NodeSocketFloat")

    # Socket theta
    new_socket(cartesian2polar, name="theta", in_out="OUTPUT", socket_type="NodeSocketFloat")

    # Socket x
    new_socket(cartesian2polar, name="x", in_out="INPUT", socket_type="NodeSocketFloat")

    # Socket y
    new_socket(cartesian2polar, name="y", in_out="INPUT", socket_type="NodeSocketFloat")

    # initialize cartesian2polar nodes
    # node Group Output
    group_output = cartesian2polar.nodes.new("NodeGroupOutput")
    group_output.name = "Group Output"
    group_output.is_active_output = True

    # node Group Input
    group_input = cartesian2polar.nodes.new("NodeGroupInput")
    group_input.name = "Group Input"

    # node Arctan2
    arctan2 = cartesian2polar.nodes.new("CompositorNodeMath")
    arctan2.name = "Arctan2"
    arctan2.operation = "ARCTAN2"
    arctan2.use_clamp = False

    # node Sqrt
    sqrt = cartesian2polar.nodes.new("CompositorNodeMath")
    sqrt.name = "Sqrt"
    sqrt.operation = "SQRT"
    sqrt.use_clamp = False

    # node Sum
    sum = cartesian2polar.nodes.new("CompositorNodeMath")
    sum.name = "Sum"
    sum.operation = "ADD"
    sum.use_clamp = False

    # node SquareX
    squarex = cartesian2polar.nodes.new("CompositorNodeMath")
    squarex.name = "SquareX"
    squarex.operation = "POWER"
    squarex.use_clamp = False
    # Value_001
    squarex.inputs[1].default_value = 2.0

    # node SquareY
    squarey = cartesian2polar.nodes.new("CompositorNodeMath")
    squarey.name = "SquareY"
    squarey.operation = "POWER"
    squarey.use_clamp = False
    # Value_001
    squarey.inputs[1].default_value = 2.0

    # node Reroute
    reroute = cartesian2polar.nodes.new("NodeReroute")
    reroute.name = "Reroute"

    # Set locations
    group_output.location = (395.625, -57.550392150878906)
    group_input.location = (-401.875, 41.049617767333984)
    arctan2.location = (-186.875, -156.150390625)
    sqrt.location = (205.625, 41.04961013793945)
    sum.location = (15.625, 41.04961013793945)
    squarex.location = (-186.875, 239.849609375)
    squarey.location = (-186.875, 41.04961013793945)
    reroute.location = (345.625, -190.34727478027344)

    # initialize cartesian2polar links
    # sum.Value -> sqrt.Value
    cartesian2polar.links.new(sum.outputs[0], sqrt.inputs[0])
    # squarey.Value -> sum.Value
    cartesian2polar.links.new(squarey.outputs[0], sum.inputs[1])
    # squarex.Value -> sum.Value
    cartesian2polar.links.new(squarex.outputs[0], sum.inputs[0])
    # group_input.x -> squarex.Value
    cartesian2polar.links.new(group_input.outputs[0], squarex.inputs[0])
    # group_input.x -> arctan2.Value
    cartesian2polar.links.new(group_input.outputs[0], arctan2.inputs[1])
    # group_input.y -> squarey.Value
    cartesian2polar.links.new(group_input.outputs[1], squarey.inputs[0])
    # sqrt.Value -> group_output.r
    cartesian2polar.links.new(sqrt.outputs[0], group_output.inputs[0])
    # group_input.y -> arctan2.Value
    cartesian2polar.links.new(group_input.outputs[1], arctan2.inputs[0])
    # arctan2.Value -> reroute.Input
    cartesian2polar.links.new(arctan2.outputs[0], reroute.inputs[0])
    # reroute.Output -> group_output.theta
    cartesian2polar.links.new(reroute.outputs[0], group_output.inputs[1])
    return cartesian2polar


# initialize FlowDebug node group
def flowdebug_node_group():
    flowdebug = bpy.data.node_groups.new(type="CompositorNodeTree", name="FlowDebug")

    if bpy.app.version >= (4, 3, 0):
        flowdebug.default_group_node_width = 140

    # flowdebug interface
    # Socket Image
    new_socket(flowdebug, name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")

    # Socket Orientation
    new_socket(flowdebug, name="Orientation", in_out="INPUT", socket_type="NodeSocketFloat")

    # Socket x
    new_socket(flowdebug, name="x", in_out="INPUT", socket_type="NodeSocketFloat")

    # Socket y
    new_socket(flowdebug, name="y", in_out="INPUT", socket_type="NodeSocketFloat")

    # initialize flowdebug nodes
    # node Group Output
    group_output = flowdebug.nodes.new("NodeGroupOutput")
    group_output.name = "Group Output"
    group_output.is_active_output = True

    # node Group Input
    group_input = flowdebug.nodes.new("NodeGroupInput")
    group_input.name = "Group Input"

    # node Group
    group = flowdebug.nodes.new("CompositorNodeGroup")
    group.label = "Cartesian2Polar"
    group.name = "Group"
    group.node_tree = cartesian2polar_node_group()

    # node Normalize
    normalize = flowdebug.nodes.new("CompositorNodeNormalize")
    normalize.name = "Normalize"

    # node Map Range
    map_range = flowdebug.nodes.new("CompositorNodeMapRange")
    map_range.name = "Map Range"
    map_range.use_clamp = False
    # From Min
    map_range.inputs[1].default_value = -3.1415927410125732
    # From Max
    map_range.inputs[2].default_value = 3.1415927410125732
    # To Min
    map_range.inputs[3].default_value = 0.0
    # To Max
    map_range.inputs[4].default_value = 6.2831854820251465

    # node Combine Color
    if bpy.app.version >= (3, 3, 0):
        combine_color = flowdebug.nodes.new("CompositorNodeCombineColor")
        combine_color.mode = "HSV"
    else:
        combine_color = flowdebug.nodes.new("CompositorNodeCombHSVA")
    combine_color.name = "Combine Color"
    # Saturation
    combine_color.inputs[1].default_value = 1.0
    # Alpha
    combine_color.inputs[3].default_value = 1.0

    # node Orientation offset
    orientation_offset = flowdebug.nodes.new("CompositorNodeMath")
    orientation_offset.name = "Orientation offset"
    orientation_offset.operation = "ADD"
    orientation_offset.use_clamp = False

    # node Mod2pi
    mod2pi = flowdebug.nodes.new("CompositorNodeMath")
    mod2pi.name = "Mod2pi"
    mod2pi.operation = "MODULO"
    mod2pi.use_clamp = False
    # Value_001
    mod2pi.inputs[1].default_value = 6.2831854820251465

    # node HueNorm
    huenorm = flowdebug.nodes.new("CompositorNodeMapRange")
    huenorm.name = "HueNorm"
    huenorm.use_clamp = False
    # From Min
    huenorm.inputs[1].default_value = 0.0
    # From Max
    huenorm.inputs[2].default_value = 6.2831854820251465
    # To Min
    huenorm.inputs[3].default_value = 0.0
    # To Max
    huenorm.inputs[4].default_value = 1.0

    # node Reroute
    reroute_1 = flowdebug.nodes.new("NodeReroute")
    reroute_1.name = "Reroute"
    # node Reroute.001
    reroute_2 = flowdebug.nodes.new("NodeReroute")
    reroute_2.name = "Reroute.001"
    # node Reroute.002
    reroute_3 = flowdebug.nodes.new("NodeReroute")
    reroute_3.name = "Reroute.002"

    # Set locations
    group_output.location = (734.375, -46.430747985839844)
    group_input.location = (-633.125, -17.430749893188477)
    group.location = (-430.625, -46.430747985839844)
    normalize.location = (341.8750305175781, -164.23074340820312)
    map_range.location = (-240.625, 71.36925506591797)
    combine_color.location = (544.375, -46.430747985839844)
    orientation_offset.location = (-38.12500762939453, 71.36925506591797)
    mod2pi.location = (151.87496948242188, 71.36925506591797)
    huenorm.location = (341.8750305175781, 71.36925506591797)
    reroute_1.location = (-430.625, 129.3692626953125)
    reroute_2.location = (-100.625, 129.3692626953125)
    reroute_3.location = (-240.625, -223.26181030273438)

    # initialize flowdebug links
    # group_input.x -> group.x
    flowdebug.links.new(group_input.outputs[1], group.inputs[0])
    # group_input.y -> group.y
    flowdebug.links.new(group_input.outputs[2], group.inputs[1])
    # normalize.Value -> combine_color.Blue
    flowdebug.links.new(normalize.outputs[0], combine_color.inputs[2])
    # combine_color.Image -> group_output.Image
    flowdebug.links.new(combine_color.outputs[0], group_output.inputs[0])
    # orientation_offset.Value -> mod2pi.Value
    flowdebug.links.new(orientation_offset.outputs[0], mod2pi.inputs[0])
    # group.theta -> map_range.Value
    flowdebug.links.new(group.outputs[1], map_range.inputs[0])
    # huenorm.Value -> combine_color.Red
    flowdebug.links.new(huenorm.outputs[0], combine_color.inputs[0])
    # mod2pi.Value -> huenorm.Value
    flowdebug.links.new(mod2pi.outputs[0], huenorm.inputs[0])
    # map_range.Value -> orientation_offset.Value
    flowdebug.links.new(map_range.outputs[0], orientation_offset.inputs[1])
    # group_input.Orientation -> reroute_1.Input
    flowdebug.links.new(group_input.outputs[0], reroute_1.inputs[0])
    # reroute_1.Output -> reroute_2.Input
    flowdebug.links.new(reroute_1.outputs[0], reroute_2.inputs[0])
    # reroute_2.Output -> orientation_offset.Value
    flowdebug.links.new(reroute_2.outputs[0], orientation_offset.inputs[0])
    # group.r -> reroute_3.Input
    flowdebug.links.new(group.outputs[0], reroute_3.inputs[0])
    # reroute_3.Output -> normalize.Value
    flowdebug.links.new(reroute_3.outputs[0], normalize.inputs[0])
    return flowdebug


# initialize Vec2RGBA node group
def vec2rgba_node_group():
    vec2rgba = bpy.data.node_groups.new(type="CompositorNodeTree", name="Vec2RGBA")

    if bpy.app.version >= (4, 3, 0):
        vec2rgba.default_group_node_width = 140

    # vec2rgba interface
    # Socket Image
    new_socket(vec2rgba, name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")

    # Socket Image
    new_socket(vec2rgba, name="Image", in_out="INPUT", socket_type="NodeSocketColor")

    # initialize vec2rgba nodes
    # node Group Output
    group_output = vec2rgba.nodes.new("NodeGroupOutput")
    group_output.name = "Group Output"
    group_output.is_active_output = True

    # node Group Input
    group_input = vec2rgba.nodes.new("NodeGroupInput")
    group_input.name = "Group Input"

    # node Separate Color
    if bpy.app.version >= (3, 3, 0):
        separate_color = vec2rgba.nodes.new("CompositorNodeSeparateColor")
        separate_color.mode = "RGB"
    else:
        separate_color = vec2rgba.nodes.new("CompositorNodeSepRGBA")
    separate_color.name = "Separate Color"

    # node Combine Color
    if bpy.app.version >= (3, 3, 0):
        combine_color = vec2rgba.nodes.new("CompositorNodeCombineColor")
        combine_color.mode = "RGB"
    else:
        combine_color = vec2rgba.nodes.new("CompositorNodeCombRGBA")
    combine_color.name = "Combine Color"

    # Set locations
    group_output.location = (285.0, 0.0)
    group_input.location = (-285.0, 0.0)
    separate_color.location = (-95.0, 0.0)
    combine_color.location = (95.0, 0.0)

    # initialize vec2rgba links
    # separate_color.Green -> combine_color.Green
    vec2rgba.links.new(separate_color.outputs[1], combine_color.inputs[1])
    # separate_color.Alpha -> combine_color.Alpha
    vec2rgba.links.new(separate_color.outputs[3], combine_color.inputs[3])
    # separate_color.Red -> combine_color.Red
    vec2rgba.links.new(separate_color.outputs[0], combine_color.inputs[0])
    # separate_color.Blue -> combine_color.Blue
    vec2rgba.links.new(separate_color.outputs[2], combine_color.inputs[2])
    # group_input.Image -> separate_color.Image
    vec2rgba.links.new(group_input.outputs[0], separate_color.inputs[0])
    # combine_color.Image -> group_output.Image
    vec2rgba.links.new(combine_color.outputs[0], group_output.inputs[0])
    return vec2rgba
