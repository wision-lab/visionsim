# These were made in Blender and exported here as code
# using https://github.com/BrendanParmer/NodeToPython
# and then manually cleaned up/formatted.

# NOTE: This needs to be imported by blender to work properly.

import bpy  # type: ignore


# initialize Cartesian2Polar node group
def cartesian2polar_node_group():
    cartesian2polar = bpy.data.node_groups.new(type="CompositorNodeTree", name="Cartesian2Polar")

    cartesian2polar.color_tag = "NONE"
    cartesian2polar.description = ""
    cartesian2polar.default_group_node_width = 140

    # cartesian2polar interface
    # Socket r
    r_socket = cartesian2polar.interface.new_socket(name="r", in_out="OUTPUT", socket_type="NodeSocketFloat")
    r_socket.subtype = "NONE"
    r_socket.attribute_domain = "POINT"

    # Socket theta
    theta_socket = cartesian2polar.interface.new_socket(name="theta", in_out="OUTPUT", socket_type="NodeSocketFloat")
    theta_socket.subtype = "NONE"
    theta_socket.attribute_domain = "POINT"

    # Socket x
    x_socket = cartesian2polar.interface.new_socket(name="x", in_out="INPUT", socket_type="NodeSocketFloat")
    x_socket.subtype = "NONE"
    x_socket.attribute_domain = "POINT"

    # Socket y
    y_socket = cartesian2polar.interface.new_socket(name="y", in_out="INPUT", socket_type="NodeSocketFloat")
    y_socket.subtype = "NONE"
    y_socket.attribute_domain = "POINT"

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

    # Set locations
    group_output.location = (265.1455993652344, -8.908363342285156)
    group_input.location = (-489.42120361328125, -36.969505310058594)
    arctan2.location = (-74.99752044677734, -49.00311279296875)
    sqrt.location = (90.75543212890625, 104.71868896484375)
    sum.location = (-78.58831787109375, 115.163330078125)
    squarex.location = (-256.676513671875, 223.12020874023438)
    squarey.location = (-260.33575439453125, 60.74333572387695)

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
    # arctan2.Value -> group_output.theta
    cartesian2polar.links.new(arctan2.outputs[0], group_output.inputs[1])
    return cartesian2polar


# initialize FlowDebug node group
def flowdebug_node_group():
    flowdebug = bpy.data.node_groups.new(type="CompositorNodeTree", name="FlowDebug")

    flowdebug.color_tag = "NONE"
    flowdebug.description = ""
    flowdebug.default_group_node_width = 140

    # flowdebug interface
    # Socket Image
    image_socket = flowdebug.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    image_socket.attribute_domain = "POINT"

    # Socket Orientation
    orientation_socket = flowdebug.interface.new_socket(
        name="Orientation", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    orientation_socket.subtype = "NONE"
    orientation_socket.attribute_domain = "POINT"

    # Socket x
    x_socket = flowdebug.interface.new_socket(name="x", in_out="INPUT", socket_type="NodeSocketFloat")
    x_socket.subtype = "NONE"
    x_socket.attribute_domain = "POINT"

    # Socket y
    y_socket = flowdebug.interface.new_socket(name="y", in_out="INPUT", socket_type="NodeSocketFloat")
    y_socket.subtype = "NONE"
    y_socket.attribute_domain = "POINT"

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
    combine_color = flowdebug.nodes.new("CompositorNodeCombineColor")
    combine_color.name = "Combine Color"
    combine_color.mode = "HSV"
    combine_color.ycc_mode = "ITUBT709"
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

    # Set locations
    group_output.location = (923.6994018554688, -206.8037872314453)
    group_input.location = (-376.9529724121094, -134.94549560546875)
    group.location = (-179.69717407226562, -88.65217590332031)
    normalize.location = (337.347900390625, -236.9689483642578)
    map_range.location = (18.62582778930664, -72.21456146240234)
    combine_color.location = (744.4407348632812, -150.51651000976562)
    orientation_offset.location = (225.88455200195312, -65.74089813232422)
    mod2pi.location = (390.1499328613281, -67.56871032714844)
    huenorm.location = (552.42919921875, -54.257347106933594)

    # initialize flowdebug links
    # group.r -> normalize.Value
    flowdebug.links.new(group.outputs[0], normalize.inputs[0])
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
    # group_input.Orientation -> orientation_offset.Value
    flowdebug.links.new(group_input.outputs[0], orientation_offset.inputs[0])
    # map_range.Value -> orientation_offset.Value
    flowdebug.links.new(map_range.outputs[0], orientation_offset.inputs[1])
    return flowdebug


# initialize Vec2RGBA node group
def vec2rgba_node_group():
    vec2rgba = bpy.data.node_groups.new(type="CompositorNodeTree", name="Vec2RGBA")

    vec2rgba.color_tag = "NONE"
    vec2rgba.description = ""
    vec2rgba.default_group_node_width = 140

    # vec2rgba interface
    # Socket Image
    output_socket = vec2rgba.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    output_socket.attribute_domain = "POINT"

    # Socket Image
    input_socket = vec2rgba.interface.new_socket(name="Image", in_out="INPUT", socket_type="NodeSocketColor")
    input_socket.attribute_domain = "POINT"

    # initialize vec2rgba nodes
    # node Group Output
    group_output = vec2rgba.nodes.new("NodeGroupOutput")
    group_output.name = "Group Output"
    group_output.is_active_output = True

    # node Group Input
    group_input = vec2rgba.nodes.new("NodeGroupInput")
    group_input.name = "Group Input"

    # node Separate Color.001
    separate_color = vec2rgba.nodes.new("CompositorNodeSeparateColor")
    separate_color.name = "Separate Color"
    separate_color.mode = "RGB"
    separate_color.ycc_mode = "ITUBT709"

    # node Combine Color
    combine_color = vec2rgba.nodes.new("CompositorNodeCombineColor")
    combine_color.name = "Combine Color"
    combine_color.mode = "RGB"
    combine_color.ycc_mode = "ITUBT709"

    # Set locations
    group_output.location = (285.7762756347656, 0.0)
    group_input.location = (-295.7762756347656, 0.0)
    separate_color.location = (-95.77627563476562, -0.670013427734375)
    combine_color.location = (95.77627563476562, 0.66998291015625)

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
