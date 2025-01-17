# These were made in Blender and exported here as code
# using https://github.com/BrendanParmer/NodeToPython
# and then manually cleaned up/formatted.

# NOTE: This needs to be imported by blender to work properly.

import bpy


# initialize NormalDebug node group
def normaldebug_node_group():
    normaldebug = bpy.data.node_groups.new(type="CompositorNodeTree", name="NormalDebug")

    # normaldebug interface
    # Socket RGBA
    rgba_socket = normaldebug.interface.new_socket(name="RGBA", in_out="OUTPUT", socket_type="NodeSocketColor")
    rgba_socket.attribute_domain = "POINT"

    # Socket Vector
    vector_socket = normaldebug.interface.new_socket(name="Vector", in_out="OUTPUT", socket_type="NodeSocketVector")
    vector_socket.subtype = "NONE"
    vector_socket.attribute_domain = "POINT"

    # Socket Normal
    normal_socket = normaldebug.interface.new_socket(name="Normal", in_out="INPUT", socket_type="NodeSocketVector")
    normal_socket.subtype = "DIRECTION"
    normal_socket.attribute_domain = "POINT"

    # initialize normaldebug nodes
    # node Group Output
    group_output = normaldebug.nodes.new("NodeGroupOutput")
    group_output.name = "Group Output"
    group_output.is_active_output = True

    # node Group Input
    group_input = normaldebug.nodes.new("NodeGroupInput")
    group_input.name = "Group Input"

    # node RotRow1
    rotrow1 = normaldebug.nodes.new("CompositorNodeNormal")
    rotrow1.name = "RotRow1"

    # node RotRow2
    rotrow2 = normaldebug.nodes.new("CompositorNodeNormal")
    rotrow2.name = "RotRow2"

    # node RotRow3
    rotrow3 = normaldebug.nodes.new("CompositorNodeNormal")
    rotrow3.name = "RotRow3"

    # node Rotation
    rotation = normaldebug.nodes.new("NodeFrame")
    rotation.label = "Rows of Camera Rotation"
    rotation.name = "Rotation"
    rotation.label_size = 14
    rotation.shrink = True

    # node RemapX
    remapx = normaldebug.nodes.new("CompositorNodeMath")
    remapx.name = "RemapX"
    remapx.operation = "MULTIPLY_ADD"
    remapx.use_clamp = True
    # Value_001
    remapx.inputs[1].default_value = 0.5
    # Value_002
    remapx.inputs[2].default_value = 0.5

    # node RemapY
    remapy = normaldebug.nodes.new("CompositorNodeMath")
    remapy.name = "RemapY"
    remapy.operation = "MULTIPLY_ADD"
    remapy.use_clamp = True
    # Value_001
    remapy.inputs[1].default_value = 0.5
    # Value_002
    remapy.inputs[2].default_value = 0.5

    # node RemapZ
    remapz = normaldebug.nodes.new("CompositorNodeMath")
    remapz.name = "RemapZ"
    remapz.operation = "MULTIPLY_ADD"
    remapz.use_clamp = True
    # Value_001
    remapz.inputs[1].default_value = 0.5
    # Value_002
    remapz.inputs[2].default_value = 0.5

    # node Colorize
    colorize = normaldebug.nodes.new("NodeFrame")
    colorize.name = "Colorize"
    colorize.label_size = 14
    colorize.shrink = True

    # node Combine Color
    combine_color = normaldebug.nodes.new("CompositorNodeCombineColor")
    combine_color.name = "Combine Color"
    combine_color.mode = "RGB"
    combine_color.ycc_mode = "ITUBT709"
    # Alpha
    combine_color.inputs[3].default_value = 1.0

    # node Combine XYZ
    combine_xyz = normaldebug.nodes.new("CompositorNodeCombineXYZ")
    combine_xyz.name = "Combine XYZ"

    # node Math
    math = normaldebug.nodes.new("CompositorNodeMath")
    math.name = "Math"
    math.operation = "ADD"
    math.use_clamp = False
    # Value
    math.inputs[0].default_value = 0.5
    # Value_001
    math.inputs[1].default_value = 0.5

    # Set parents
    rotrow1.parent = rotation
    rotrow2.parent = rotation
    rotrow3.parent = rotation
    remapx.parent = colorize
    remapy.parent = colorize
    remapz.parent = colorize
    combine_color.parent = colorize

    # initialize normaldebug links
    # remapx.Value -> combine_color_1.Red
    normaldebug.links.new(remapx.outputs[0], combine_color.inputs[0])
    # remapy.Value -> combine_color_1.Green
    normaldebug.links.new(remapy.outputs[0], combine_color.inputs[1])
    # remapz.Value -> combine_color_1.Blue
    normaldebug.links.new(remapz.outputs[0], combine_color.inputs[2])
    # rotrow1.Dot -> remapx.Value
    normaldebug.links.new(rotrow1.outputs[1], remapx.inputs[0])
    # rotrow2.Dot -> remapy.Value
    normaldebug.links.new(rotrow2.outputs[1], remapy.inputs[0])
    # rotrow3.Dot -> remapz.Value
    normaldebug.links.new(rotrow3.outputs[1], remapz.inputs[0])
    # rotrow1.Dot -> combine_xyz.X
    normaldebug.links.new(rotrow1.outputs[1], combine_xyz.inputs[0])
    # rotrow2.Dot -> combine_xyz.Y
    normaldebug.links.new(rotrow2.outputs[1], combine_xyz.inputs[1])
    # rotrow3.Dot -> combine_xyz.Z
    normaldebug.links.new(rotrow3.outputs[1], combine_xyz.inputs[2])
    # group_input.Normal -> rotrow2.Normal
    normaldebug.links.new(group_input.outputs[0], rotrow2.inputs[0])
    # group_input.Normal -> rotrow1.Normal
    normaldebug.links.new(group_input.outputs[0], rotrow1.inputs[0])
    # group_input.Normal -> rotrow3.Normal
    normaldebug.links.new(group_input.outputs[0], rotrow3.inputs[0])
    # combine_color.Image -> group_output.RGBA
    normaldebug.links.new(combine_color.outputs[0], group_output.inputs[0])
    # combine_xyz.Vector -> group_output.Vector
    normaldebug.links.new(combine_xyz.outputs[0], group_output.inputs[1])
    return normaldebug
