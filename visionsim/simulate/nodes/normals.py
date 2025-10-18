# These were made in Blender and exported here as code
# using https://github.com/BrendanParmer/NodeToPython
# and then manually cleaned up/formatted.

# NOTE: This needs to be imported by blender to work properly.

import bpy  # type: ignore

from .common import new_socket


# initialize NormalDebug node group
def normaldebug_node_group():
    normaldebug = bpy.data.node_groups.new(type="CompositorNodeTree", name="NormalDebug")

    if bpy.app.version >= (4, 3, 0):
        normaldebug.default_group_node_width = 140

    # normaldebug interface
    # Socket RGBA
    new_socket(normaldebug, name="RGBA", in_out="OUTPUT", socket_type="NodeSocketColor")

    # Socket Vector
    new_socket(normaldebug, name="Vector", in_out="OUTPUT", socket_type="NodeSocketVector")

    # Socket Normal
    new_socket(normaldebug, name="Normal", in_out="INPUT", socket_type="NodeSocketVector", subtype="DIRECTION")

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
    rotation.label = "Rows of Inverse Camera Rotation"
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

    # node Combine Color
    if bpy.app.version >= (3, 3, 0):
        combine_color = normaldebug.nodes.new("CompositorNodeCombineColor")
        combine_color.mode = "RGB"
    else:
        combine_color = normaldebug.nodes.new("CompositorNodeCombRGBA")
    combine_color.name = "Combine Color"
    # Alpha
    combine_color.inputs[3].default_value = 1.0

    # node DebugNormal
    debugnormal = normaldebug.nodes.new("NodeFrame")
    debugnormal.label = "Debug Normal"
    debugnormal.name = "DebugNormal"
    debugnormal.label_size = 20
    debugnormal.shrink = True

    # node Separate XYZ
    if bpy.app.version >= (3, 2, 0):
        separate_xyz = normaldebug.nodes.new("CompositorNodeSeparateXYZ")
    else:
        separate_xyz = normaldebug.nodes.new("CompositorNodeSepRGBA")
    separate_xyz.name = "Separate XYZ"

    # node Combine XYZ.002
    if bpy.app.version >= (3, 2, 0):
        combine_xyz = normaldebug.nodes.new("CompositorNodeCombineXYZ")
    else:
        combine_xyz = normaldebug.nodes.new("CompositorNodeCombRGBA")
    combine_xyz.name = "Combine XYZ.002"

    # node Reroute
    reroute_1 = normaldebug.nodes.new("NodeReroute")
    reroute_1.name = "Reroute"
    # node Reroute.001
    reroute_2 = normaldebug.nodes.new("NodeReroute")
    reroute_2.name = "Reroute.001"
    # Set parents
    rotrow1.parent = rotation
    rotrow2.parent = rotation
    rotrow3.parent = rotation
    remapx.parent = debugnormal
    remapy.parent = debugnormal
    remapz.parent = debugnormal
    combine_color.parent = debugnormal
    separate_xyz.parent = debugnormal
    combine_xyz.parent = rotation

    # Set locations
    group_output.location = (667.72314453125, -93.61640930175781)
    group_input.location = (-691.4769287109375, -27.496484756469727)
    rotrow1.location = (30.023101806640625, -32.533050537109375)
    rotrow2.location = (30.023101806640625, -244.13314819335938)
    rotrow3.location = (30.023101806640625, -455.7331237792969)
    rotation.location = (-489.20001220703125, 246.89999389648438)
    remapx.location = (244.82305908203125, -39.7083740234375)
    remapy.location = (244.82305908203125, -258.50830078125)
    remapz.location = (244.82305908203125, -477.308349609375)
    combine_color.location = (447.32305908203125, -333.3519287109375)
    debugnormal.location = (-24.399999618530273, 444.0)
    separate_xyz.location = (29.82306480407715, -353.8504638671875)
    combine_xyz.location = (232.52313232421875, -287.1443786621094)
    reroute_1.location = (5.423047065734863, -302.10833740234375)
    reroute_2.location = (562.9230346679688, -302.10833740234375)

    # initialize normaldebug links
    # remapx.Value -> combine_color.Red
    normaldebug.links.new(remapx.outputs[0], combine_color.inputs[0])
    # remapy.Value -> combine_color.Green
    normaldebug.links.new(remapy.outputs[0], combine_color.inputs[1])
    # remapz.Value -> combine_color.Blue
    normaldebug.links.new(remapz.outputs[0], combine_color.inputs[2])
    # group_input.Normal -> rotrow2.Normal
    normaldebug.links.new(group_input.outputs[0], rotrow2.inputs[0])
    # group_input.Normal -> rotrow1.Normal
    normaldebug.links.new(group_input.outputs[0], rotrow1.inputs[0])
    # group_input.Normal -> rotrow3.Normal
    normaldebug.links.new(group_input.outputs[0], rotrow3.inputs[0])
    # combine_color.Image -> group_output.RGBA
    normaldebug.links.new(combine_color.outputs[0], group_output.inputs[0])
    # separate_xyz.X -> remapx.Value
    normaldebug.links.new(separate_xyz.outputs[0], remapx.inputs[0])
    # separate_xyz.Z -> remapz.Value
    normaldebug.links.new(separate_xyz.outputs[2], remapz.inputs[0])
    # separate_xyz.Y -> remapy.Value
    normaldebug.links.new(separate_xyz.outputs[1], remapy.inputs[0])
    # rotrow1.Dot -> combine_xyz.X
    normaldebug.links.new(rotrow1.outputs[1], combine_xyz.inputs[0])
    # rotrow2.Dot -> combine_xyz.Y
    normaldebug.links.new(rotrow2.outputs[1], combine_xyz.inputs[1])
    # rotrow3.Dot -> combine_xyz.Z
    normaldebug.links.new(rotrow3.outputs[1], combine_xyz.inputs[2])
    # combine_xyz.Vector -> separate_xyz.Vector
    normaldebug.links.new(combine_xyz.outputs[0], separate_xyz.inputs[0])
    # combine_xyz.Vector -> reroute_1.Input
    normaldebug.links.new(combine_xyz.outputs[0], reroute_1.inputs[0])
    # reroute_1.Output -> reroute_2.Input
    normaldebug.links.new(reroute_1.outputs[0], reroute_2.inputs[0])
    # reroute_2.Output -> group_output.Vector
    normaldebug.links.new(reroute_2.outputs[0], group_output.inputs[1])

    # Register drivers to auto-update camera matrix inverse.
    # For each element in each RotRows, we create a new driver input variable
    # link it to the camera world matrix, and set its value. This enables us
    # to use non-python drivers, meaning we do not need to save, and run a py
    # script on startup and autoexec does not need to be true!
    #
    # Notes:
    #   1) Blender's normals node actually returns the negative dot product (which may be a bug)
    #      so we negate that here. See: https://projects.blender.org/blender/blender/issues/132770
    #   2) We need the inverse of the camera matrix, it's a rotation so it's the same as it's
    #      transpose, which we do here by flipping the i/j indices we sample from.
    # TODO: Negation of dot product might become version specific once the bug is fixed.
    for row, node in enumerate([rotrow1, rotrow2, rotrow3]):
        fcurves = node.outputs[0].driver_add("default_value")
        for col, fcurve in enumerate(fcurves):
            mat = fcurve.driver.variables.new()
            mat.name = "camera_world_matrix"
            mat.targets[0].id = bpy.context.scene.camera
            mat.targets[0].data_path = "matrix_world"
            fcurve.driver.expression = f"-camera_world_matrix[{col}][{row}]"
    return normaldebug
