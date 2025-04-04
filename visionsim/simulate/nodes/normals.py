# These were made in Blender and exported here as code
# using https://github.com/BrendanParmer/NodeToPython
# and then manually cleaned up/formatted.

# NOTE: This needs to be imported by blender to work properly.

import bpy  # type: ignore


# initialize NormalDebug node group
def normaldebug_node_group():
    normaldebug = bpy.data.node_groups.new(type="CompositorNodeTree", name="NormalDebug")

    normaldebug.color_tag = "NONE"
    normaldebug.description = ""
    normaldebug.default_group_node_width = 140

    # normaldebug interface
    # Socket RGBA
    rgba_socket = normaldebug.interface.new_socket(name="RGBA", in_out="OUTPUT", socket_type="NodeSocketColor")
    rgba_socket.default_value = (0.0, 0.0, 0.0, 1.0)
    rgba_socket.attribute_domain = "POINT"

    # Socket Vector
    vector_socket = normaldebug.interface.new_socket(name="Vector", in_out="OUTPUT", socket_type="NodeSocketVector")
    vector_socket.default_value = (0.0, 0.0, 0.0)
    vector_socket.min_value = -3.4028234663852886e38
    vector_socket.max_value = 3.4028234663852886e38
    vector_socket.subtype = "NONE"
    vector_socket.attribute_domain = "POINT"

    # Socket Normal
    normal_socket = normaldebug.interface.new_socket(name="Normal", in_out="INPUT", socket_type="NodeSocketVector")
    normal_socket.default_value = (0.0, 0.0, 0.0)
    normal_socket.min_value = -3.4028234663852886e38
    normal_socket.max_value = 3.4028234663852886e38
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

    rotrow1.outputs[0].default_value = (-0.31883853673934937, 0.23669059574604034, 0.0002839671797119081)
    # node RotRow2
    rotrow2 = normaldebug.nodes.new("CompositorNodeNormal")
    rotrow2.name = "RotRow2"

    rotrow2.outputs[0].default_value = (-0.0801093578338623, -0.10746438056230545, -0.3737839460372925)
    # node RotRow3
    rotrow3 = normaldebug.nodes.new("CompositorNodeNormal")
    rotrow3.name = "RotRow3"

    rotrow3.outputs[0].default_value = (0.22272183001041412, 0.30018243193626404, -0.13403739035129547)
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
    combine_color = normaldebug.nodes.new("CompositorNodeCombineColor")
    combine_color.name = "Combine Color"
    combine_color.mode = "RGB"
    combine_color.ycc_mode = "ITUBT709"
    # Alpha
    combine_color.inputs[3].default_value = 1.0

    # node DebugNormal
    debugnormal = normaldebug.nodes.new("NodeFrame")
    debugnormal.label = "Debug Normal"
    debugnormal.name = "DebugNormal"
    debugnormal.label_size = 20
    debugnormal.shrink = True

    # node Separate XYZ
    separate_xyz = normaldebug.nodes.new("CompositorNodeSeparateXYZ")
    separate_xyz.name = "Separate XYZ"

    # node Reroute.004
    reroute = normaldebug.nodes.new("NodeReroute")
    reroute.name = "Reroute.004"
    reroute.socket_idname = "NodeSocketVector"
    # node Combine XYZ.002
    combine_xyz = normaldebug.nodes.new("CompositorNodeCombineXYZ")
    combine_xyz.name = "Combine XYZ.002"

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
    group_output.location = (1266.0460205078125, 1144.85546875)
    group_input.location = (-300.0, 1080.0)
    rotrow1.location = (30.3599853515625, -32.5997314453125)
    rotrow2.location = (30.3599853515625, -212.5997314453125)
    rotrow3.location = (30.3599853515625, -392.5997314453125)
    rotation.location = (39.599998474121094, 1359.699951171875)
    remapx.location = (232.80599975585938, -39.957275390625)
    remapy.location = (230.58395385742188, -226.56103515625)
    remapz.location = (230.58395385742188, -406.56103515625)
    combine_color.location = (430.9986267089844, -194.6773681640625)
    debugnormal.location = (487.6000061035156, 1371.199951171875)
    separate_xyz.location = (30.207672119140625, -179.9708251953125)
    reroute.location = (800.0, 740.0)
    combine_xyz.location = (200.4000244140625, -239.8974609375)

    # Set dimensions
    group_output.width, group_output.height = 140.0, 100.0
    group_input.width, group_input.height = 140.0, 100.0
    rotrow1.width, rotrow1.height = 140.0, 100.0
    rotrow2.width, rotrow2.height = 140.0, 100.0
    rotrow3.width, rotrow3.height = 140.0, 100.0
    rotation.width, rotation.height = 370.3999938964844, 584.0999755859375
    remapx.width, remapx.height = 140.0, 100.0
    remapy.width, remapy.height = 140.0, 100.0
    remapz.width, remapz.height = 140.0, 100.0
    combine_color.width, combine_color.height = 140.0, 100.0
    debugnormal.width, debugnormal.height = 600.800048828125, 605.199951171875
    separate_xyz.width, separate_xyz.height = 140.0, 100.0
    reroute.width, reroute.height = 12.5, 100.0
    combine_xyz.width, combine_xyz.height = 140.0, 100.0

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
    # combine_xyz.Vector -> reroute.Input
    normaldebug.links.new(combine_xyz.outputs[0], reroute.inputs[0])
    # rotrow1.Dot -> combine_xyz.X
    normaldebug.links.new(rotrow1.outputs[1], combine_xyz.inputs[0])
    # rotrow2.Dot -> combine_xyz.Y
    normaldebug.links.new(rotrow2.outputs[1], combine_xyz.inputs[1])
    # rotrow3.Dot -> combine_xyz.Z
    normaldebug.links.new(rotrow3.outputs[1], combine_xyz.inputs[2])
    # reroute.Output -> group_output.Vector
    normaldebug.links.new(reroute.outputs[0], group_output.inputs[1])
    # combine_xyz.Vector -> separate_xyz.Vector
    normaldebug.links.new(combine_xyz.outputs[0], separate_xyz.inputs[0])

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
