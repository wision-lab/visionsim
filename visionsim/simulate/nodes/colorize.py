# These were made in Blender and exported here as code
# using https://github.com/BrendanParmer/NodeToPython
# and then manually cleaned up/formatted.

# NOTE: This needs to be imported by blender to work properly.

import bpy  # type: ignore

from .common import MAPRANGE_NODE, MATH_NODE, MIX_NODE, new_socket, set_clamp


# initialize ColorizeIndices node group
def colorize_indices_node_group(shade: bool = False):
    colorize_indices = bpy.data.node_groups.new(type="CompositorNodeTree", name="ColorizeIndices")

    if bpy.app.version >= (4, 3, 0):
        colorize_indices.default_group_node_width = 140

    # colorize_indices interface
    # Socket Image
    new_socket(colorize_indices, name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")

    # Socket Index
    new_socket(colorize_indices, name="Index", in_out="INPUT", socket_type="NodeSocketFloat")

    # Socket Normal
    new_socket(colorize_indices, name="Normal", in_out="INPUT", socket_type="NodeSocketVector")

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

    # Node Vector Math
    if bpy.app.version >= (5, 0, 0):
        vector_math = colorize_indices.nodes.new("ShaderNodeVectorMath")
        vector_math.operation = "DOT_PRODUCT"
    else:
        vector_math = colorize_indices.nodes.new("CompositorNodeNormal")
    vector_math.name = "Vector Math"

    # Node Mix
    mix = colorize_indices.nodes.new(MIX_NODE)
    mix.name = "Mix"
    mix.blend_type = "MULTIPLY"
    mix.inputs[0].default_value = 1.0
    mix_A = mix.inputs[6] if len(mix.inputs) > 3 else mix.inputs[1]
    mix_B = mix.inputs[7] if len(mix.inputs) > 3 else mix.inputs[2]

    # Node Math.001
    math_001 = colorize_indices.nodes.new(MATH_NODE)
    math_001.name = "Math.001"
    math_001.operation = "ABSOLUTE"
    set_clamp(math_001, False)

    # Node Switch
    switch = colorize_indices.nodes.new("CompositorNodeSwitch")
    switch.name = "Switch"

    # Node Reroute
    reroute = colorize_indices.nodes.new("NodeReroute")

    # Set locations
    group_output.location = (860.0057373046875, 146.49403381347656)
    group_input.location = (-164.99423217773438, 64.7919921875)
    combine_color.location = (252.50576782226562, 341.8763122558594)
    normalizeidx.location = (50.005767822265625, 500.87396240234375)
    math.location = (50.005767822265625, 207.67396545410156)
    vector_math.location = (50.005767822265625, -22.210357666015625)
    mix.location = (455.0057678222656, 146.49403381347656)
    math_001.location = (252.50576782226562, -22.210357666015625)
    switch.location = (670.0057373046875, 146.49403381347656)
    reroute.location = (595.0057373046875, 307.612060546875)

    # Initialize colorize_indices links
    # normalizeidx.Result -> combine_color.Red
    colorize_indices.links.new(normalizeidx.outputs[0], combine_color.inputs[0])
    # math.Value -> combine_color.Blue
    colorize_indices.links.new(math.outputs[0], combine_color.inputs[2])
    # group_input.Index -> normalizeidx.Value
    colorize_indices.links.new(group_input.outputs[0], normalizeidx.inputs[0])
    # group_input.Index -> math.Value
    colorize_indices.links.new(group_input.outputs[0], math.inputs[0])
    # group_input.Normal -> vector_math.Vector
    colorize_indices.links.new(group_input.outputs[1], vector_math.inputs[0])
    # combine_color.Image -> mix.A
    colorize_indices.links.new(combine_color.outputs[0], mix_A)
    # math_001.Value -> mix.B
    colorize_indices.links.new(math_001.outputs[0], mix_B)
    # vector_math.Value -> math_001.Value
    colorize_indices.links.new(vector_math.outputs[1], math_001.inputs[0])
    # switch.Image -> group_output.Image
    colorize_indices.links.new(switch.outputs[0], group_output.inputs[0])
    # combine_color.Image -> reroute.Input
    colorize_indices.links.new(combine_color.outputs[0], reroute.inputs[0])

    # It used to be ["Off", "On"], and `check` as an attribute, but now it's ["Switch", "On", "Off"]
    if not hasattr(switch, "check"):
        # mix.Result -> switch.On
        colorize_indices.links.new(mix.outputs[0], switch.inputs[2])
        # reroute.Output -> switch.Off
        colorize_indices.links.new(reroute.outputs[0], switch.inputs[1])
        switch.inputs[0].default_value = shade
    else:
        # mix.Result -> switch.On
        colorize_indices.links.new(mix.outputs[0], switch.inputs[1])
        # reroute.Output -> switch.Off
        colorize_indices.links.new(reroute.outputs[0], switch.inputs[0])
        switch.check = shade

    # Add drivers to vector_math node to use camera's optical axis as basis for shading
    if bpy.app.version >= (5, 0, 0):
        # Use second input of vector math node
        fcurves = vector_math.inputs[1].driver_add("default_value")
    else:
        fcurves = vector_math.outputs[0].driver_add("default_value")

    for row, fcurve in enumerate(fcurves):
        value = fcurve.driver.variables.new()
        value.type = "CONTEXT_PROP"
        value.name = "value"

        for target in value.targets:
            target.context_property = "ACTIVE_SCENE"
            target.data_path = f"camera.matrix_world[2][{row}]"

        # There's an absolute value after the dot product, so no need to
        # bother negating it to fix the bug that affects the normals debug group.
        fcurve.driver.expression = "value"
    return colorize_indices
