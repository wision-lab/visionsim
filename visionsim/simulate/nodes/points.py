# These were made in Blender and exported here as code
# using https://github.com/BrendanParmer/NodeToPython
# and then manually cleaned up/formatted.

# NOTE: This needs to be imported by blender to work properly.

import bpy  # type: ignore

from .common import new_socket, set_clamp


def point_preview_node_group():
    """Initialize Preview Points node group"""
    pointsdebug = bpy.data.node_groups.new(type="CompositorNodeTree", name="Preview Points")

    if bpy.app.version >= (4, 3, 0):
        pointsdebug.default_group_node_width = 140

    # pointsdebug interface
    # Socket Vector
    new_socket(pointsdebug, name="Vector", in_out="OUTPUT", socket_type="NodeSocketVector")

    # Socket Vector
    new_socket(pointsdebug, name="Vector", in_out="INPUT", socket_type="NodeSocketVector")

    # initialize pointsdebug nodes
    # Node Group Output
    group_output = pointsdebug.nodes.new("NodeGroupOutput")
    group_output.name = "Group Output"
    group_output.is_active_output = True

    # Node Group Input
    group_input = pointsdebug.nodes.new("NodeGroupInput")
    group_input.name = "Group Input"

    # Use Vector Math node if available
    if bpy.app.version >= (5, 0, 0):
        abs_node = pointsdebug.nodes.new("ShaderNodeVectorMath")
        abs_node.operation = "ABSOLUTE"
        pointsdebug.links.new(group_input.outputs[0], abs_node.inputs[0])

        frac_node = pointsdebug.nodes.new("ShaderNodeVectorMath")
        frac_node.operation = "FRACTION"
        pointsdebug.links.new(abs_node.outputs[0], frac_node.inputs[0])
        pointsdebug.links.new(frac_node.outputs[0], group_output.inputs[0])
        return pointsdebug

    # Node Math
    math = pointsdebug.nodes.new("CompositorNodeMath")
    math.name = "Math"
    math.operation = "FRACT"
    set_clamp(math, False)

    # Node Math.001
    math_001 = pointsdebug.nodes.new("CompositorNodeMath")
    math_001.name = "Math.001"
    math_001.operation = "FRACT"
    set_clamp(math_001, False)

    # Node Math.002
    math_002 = pointsdebug.nodes.new("CompositorNodeMath")
    math_002.name = "Math.002"
    math_002.operation = "FRACT"
    set_clamp(math_002, False)

    # Node Separate XYZ
    separate_xyz = pointsdebug.nodes.new("CompositorNodeSeparateXYZ")
    separate_xyz.name = "Separate XYZ"

    # Node Math.003
    math_003 = pointsdebug.nodes.new("CompositorNodeMath")
    math_003.name = "Math.003"
    math_003.operation = "ABSOLUTE"
    set_clamp(math_003, False)

    # Node Math.004
    math_004 = pointsdebug.nodes.new("CompositorNodeMath")
    math_004.name = "Math.004"
    math_004.operation = "ABSOLUTE"
    set_clamp(math_004, False)

    # Node Math.005
    math_005 = pointsdebug.nodes.new("CompositorNodeMath")
    math_005.name = "Math.005"
    math_005.operation = "ABSOLUTE"
    set_clamp(math_005, False)

    # Node Combine XYZ
    combine_xyz = pointsdebug.nodes.new("CompositorNodeCombineXYZ")
    combine_xyz.name = "Combine XYZ"

    # Set locations
    pointsdebug.nodes["Group Output"].location = (460.0, 0.0)
    pointsdebug.nodes["Group Input"].location = (-480.0, 0.0)
    pointsdebug.nodes["Math"].location = (80.0, 160.0)
    pointsdebug.nodes["Math.001"].location = (80.0, 20.0)
    pointsdebug.nodes["Math.002"].location = (80.0, -120.0)
    pointsdebug.nodes["Separate XYZ"].location = (-300.0, 20.0)
    pointsdebug.nodes["Math.003"].location = (-100.0, 160.0)
    pointsdebug.nodes["Math.004"].location = (-100.0, 20.0)
    pointsdebug.nodes["Math.005"].location = (-100.0, -120.0)
    pointsdebug.nodes["Combine XYZ"].location = (280.0, 20.0)

    # Initialize pointsdebug links
    # math_005.Value -> math_002.Value
    pointsdebug.links.new(pointsdebug.nodes["Math.005"].outputs[0], pointsdebug.nodes["Math.002"].inputs[0])
    # separate_xyz.X -> math_003.Value
    pointsdebug.links.new(pointsdebug.nodes["Separate XYZ"].outputs[0], pointsdebug.nodes["Math.003"].inputs[0])
    # math_002.Value -> combine_xyz.Z
    pointsdebug.links.new(pointsdebug.nodes["Math.002"].outputs[0], pointsdebug.nodes["Combine XYZ"].inputs[2])
    # math.Value -> combine_xyz.X
    pointsdebug.links.new(pointsdebug.nodes["Math"].outputs[0], pointsdebug.nodes["Combine XYZ"].inputs[0])
    # math_003.Value -> math.Value
    pointsdebug.links.new(pointsdebug.nodes["Math.003"].outputs[0], pointsdebug.nodes["Math"].inputs[0])
    # math_001.Value -> combine_xyz.Y
    pointsdebug.links.new(pointsdebug.nodes["Math.001"].outputs[0], pointsdebug.nodes["Combine XYZ"].inputs[1])
    # math_004.Value -> math_001.Value
    pointsdebug.links.new(pointsdebug.nodes["Math.004"].outputs[0], pointsdebug.nodes["Math.001"].inputs[0])
    # separate_xyz.Z -> math_005.Value
    pointsdebug.links.new(pointsdebug.nodes["Separate XYZ"].outputs[2], pointsdebug.nodes["Math.005"].inputs[0])
    # separate_xyz.Y -> math_004.Value
    pointsdebug.links.new(pointsdebug.nodes["Separate XYZ"].outputs[1], pointsdebug.nodes["Math.004"].inputs[0])
    # group_input.Vector -> separate_xyz.Vector
    pointsdebug.links.new(pointsdebug.nodes["Group Input"].outputs[0], pointsdebug.nodes["Separate XYZ"].inputs[0])
    # combine_xyz.Vector -> group_output.Vector
    pointsdebug.links.new(pointsdebug.nodes["Combine XYZ"].outputs[0], pointsdebug.nodes["Group Output"].inputs[0])

    return pointsdebug
