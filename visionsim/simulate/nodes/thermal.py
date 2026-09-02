# NOTE: This needs to be imported by blender to work properly.

# Inferno colormap stop data sourced verbatim from heat-sim-blender's
# scripts/post/_exr_io.py ``_INFERNO_STOPS`` (matplotlib inferno sampled at 11
# even t values), so the preview matches heat-sim's rendered To/thermal PNGs.

from __future__ import annotations

import bpy  # type: ignore

from .common import MAPRANGE_NODE, new_socket, set_clamp


def thermal_preview_node_group(tmin: float = 295.0, tmax: float = 400.0) -> bpy.types.NodeTree:
    """Compositor node group mapping a scalar temperature (K) to an inferno-coloured RGB image.

    Structure: ``Temperature`` float INPUT → MapRange([tmin, tmax] → [0, 1]) →
    ``ShaderNodeValToRGB`` with inferno colour stops → ``Image`` colour OUTPUT.

    The inferno stops match heat-sim-blender's colormap. heat-sim treats the LUT
    output as scene-linear and sRGB-encodes it before writing the PNG; the caller
    (``include_thermal``) reproduces that by rendering this group's output through
    the ``Standard`` (sRGB) view transform rather than the preview default ``Raw``.

    Args:
        tmin: Minimum temperature in Kelvin — maps to the cool (near-black) end of inferno.
        tmax: Maximum temperature in Kelvin — maps to the hot (bright yellow) end of inferno.

    Returns:
        A ``CompositorNodeTree`` named ``"ThermalPreview"``.
    """
    ng = bpy.data.node_groups.new(type="CompositorNodeTree", name="ThermalPreview")

    if bpy.app.version >= (4, 3, 0):
        ng.default_group_node_width = 140

    # -- Sockets -----------------------------------------------------------------
    # Output first so the interface order matches the socket panel.
    new_socket(ng, name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    new_socket(ng, name="Temperature", in_out="INPUT", socket_type="NodeSocketFloat")

    # -- Nodes -------------------------------------------------------------------
    group_input = ng.nodes.new("NodeGroupInput")
    group_input.name = "Group Input"

    group_output = ng.nodes.new("NodeGroupOutput")
    group_output.name = "Group Output"
    group_output.is_active_output = True

    # MapRange: Temperature (K) → [0, 1]
    map_range = ng.nodes.new(MAPRANGE_NODE)
    map_range.name = "TempNormalize"
    set_clamp(map_range, True)
    map_range.inputs[1].default_value = float(tmin)   # From Min
    map_range.inputs[2].default_value = float(tmax)   # From Max
    map_range.inputs[3].default_value = 0.0           # To Min
    map_range.inputs[4].default_value = 1.0           # To Max

    # Inferno colour ramp (ShaderNodeValToRGB is correct for the Blender ≥5.0 compositor).
    color_ramp = ng.nodes.new("ShaderNodeValToRGB")
    color_ramp.name = "InfernoRamp"
    ramp = color_ramp.color_ramp
    ramp.interpolation = "LINEAR"

    # Inferno colormap stops — verbatim from heat-sim-blender _exr_io._INFERNO_STOPS
    # (matplotlib inferno at 11 even positions 0.0, 0.1, …, 1.0).
    _inferno_stops = [
        (0.00, (0.00146, 0.00047, 0.01387, 1.0)),  # near-black
        (0.10, (0.08741, 0.04456, 0.22481, 1.0)),  # deep purple
        (0.20, (0.25823, 0.03857, 0.40648, 1.0)),  # purple
        (0.30, (0.41633, 0.09020, 0.43294, 1.0)),  # magenta
        (0.40, (0.57830, 0.14804, 0.40441, 1.0)),  # red-magenta
        (0.50, (0.73568, 0.21591, 0.33025, 1.0)),  # red
        (0.60, (0.86501, 0.31682, 0.22606, 1.0)),  # red-orange
        (0.70, (0.95451, 0.46874, 0.09987, 1.0)),  # orange
        (0.80, (0.98762, 0.64532, 0.03989, 1.0)),  # amber
        (0.90, (0.96439, 0.84385, 0.27339, 1.0)),  # yellow
        (1.00, (0.98836, 0.99836, 0.64492, 1.0)),  # pale yellow
    ]

    # Mirror colorize_indices_node_group: clear existing, resize, then fill.
    while len(ramp.elements) < len(_inferno_stops):
        ramp.elements.new(0.5)
    while len(ramp.elements) > len(_inferno_stops):
        ramp.elements.remove(ramp.elements[-1])
    for i, (pos, color) in enumerate(_inferno_stops):
        ramp.elements[i].position = pos
        ramp.elements[i].color = color

    # -- Locations ---------------------------------------------------------------
    group_input.location = (-300.0, 0.0)
    map_range.location = (-100.0, 0.0)
    color_ramp.location = (150.0, 0.0)
    group_output.location = (500.0, 0.0)

    # -- Links -------------------------------------------------------------------
    ng.links.new(group_input.outputs[0], map_range.inputs[0])
    ng.links.new(map_range.outputs[0], color_ramp.inputs[0])
    ng.links.new(color_ramp.outputs[0], group_output.inputs[0])

    return ng
