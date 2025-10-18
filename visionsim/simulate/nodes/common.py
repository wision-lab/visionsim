# NOTE: This needs to be imported by blender to work properly.

import bpy  # type: ignore


def new_socket(nodegroup, *, name, in_out, socket_type, attribute_domain="POINT", subtype=None):
    if bpy.app.version < (4, 0, 0):
        if in_out == "INPUT":
            socket = nodegroup.inputs.new(name=name, type=socket_type)
        elif in_out == "OUTPUT":
            socket = nodegroup.outputs.new(name=name, type=socket_type)
        else:
            raise ValueError("Unknown in/out type!")
    else:
        socket = nodegroup.interface.new_socket(name=name, in_out=in_out, socket_type=socket_type)

    # Subtypes weren't supported in 3.5 and under
    if subtype:
        if bpy.app.version >= (4, 0, 0):
            socket.subtype = subtype
        elif bpy.app.version >= (3, 6, 0):
            socket.bl_subtype_label = subtype

    # Attribute domains didn't exist before v3.0
    if bpy.app.version >= (3, 0, 0):
        socket.attribute_domain = attribute_domain
    return socket
