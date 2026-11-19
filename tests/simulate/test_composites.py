"""Compositor output selection across Blender's legacy and group-based APIs."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from visionsim.simulate import blender


class RenderLayers:
    bl_idname = "CompositorNodeRLayers"
    type = "R_LAYERS"
    inputs = ()


class GroupInput:
    bl_idname = "NodeGroupInput"
    inputs = ()


class Reroute:
    type = "REROUTE"

    def __init__(self, source):
        self.inputs = [socket_from(source)]


def socket_from(source):
    return SimpleNamespace(links=[SimpleNamespace(from_node=source, from_socket=SimpleNamespace(name="Image"))])


def service_with(monkeypatch, nodes, version=(5, 3, 0)):
    monkeypatch.setattr(
        blender,
        "bpy",
        SimpleNamespace(
            app=SimpleNamespace(version=version),
            types=SimpleNamespace(NodeReroute=Reroute, CompositorNodeRLayers=RenderLayers, NodeGroupInput=GroupInput),
        ),
    )
    return SimpleNamespace(
        _initialized=True,
        tree=SimpleNamespace(nodes=nodes),
        scene=SimpleNamespace(render=SimpleNamespace(image_settings=SimpleNamespace(color_mode="RGBA"))),
        log=Mock(),
        register_output_type=Mock(),
    )


@pytest.mark.parametrize("version", [(4, 5, 0), (5, 0, 0), (5, 3, 0)])
@pytest.mark.parametrize("source_type", [RenderLayers, GroupInput])
@pytest.mark.parametrize("reroute", [False, True])
def test_composites_accept_connected_output_without_other_nodes(monkeypatch, version, source_type, reroute):
    source = source_type()
    upstream = Reroute(source) if reroute else source
    output = SimpleNamespace(
        bl_idname="CompositorNodeComposite" if version < (5, 0, 0) else "NodeGroupOutput",
        inputs=[socket_from(upstream)],
        is_active_output=True,
    )
    service = service_with(monkeypatch, [source, output], version)

    blender.BlenderService.exposed_include_composites(service)

    service.register_output_type.assert_called_once_with("composites", object, object, c=4)
    service.log.warning.assert_called_once()


def test_composites_do_not_mistake_other_output_links_for_active_composite(monkeypatch):
    source = RenderLayers()
    active = SimpleNamespace(type="GROUP_OUTPUT", inputs=[SimpleNamespace(links=[])], is_active_output=True)
    inactive = SimpleNamespace(type="GROUP_OUTPUT", inputs=[socket_from(source)], is_active_output=False)
    viewer = SimpleNamespace(type="VIEWER", inputs=[socket_from(source)])
    service = service_with(monkeypatch, [source, active, inactive, viewer])

    blender.BlenderService.exposed_include_composites(service)
    service.log.warning.assert_called_once()
    service.register_output_type.assert_called_once()


def test_composites_warn_when_output_is_absent(monkeypatch):
    service = service_with(monkeypatch, [RenderLayers()])

    blender.BlenderService.exposed_include_composites(service)
    service.log.warning.assert_called_once()
    service.register_output_type.assert_called_once()


def test_composites_recognize_custom_processing(monkeypatch):
    processing = SimpleNamespace(type="BLUR", inputs=[socket_from(RenderLayers())])
    output = SimpleNamespace(type="GROUP_OUTPUT", inputs=[socket_from(processing)], is_active_output=True)
    service = service_with(monkeypatch, [processing, output])

    blender.BlenderService.exposed_include_composites(service)

    service.register_output_type.assert_called_once()
    service.log.warning.assert_not_called()
