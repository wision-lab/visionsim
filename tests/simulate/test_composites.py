"""Compositor output selection across Blender's legacy and group-based APIs."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from visionsim.simulate import blender


class RenderLayers:
    type = "R_LAYERS"
    inputs = ()


class Reroute:
    type = "REROUTE"

    def __init__(self, source):
        self.inputs = [socket_from(source)]


def socket_from(source):
    return SimpleNamespace(links=[SimpleNamespace(from_node=source, from_socket=SimpleNamespace(name="Image"))])


def service_with(monkeypatch, nodes):
    monkeypatch.setattr(
        blender, "bpy", SimpleNamespace(types=SimpleNamespace(NodeReroute=Reroute, CompositorNodeRLayers=RenderLayers))
    )
    return SimpleNamespace(
        _initialized=True,
        tree=SimpleNamespace(nodes=nodes),
        scene=SimpleNamespace(render=SimpleNamespace(image_settings=SimpleNamespace(color_mode="RGBA"))),
        log=Mock(),
        register_output_type=Mock(),
    )


@pytest.mark.parametrize("output_type", ["COMPOSITE", "GROUP_OUTPUT"])
@pytest.mark.parametrize("reroute", [False, True])
def test_composites_accept_connected_output_without_other_nodes(monkeypatch, output_type, reroute):
    source = RenderLayers()
    upstream = Reroute(source) if reroute else source
    output = SimpleNamespace(type=output_type, inputs=[socket_from(upstream)], is_active_output=True)
    service = service_with(monkeypatch, [source, output])

    blender.BlenderService.exposed_include_composites(service)

    service.register_output_type.assert_called_once_with("composites", object, object, c=4)
    service.log.warning.assert_called_once()


def test_composites_do_not_mistake_other_output_links_for_active_composite(monkeypatch):
    source = RenderLayers()
    active = SimpleNamespace(type="GROUP_OUTPUT", inputs=[SimpleNamespace(links=[])], is_active_output=True)
    inactive = SimpleNamespace(type="GROUP_OUTPUT", inputs=[socket_from(source)], is_active_output=False)
    viewer = SimpleNamespace(type="VIEWER", inputs=[socket_from(source)])
    service = service_with(monkeypatch, [source, active, inactive, viewer])

    with pytest.raises(RuntimeError, match="nothing is connected"):
        blender.BlenderService.exposed_include_composites(service)
    service.register_output_type.assert_not_called()


def test_composites_require_actual_output_node(monkeypatch):
    service = service_with(monkeypatch, [RenderLayers()])

    with pytest.raises(RuntimeError, match="no active"):
        blender.BlenderService.exposed_include_composites(service)


def test_composites_recognize_custom_processing(monkeypatch):
    processing = SimpleNamespace(type="BLUR", inputs=[socket_from(RenderLayers())])
    output = SimpleNamespace(type="GROUP_OUTPUT", inputs=[socket_from(processing)], is_active_output=True)
    service = service_with(monkeypatch, [processing, output])

    blender.BlenderService.exposed_include_composites(service)

    service.register_output_type.assert_called_once()
    service.log.warning.assert_not_called()
