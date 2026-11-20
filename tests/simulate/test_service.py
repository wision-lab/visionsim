from types import SimpleNamespace

import pytest

from visionsim.simulate.blender import BlenderService


def make_service():
    return SimpleNamespace(
        _initialized=True,
        scene=SimpleNamespace(render=SimpleNamespace(resolution_x=640, resolution_y=480, resolution_percentage=100)),
    )


def test_set_resolution_applies_keywords():
    service = make_service()

    BlenderService.exposed_set_resolution(service, height=720, width=1280, resolution_percentage=50)

    assert service.scene.render.resolution_y == 720
    assert service.scene.render.resolution_x == 1280
    assert service.scene.render.resolution_percentage == 50


def test_set_resolution_parses_tuple():
    service = make_service()

    BlenderService.exposed_set_resolution(service, height=(1080, 1920))

    assert service.scene.render.resolution_y == 1080
    assert service.scene.render.resolution_x == 1920


def test_set_resolution_leaves_unspecified_dimension():
    service = make_service()

    BlenderService.exposed_set_resolution(service, height=720)

    assert service.scene.render.resolution_y == 720
    assert service.scene.render.resolution_x == 640


def test_set_resolution_rejects_tuple_with_width():
    with pytest.raises(ValueError, match="tuple"):
        BlenderService.exposed_set_resolution(make_service(), height=(1080, 1920), width=640)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"height": 0},
        {"height": -1},
        {"width": 0},
        {"width": -1},
        {"resolution_percentage": 0},
        {"resolution_percentage": -5},
        {"height": (0, 640)},
        {"height": (480, 0)},
    ],
)
def test_set_resolution_rejects_non_positive_values(kwargs):
    with pytest.raises(ValueError, match="positive"):
        BlenderService.exposed_set_resolution(make_service(), **kwargs)


def test_skipped_render_does_not_apply_transient_frame(monkeypatch):
    """A frame whose outputs already exist must not refresh transient playback state."""
    applied = []
    monkeypatch.setattr("visionsim.simulate.blender.bpy", SimpleNamespace(app=SimpleNamespace(version=(5, 0, 0))))
    service = SimpleNamespace(
        _initialized=True,
        _thermal_transient={"root": "unused"},
        _apply_transient_frame=lambda: applied.append(1),
        scene=SimpleNamespace(frame_current=1),
        _outputs={},
        _warned_no_outputs=True,
        root_path="unused",
        _save_metadata=lambda **kwargs: None,
        exposed_camera_extrinsics=lambda: SimpleNamespace(tolist=list),
        exposed_camera_info=lambda: None,
    )

    BlenderService.exposed_render_current_frame(service, allow_skips=True)

    assert applied == []
