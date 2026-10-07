from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from tests.heatsim_test_service import SERVICE_PATH, call_service
from visionsim.simulate.blender import BlenderClient
from visionsim.simulate.heatsim import cache


def test_cache_roundtrip_and_miss(tmp_path):
    key = cache.cache_key(tmp_path / "scene.blend", {"dt": 0.05, "domain": "POINTS"})
    assert isinstance(key, str) and key

    assert cache.read_temperatures(tmp_path, key) is None  # miss before write

    per_object = {"cup": np.full((4, 10), 295.0), "plate": np.full((4, 7), 296.0)}
    out = cache.write_temperatures(tmp_path, key, per_object, {"num_timesteps": 4})
    assert out.exists()

    back = cache.read_temperatures(tmp_path, key)
    assert back is not None
    assert set(back) == {"cup", "plate"}
    assert np.allclose(back["cup"], 295.0) and back["plate"].shape == (4, 7)


def test_solver_settings_change_cache_key(tmp_path):
    blend = tmp_path / "scene.blend"
    assert cache.cache_key(blend, {"dt": 0.05}, "source") != cache.cache_key(blend, {"dt": 0.1}, "source")
    assert cache.cache_key(blend, {"dt": 0.05}, "source") != cache.cache_key(blend, {"dt": 0.05}, "changed")


def test_cache_rejects_wrong_shape_and_corruption(tmp_path):
    key = "sample"
    cache.write_temperatures(tmp_path, key, {"wall": np.full((3, 4), 295.0)}, {})
    assert cache.read_temperatures(tmp_path, key, {"wall": 4}) is not None
    assert cache.read_temperatures(tmp_path, key, {"wall": 5}) is None
    archive_path = tmp_path / key / "temperatures.npz"
    with np.load(archive_path) as archive:
        stale = json.loads(archive["__meta__"].tobytes())
        stale["schema"] = -1
        wall = archive["wall"].copy()
    np.savez_compressed(archive_path, __meta__=np.frombuffer(json.dumps(stale).encode(), dtype=np.uint8), wall=wall)
    assert cache.read_temperatures(tmp_path, key, {"wall": 4}) is None
    archive_path.write_bytes(b"incomplete")
    assert cache.read_temperatures(tmp_path, key, {"wall": 4}) is None


def test_source_identity_requires_clean_saved_inputs(tmp_path):
    blend = tmp_path / "room.blend"
    image = tmp_path / "albedo.png"
    blend.write_bytes(b"room A")
    image.write_bytes(b"image A")
    data = SimpleNamespace(
        filepath=str(blend),
        is_dirty=False,
        images=[SimpleNamespace(filepath=str(image), packed_file=None)],
        libraries=[],
    )
    original = cache.source_identity(data)
    assert original == cache.source_identity(data)
    image.write_bytes(b"image B")
    assert cache.source_identity(data) != original
    data.images[0].is_dirty = True
    assert cache.source_identity(data) is None
    data.images[0].is_dirty = False
    data.is_dirty = True
    assert cache.source_identity(data) is None


def test_saved_blend_cache_hit_and_explicit_recompute(executable, tmp_path):
    scene_path = str(tmp_path / "cached.blend")

    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        call_service(client, "build_scene", "grid", "Grid")
        saved = call_service(client, "save_scene", scene_path)
        created = call_service(client, "cache_solve", str(tmp_path))
    assert saved["source_is_none"] is False, "source identity unavailable for a clean saved blend"
    assert created["source_is_none"] is False
    assert created["solved"], "cache-priming solve did not produce the Grid object"

    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        reused = call_service(client, "cache_solve", str(tmp_path), scene_path=scene_path, forbid_bake=True)
    assert reused["source_is_none"] is False
    assert reused["solved"], "cache hit did not return the solved Grid object"
    assert reused["recompute_reached_bake"], "recompute reused the old cache"
