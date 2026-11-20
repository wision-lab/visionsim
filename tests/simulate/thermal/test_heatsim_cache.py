from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tests.simulate.thermal.heatsim_test_service import SERVICE_PATH, call_service
from visionsim.simulate.blender import BlenderClient
from visionsim.simulate.heatsim import cache


def test_bake_uv_cache_keeps_the_first_published_addresses(tmp_path):
    path = tmp_path / "uv.npz"
    uv = np.array([[0.1, 0.2], [0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
    assert cache.read_bake_uv(path, "geometry", 3) is None
    np.testing.assert_array_equal(cache.publish_bake_uv(path, "geometry", uv), uv.ravel())
    np.testing.assert_array_equal(cache.publish_bake_uv(path, "geometry", uv[::-1]), uv.ravel())
    np.testing.assert_array_equal(cache.read_bake_uv(path, "geometry", 3), uv.ravel())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["uv.npz"]


def test_bake_uv_cache_competing_writers_use_the_same_complete_layout(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    path = tmp_path / "uv.npz"
    barrier = Barrier(4)
    candidates = [np.array([[v, v], [v + 0.1, v], [v, v + 0.1]], dtype=np.float32) for v in [0.1, 0.2, 0.3, 0.4]]

    def publish(uv):
        barrier.wait()
        return cache.publish_bake_uv(path, "geometry", uv)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(publish, candidates))
    assert any(np.array_equal(results[0], uv.ravel()) for uv in candidates)
    for result in results:
        np.testing.assert_array_equal(result, results[0])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["uv.npz"]


@pytest.mark.parametrize("kind", ["signature", "count", "dtype", "nonfinite", "bounds", "truncated"])
def test_bake_uv_cache_rejects_stale_or_damaged_layouts(tmp_path, kind):
    path = tmp_path / "uv.npz"
    uv = np.full((3, 2), 0.5, dtype=np.float32)
    signature = "different" if kind == "signature" else "geometry"
    if kind == "dtype":
        uv = uv.astype(np.float64)
    if kind in {"nonfinite", "bounds"}:
        uv[0, 0] = np.nan if kind == "nonfinite" else 1.1
    np.savez(path, signature=np.asarray(signature), uv=uv)
    if kind == "truncated":
        path.write_bytes(path.read_bytes()[:100])
    with pytest.raises(ValueError, match="Invalid thermal UV cache"):
        cache.read_bake_uv(path, "geometry", 4 if kind == "count" else 3)


def test_bake_uv_cache_does_not_publish_invalid_coordinates(tmp_path):
    path = tmp_path / "uv.npz"
    with pytest.raises(ValueError, match="Invalid thermal UV cache"):
        cache.publish_bake_uv(path, "geometry", np.full((3, 2), np.nan, dtype=np.float32))
    assert not list(tmp_path.iterdir())


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


def test_pipeline_revision_invalidates_keys_and_saved_temperatures(tmp_path, monkeypatch):
    blend = tmp_path / "scene.blend"
    config = {"dt": 0.05}
    key = cache.cache_key(blend, config, "source")
    values = {"wall": np.full((3, 4), 295.0)}
    cache.write_temperatures(tmp_path, key, values, {})
    assert cache.read_temperatures(tmp_path, key, {"wall": 4}) is not None

    monkeypatch.setattr(cache, "BAKE_PIPELINE_VERSION", cache.BAKE_PIPELINE_VERSION + 1)
    new_key = cache.cache_key(blend, config, "source")
    assert new_key != key
    assert cache.read_temperatures(tmp_path, key, {"wall": 4}) is None
    cache.write_temperatures(tmp_path, new_key, values, {})
    fresh = cache.read_temperatures(tmp_path, new_key, {"wall": 4})
    assert fresh is not None
    np.testing.assert_array_equal(fresh["wall"], values["wall"])


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

    # An interrupted write leaves a truncated but otherwise well-formed zip, which np.load
    # reports as zipfile.BadZipFile -- not an OSError. It must be a cache miss too.
    cache.write_temperatures(tmp_path, key, {"wall": np.full((3, 4), 295.0)}, {})
    truncated = archive_path.read_bytes()
    archive_path.write_bytes(truncated[: len(truncated) // 2])
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
