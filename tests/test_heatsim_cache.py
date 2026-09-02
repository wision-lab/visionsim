from __future__ import annotations

import numpy as np

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


def test_animated_roundtrip_and_miss(tmp_path):
    cache_dir = tmp_path / "some_animated_key"
    assert cache.read_animated(cache_dir) is None  # miss: dir doesn't exist yet

    cache_dir.mkdir()
    assert cache.read_animated(cache_dir) is None  # miss: dir exists but empty

    history = {"cup": np.full((5, 100), 300.0), "plate": np.full((5, 40), 295.0)}
    frames = [1, 2, 3, 4, 5]
    out = cache.write_animated(cache_dir, history, frames, {"substeps": 4})
    assert out.exists()

    back = cache.read_animated(cache_dir)
    assert back is not None
    hist_back, frames_back = back
    assert set(hist_back) == {"cup", "plate"}
    assert hist_back["cup"].shape == (5, 100)
    assert hist_back["plate"].shape == (5, 40)
    assert np.allclose(hist_back["cup"], 300.0) and np.allclose(hist_back["plate"], 295.0)
    assert frames_back == frames


def test_animated_key_differs_from_static_key(tmp_path):
    blend = tmp_path / "scene.blend"
    static_cfg = {"dt": 0.05, "domain": "POINTS"}
    animated_cfg = {
        **static_cfg,
        "animated": True,
        "frame_start": 1,
        "frame_end": 5,
        "every_n": 1,
        "substeps": 4,
    }
    static_key = cache.cache_key(blend, static_cfg)
    animated_key = cache.cache_key(blend, animated_cfg)
    assert static_key != animated_key
