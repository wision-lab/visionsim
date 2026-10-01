from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

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


def test_solver_settings_change_cache_key(tmp_path):
    blend = tmp_path / "scene.blend"
    assert cache.cache_key(blend, {"dt": 0.05}, "source") != cache.cache_key(
        blend, {"dt": 0.1}, "source"
    )
    assert cache.cache_key(blend, {"dt": 0.05}, "source") != cache.cache_key(
        blend, {"dt": 0.05}, "changed"
    )


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
        filepath=str(blend), is_dirty=False,
        images=[SimpleNamespace(filepath=str(image), packed_file=None)], libraries=[],
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
    scene_path = tmp_path / "cached.blend"
    common = f"""
from pathlib import Path
import bpy
from visionsim.simulate.heatsim import adapter, cache, register
register()
defaults = dict(initial_temperature_K=295.0, thermal_diffusivity_mm2_s=0.17,
                density_kg_m3=1330.0, specific_heat_J_kgK=880.0, emissivity=0.9,
                irradiance_scale=100.0)
settings = dict(sim_time_s=0.1, timestep_s=0.05, device='cpu', bake_samples=4,
                irradiance_texture_size=64)
root = Path(r'{tmp_path}') / 'cache'
"""
    create = common + f"""
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.mesh.primitive_grid_add(x_subdivisions=12, y_subdivisions=12, size=2)
bpy.ops.object.light_add(type='SUN')
bpy.context.active_object.data.energy = 10.0
bpy.ops.wm.save_as_mainfile(filepath=r'{scene_path}')
source = cache.source_identity(bpy.data)
assert source is not None
adapter.solve_scene(bpy.context.scene, defaults=defaults, solver_cfg=settings,
                    cache_root=root, source_digest=source)
print('CACHE_WRITTEN')
"""
    first = subprocess.run([str(executable), "-b", "--python-expr", create], capture_output=True, text=True,
                           check=False)
    assert "CACHE_WRITTEN" in first.stdout, first.stdout + "\n" + first.stderr

    reuse = common + """
source = cache.source_identity(bpy.data)
assert source is not None
def forbid_bake(*args, **kwargs):
    raise RuntimeError('bake was reached')
adapter._compute_irradiance = forbid_bake
history = adapter.solve_scene(bpy.context.scene, defaults=defaults, solver_cfg=settings,
                              cache_root=root, source_digest=source)
assert 'Grid' in history
try:
    adapter.solve_scene(bpy.context.scene, defaults=defaults, solver_cfg=settings,
                        cache_root=root, source_digest=source, recompute=True)
except RuntimeError as exc:
    assert str(exc) == 'bake was reached'
else:
    raise AssertionError('recompute reused the old cache')
print('CACHE_HIT_AND_RECOMPUTE_OK')
"""
    second = subprocess.run(
        [str(executable), "-b", str(scene_path), "--python-expr", reuse], capture_output=True, text=True,
        check=False,
    )
    assert "CACHE_HIT_AND_RECOMPUTE_OK" in second.stdout, second.stdout + "\n" + second.stderr
