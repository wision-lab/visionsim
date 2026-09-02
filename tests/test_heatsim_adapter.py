from __future__ import annotations

import subprocess
from types import SimpleNamespace

from visionsim.simulate.heatsim import adapter

# Distinctive globals so a leaked PropertyGroup default can never masquerade as a
# global fallback (the PropertyGroup ships emissivity=0.9, density=1330.0).
_GLOBAL_DEFAULTS = {
    "initial_temperature_K": 300.0,
    "thermal_diffusivity_mm2_s": 0.42,
    "density_kg_m3": 1234.0,
    "specific_heat_J_kgK": 777.0,
    "emissivity": 0.5,
}


class _FakeMat:
    """Stand-in for ``obj.heat_sim_material`` with a controllable ``is_property_set``.

    ``always_set`` mirrors a PropertyGroup where every field was authored (True) or
    where every field is still at its registered default (False) - the exact axis
    that ``resolve_material`` must branch on.
    """

    def __init__(self, *, always_set: bool, **values):
        self._always_set = always_set
        for k, v in values.items():
            setattr(self, k, v)

    def is_property_set(self, attr):
        return self._always_set


def test_resolve_material_falls_back_to_globals_when_unset():
    """I1 regression: unset per-object props must defer to the global defaults.

    The PointerProperty is always present and a FloatProperty never returns None,
    so without ``is_property_set`` gating the (distinctive) per-object values below
    would shadow the globals and the ``--config.thermal.*`` knobs would be inert.
    """
    mat = _FakeMat(
        always_set=False,
        initial_temperature_K=295.372,  # PropertyGroup-style defaults that must be ignored
        thermal_diffusivity_mm2_s=0.17,
        density_kg_m3=1330.0,
        specific_heat_J_kgK=880.0,
        emissivity=0.9,
        thermal_role="DIRICHLET_SOURCE",
        dirichlet_temperature_K=400.0,
    )
    obj = SimpleNamespace(heat_sim_material=mat)

    out = adapter.resolve_material(obj, _GLOBAL_DEFAULTS)

    assert out["initial_temperature_K"] == 300.0
    assert out["thermal_diffusivity_mm2_s"] == 0.42
    assert out["density_kg_m3"] == 1234.0
    assert out["specific_heat_J_kgK"] == 777.0
    assert out["emissivity"] == 0.5
    # thermal_role / dirichlet_temperature_K have no global key -> hard defaults,
    # NOT the stale group values.
    assert out["thermal_role"] == "FEM_PARTICIPANT"
    assert out["dirichlet_temperature_K"] == 0.0


def test_resolve_material_uses_per_object_when_set():
    """Explicitly-set per-object values win over the globals (and are clamped)."""
    mat = _FakeMat(
        always_set=True,
        initial_temperature_K=310.0,
        thermal_diffusivity_mm2_s=0.99,
        density_kg_m3=7777.0,
        specific_heat_J_kgK=500.0,
        emissivity=2.0,  # out of range -> must clamp to 1.0
        thermal_role="dirichlet_source",  # lower-case -> upper-cased
        dirichlet_temperature_K=400.0,
    )
    obj = SimpleNamespace(heat_sim_material=mat)

    out = adapter.resolve_material(obj, _GLOBAL_DEFAULTS)

    assert out["initial_temperature_K"] == 310.0
    assert out["thermal_diffusivity_mm2_s"] == 0.99
    assert out["density_kg_m3"] == 7777.0
    assert out["specific_heat_J_kgK"] == 500.0
    assert out["emissivity"] == 1.0
    assert out["thermal_role"] == "DIRICHLET_SOURCE"
    assert out["dirichlet_temperature_K"] == 400.0


def test_solve_writes_finite_sim_temperature(executable, tmp_path):
    """End-to-end adapter smoke test inside a real Blender process.

    Builds a tiny lit scene (subdivided plane + overhead sun + a world with
    some background light), runs the cached FEM solve via the adapter, writes
    the last-timestep ``sim_temperature`` attribute, and asserts the result is
    finite and physical. Also checks that the Direct-Kernel irradiance actually
    produced non-zero per-vertex flux (so the test cannot pass with a silent
    zero-flux fallback) and that a second ``solve_scene`` reuses the cache.
    """
    code = f"""
import bpy, numpy as np
from pathlib import Path
from visionsim.simulate.heatsim import register, adapter

register()

# --- start from a clean scene -------------------------------------------------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Subdivided plane (grid) -> enough points for the ROBUST point-cloud Laplacian.
bpy.ops.mesh.primitive_grid_add(x_subdivisions=15, y_subdivisions=15, size=2.0)
plane = bpy.context.active_object
plane.name = 'ThermalPlane'
plane.heat_simulation_enabled = True

# Overhead sun (default rotation emits along -Z, i.e. straight down).
bpy.ops.object.light_add(type='SUN')
sun = bpy.context.active_object
sun.data.energy = 10.0

# Give the world some light so the sky term is non-zero too.
world = bpy.context.scene.world
if world is None:
    world = bpy.data.worlds.new('World')
    bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get('Background')
if bg is not None:
    bg.inputs['Color'].default_value = (0.2, 0.2, 0.2, 1.0)
    bg.inputs['Strength'].default_value = 1.0

defaults = dict(initial_temperature_K=295.0, thermal_diffusivity_mm2_s=0.17,
                density_kg_m3=1330.0, specific_heat_J_kgK=880.0, emissivity=0.9,
                irradiance_scale=100.0)
solver_cfg = dict(sim_time_s=0.15, timestep_s=0.05, domain='POINTS',
                  laplacian_backend='ROBUST', device='cpu')
cache_root = Path(r'{tmp_path}')

hist = adapter.solve_scene(bpy.context.scene, defaults=defaults,
                           solver_cfg=solver_cfg, cache_root=cache_root)
assert 'ThermalPlane' in hist, list(hist.keys())
T_hist = np.asarray(hist['ThermalPlane'])
assert T_hist.ndim == 2 and T_hist.shape[0] >= 2, T_hist.shape

# Second call must come straight from the cache (no re-solve). We assert the
# cache hit via the RETURN value (cached history equals the first solve), not via
# captured stdout, so the test does not depend on any debug print side-effect.
hist2 = adapter.solve_scene(bpy.context.scene, defaults=defaults,
                            solver_cfg=solver_cfg, cache_root=cache_root)
assert hist2.keys() == hist.keys(), (list(hist2.keys()), list(hist.keys()))
assert np.array_equal(T_hist, np.asarray(hist2['ThermalPlane']))

adapter.write_frame_attributes(bpy.context.scene, hist, timestep=-1, defaults=defaults)

# sim_temperature: finite + physical.
attr = plane.data.attributes['sim_temperature'].data
vals = np.array([d.value for d in attr])
assert np.isfinite(vals).all(), 'non-finite temperatures'
assert vals.min() > 200 and vals.max() < 2000, (float(vals.min()), float(vals.max()))

# emissivity attribute written.
eps = np.array([d.value for d in plane.data.attributes['emissivity'].data])
assert np.allclose(eps, 0.9), float(eps.mean())

# The Direct-Kernel irradiance pass produced real (non-zero) flux.
irr = np.array([d.value for d in plane.data.attributes['sim_irradiance'].data])
assert np.isfinite(irr).all() and irr.max() > 0.0, float(irr.max())

print('THERMAL_ADAPTER_OK')
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "THERMAL_ADAPTER_OK" in out.stdout, out.stderr


def test_shared_mesh_objects_get_independent_copies(executable, tmp_path):
    """Fix 3: linked duplicates (multiple objects pointing at one Mesh datablock) share
    per-vertex attributes AND UV layers on that datablock, so writing sim_temperature for
    one object overwrites the other -- last write wins. This is the minimal repro: two
    objects sharing a mesh, given distinct fields (310 K / 350 K), both used to end up at
    350 K. gather_meshes must un-share each object's mesh (once, idempotently) BEFORE any
    per-vertex write happens, so each object ends up with -- and keeps -- its own values.
    """
    code = """
import bpy, numpy as np
from visionsim.simulate.heatsim import register, adapter

register()

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

bpy.ops.mesh.primitive_grid_add(x_subdivisions=6, y_subdivisions=6, size=1.0)
obj_a = bpy.context.active_object
obj_a.name = 'SharedA'
obj_a.heat_simulation_enabled = True

# Linked duplicate: obj_b shares obj_a's mesh datablock (mirrors Blender's Alt-D).
obj_b = obj_a.copy()
obj_b.name = 'SharedB'
obj_b.location.x += 2.0
bpy.context.collection.objects.link(obj_b)
obj_b.heat_simulation_enabled = True

assert obj_a.data is obj_b.data, 'objects must start out sharing one mesh datablock'
assert obj_a.data.users >= 2, obj_a.data.users

sim_objects = adapter.gather_meshes(bpy.context.scene)
names = {o.name for o in sim_objects}
assert names == {'SharedA', 'SharedB'}, names

# After gather_meshes, each object must have its own single-user mesh.
assert obj_a.data is not obj_b.data, 'meshes still shared after gather_meshes'
assert obj_a.data.users == 1 and obj_b.data.users == 1, (obj_a.data.users, obj_b.data.users)

# Hand-craft distinct histories (skip a full FEM solve -- Fix 3 is about write-back, not
# the solver) and write them via the same function prepare_thermal uses.
n = len(obj_a.data.vertices)
defaults = dict(initial_temperature_K=295.0, thermal_diffusivity_mm2_s=0.17,
                density_kg_m3=1330.0, specific_heat_J_kgK=880.0, emissivity=0.9)
history = {'SharedA': np.full((2, n), 310.0), 'SharedB': np.full((2, n), 350.0)}
adapter.write_frame_attributes(bpy.context.scene, history, timestep=-1, defaults=defaults)

vals_a = np.array([d.value for d in obj_a.data.attributes['sim_temperature'].data])
vals_b = np.array([d.value for d in obj_b.data.attributes['sim_temperature'].data])
assert np.allclose(vals_a, 310.0), vals_a  # NOT 350.0 (the old last-write-wins bug)
assert np.allclose(vals_b, 350.0), vals_b

# Idempotent: a second gather_meshes call must not re-copy (already single-user).
mesh_a_name, mesh_b_name = obj_a.data.name, obj_b.data.name
adapter.gather_meshes(bpy.context.scene)
assert obj_a.data.name == mesh_a_name and obj_b.data.name == mesh_b_name

print('SHARED_MESH_OK')
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "SHARED_MESH_OK" in out.stdout, out.stderr


def test_read_authored_irradiance_scale():
    from visionsim.simulate.heatsim.adapter import read_authored_irradiance_scale

    class _FakeScene:
        def __init__(self, data):
            self._data = data
        def get(self, key, default=None):
            return self._data.get(key, default)

    # Authored heat_sim_settings with an irradiance_scale -> returns it.
    authored = _FakeScene({"heat_sim_settings": {"irradiance_scale": 1000.0}})
    assert read_authored_irradiance_scale(authored) == 1000.0

    # No heat_sim_settings at all -> None (caller keeps its default).
    assert read_authored_irradiance_scale(_FakeScene({})) is None

    # heat_sim_settings present but no irradiance_scale key -> None.
    partial = _FakeScene({"heat_sim_settings": {"fem_domain": "POINTS"}})
    assert read_authored_irradiance_scale(partial) is None
