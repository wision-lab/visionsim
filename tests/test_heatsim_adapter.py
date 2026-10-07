from __future__ import annotations

from types import SimpleNamespace

from tests.heatsim_test_service import SERVICE_PATH, call_service
from visionsim.simulate.blender import BlenderClient
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

    Builds a tiny lit scene (subdivided plane + overhead sun + a world with some
    background light), runs the thermal solve via the adapter, writes the last-timestep
    ``sim_temperature`` attribute, and asserts the result is finite and physical. A
    second solve must return the same field.
    """
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        call_service(client, "build_scene", "grid", "ThermalPlane", subdivisions=15)
        solved = call_service(client, "solve", str(tmp_path), "ThermalPlane", sim_time_s=0.15)
        twice = call_service(client, "solve_twice", str(tmp_path), "ThermalPlane")

    assert solved["keys"] == ["ThermalPlane"], solved["keys"]
    assert twice["keys2"] == twice["keys"] == solved["keys"], (twice["keys2"], twice["keys"])
    assert solved["ndim"] == 2 and solved["steps"] >= 2, (solved["ndim"], solved["steps"])
    assert twice["deterministic"], "a repeated solve returned a different field"
    assert solved["finite"], "non-finite temperatures"
    assert solved["min"] > 200 and solved["max"] < 2000, (solved["min"], solved["max"])
    # The same bound must hold for the per-vertex attribute actually written back.
    assert solved["written_finite"], "non-finite written sim_temperature"
    assert solved["written_min"] > 200 and solved["written_max"] < 2000, (
        solved["written_min"],
        solved["written_max"],
    )
    assert abs(solved["emissivity"] - 0.9) < 1e-6, solved["emissivity"]
    assert solved["rose"], "Cycles heating produced no temperature rise"


def test_shared_mesh_objects_get_independent_copies(executable, tmp_path):
    """Fix 3: linked duplicates (multiple objects pointing at one Mesh datablock) share
    per-vertex attributes AND UV layers on that datablock, so writing sim_temperature for
    one object overwrites the other -- last write wins. This is the minimal repro: two
    objects sharing a mesh, given distinct fields (310 K / 350 K), both used to end up at
    350 K. gather_meshes must un-share each object's mesh (once, idempotently) BEFORE any
    per-vertex write happens, so each object ends up with -- and keeps -- its own values.
    """
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        call_service(client, "build_scene", "grid", "SharedA", subdivisions=6, size=1.0, lit=False)
        linked = call_service(client, "duplicate_linked", "SharedA", "SharedB")
        assert linked["shared"], "objects must start out sharing one mesh datablock"
        gathered = call_service(client, "gather_meshes")
        written = call_service(
            client,
            "write_history",
            {
                "SharedA": {"vertices": gathered["vertices"]["SharedA"], "value": 310.0},
                "SharedB": {"vertices": gathered["vertices"]["SharedB"], "value": 350.0},
            },
        )
        again = call_service(client, "gather_meshes")

    assert gathered["names"] == ["SharedA", "SharedB"], gathered["names"]
    assert linked["users"] >= 2, linked["users"]
    # After gather_meshes, each object must have its own single-user mesh.
    assert gathered["users"] == {"SharedA": 1, "SharedB": 1}, gathered["users"]
    assert abs(written["SharedA"] - 310.0) < 1e-6, written["SharedA"]  # NOT 350.0 (old last-write-wins)
    assert abs(written["SharedB"] - 350.0) < 1e-6, written["SharedB"]
    # Idempotent: a second gather_meshes call must not re-copy (already single-user).
    assert again["mesh_names"] == gathered["mesh_names"], "gather_meshes re-copied a single-user mesh"
