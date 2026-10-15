"""The bake must describe the same mesh the solver does.

Regression cover for two defects that each silently produced a complete, plausible-looking
render:

* The Cycles irradiance bake reduced per-vertex values against ``obj.data`` while the
  solver builds its nodes from the EVALUATED mesh (modifiers applied). When a modifier
  changed the vertex count the arrays disagreed, ``_combine`` dropped the flux, and the
  object absorbed nothing -- 229 of 289 objects on one interior scene, every one of them
  pinned at its initial temperature regardless of the flux computed for it.

* ``prepare_object_bake_uv`` guarded on the truthiness of ``mesh.uv_layers``. An empty UV
  collection is falsy, so it returned early on exactly the meshes that needed a UV layer
  created -- no bake UV, hence no atlas UV, hence demotion out of the texture atlas. On a
  scene where 85% of objects carry no authored UVs that was 231 objects demoted, most of
  which then rendered as a single flat temperature.
"""

from __future__ import annotations

from tests.simulate.thermal.heatsim_test_service import SERVICE_PATH, call_service
from visionsim.simulate.blender import BlenderClient


def test_irradiance_bake_is_indexed_against_the_evaluated_mesh(executable):
    """vertex_flux must be sized to the mesh the solver uses, not the pre-modifier one."""
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        call_service(client, "build_scene", "plane", "Plane", lit=False)
        modified = call_service(client, "add_subsurf", levels=2)
        baked = call_service(client, "bake_irradiance")

    assert modified["eval_n"] != modified["base_n"], "modifier did not change the vertex count; test is vacuous"
    assert baked["flux_n"] is not None, "bake returned None"
    assert baked["flux_n"] == baked["eval_n"], (
        f"vertex_flux has {baked['flux_n']} entries, evaluated mesh has {baked['eval_n']} (base has {baked['base_n']})"
    )


def test_albedo_and_irradiance_bakes_agree_on_the_mesh(executable):
    """The two bakes feed one multiply, so they must return the same length.

    They are produced by separate code paths (``bake_vertex_albedo`` vs
    ``bake_irradiance_map``), and the evaluated-mesh fix was originally applied to only
    one of them. A mismatch is not loud: the caller discards the albedo and assumes full
    absorption, overestimating absorbed flux by up to ~4x on a light surface.
    """
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        call_service(client, "build_scene", "plane", "Plane", material="plain", lit=False)
        call_service(client, "add_subsurf", levels=2)
        result = call_service(client, "bake_irradiance", with_albedo=True)

    assert result["flux_n"] is not None and result["albedo_n"] is not None, "one of the bakes returned nothing"
    assert result["eval_n"] != result["base_n"], "modifier did not change the count; test is vacuous"
    assert result["flux_n"] == result["eval_n"], f"irradiance has {result['flux_n']}, evaluated has {result['eval_n']}"
    assert result["albedo_n"] == result["eval_n"], f"albedo has {result['albedo_n']}, evaluated has {result['eval_n']}"


def test_bake_uv_is_created_on_a_mesh_with_no_authored_uvs(executable):
    """A mesh with zero UV layers must still get a bake UV, and still reach the atlas."""
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        call_service(client, "build_scene", "plane", "Plane", size=8.0, lit=False)
        result = call_service(client, "prepare_bake_uv")

    assert result["uv_before"] == 0, "failed to strip UVs; test is vacuous"
    assert result["has_bake_uv"], "prepare_object_bake_uv left a UV-less mesh without a bake UV"
    assert result["in_plan"], f"object demoted from the atlas; texels={result['texels']}"
