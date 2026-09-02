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

import subprocess


def test_irradiance_bake_is_indexed_against_the_evaluated_mesh(executable):
    """vertex_flux must be sized to the mesh the solver uses, not the pre-modifier one."""
    code = r"""
import bpy
from visionsim.simulate.heatsim import irradiance

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

bpy.ops.mesh.primitive_plane_add(size=2)
obj = bpy.context.active_object
obj.modifiers.new("Subsurf", "SUBSURF").levels = 2   # changes the vertex count

base_n = len(obj.data.vertices)
dg = bpy.context.evaluated_depsgraph_get()
eval_n = len(obj.evaluated_get(dg).data.vertices)
assert eval_n != base_n, "modifier did not change the vertex count; test is vacuous"

bpy.context.scene.cycles.device = 'CPU'
bpy.context.scene.cycles.samples = 4
baked = irradiance.bake_irradiance_map(bpy.context.scene, obj, 64, samples=4)
assert baked is not None, "bake returned None"

n = len(baked.vertex_flux)
assert n == eval_n, f"vertex_flux has {n} entries, evaluated mesh has {eval_n} (base has {base_n})"
print("EVALUATED_MESH_ALIGNMENT_OK")
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "EVALUATED_MESH_ALIGNMENT_OK" in out.stdout, out.stdout + out.stderr


def test_albedo_and_irradiance_bakes_agree_on_the_mesh(executable):
    """The two bakes feed one multiply, so they must return the same length.

    They are produced by separate code paths (``bake_albedo_map`` /
    ``irradiance_kernel.get_or_bake_vertex_albedo`` vs ``bake_irradiance_map``), and the
    evaluated-mesh fix was originally applied to only one of them. A mismatch is not
    loud: the caller discards the albedo and assumes full absorption, overestimating
    absorbed flux by up to ~4x on a light surface.
    """
    code = r"""
import bpy
from visionsim.simulate.heatsim import irradiance, irradiance_kernel

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

bpy.ops.mesh.primitive_plane_add(size=2)
obj = bpy.context.active_object
obj.data.materials.append(bpy.data.materials.new("m"))
obj.modifiers.new("Subsurf", "SUBSURF").levels = 2

bpy.context.scene.cycles.device = 'CPU'
bpy.context.scene.cycles.samples = 4

flux = irradiance.bake_irradiance_map(bpy.context.scene, obj, 64, samples=4)
albedo = irradiance_kernel.get_or_bake_vertex_albedo(bpy.context.scene, [obj], texture_size=64)
a = albedo.get(obj.name)
assert flux is not None and a is not None, "one of the bakes returned nothing"
dg = bpy.context.evaluated_depsgraph_get()
eval_n = len(obj.evaluated_get(dg).data.vertices)
assert eval_n != len(obj.data.vertices), "modifier did not change the count; test is vacuous"
assert len(flux.vertex_flux) == eval_n, f"irradiance has {len(flux.vertex_flux)}, evaluated has {eval_n}"
assert len(a) == eval_n, f"albedo has {len(a)}, evaluated has {eval_n}"
print("BAKE_LENGTHS_AGREE_OK")
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "BAKE_LENGTHS_AGREE_OK" in out.stdout, out.stdout + out.stderr


def test_bake_uv_is_created_on_a_mesh_with_no_authored_uvs(executable):
    """A mesh with zero UV layers must still get a bake UV, and still reach the atlas."""
    code = r"""
import bpy
from visionsim.simulate.heatsim import adapter, atlas, irradiance
from visionsim.simulate.heatsim.constants import BAKE_UV_LAYER_NAME

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

bpy.ops.mesh.primitive_plane_add(size=8)   # large + few verts => wants the atlas
obj = bpy.context.active_object
while obj.data.uv_layers:
    obj.data.uv_layers.remove(obj.data.uv_layers[0])
assert len(obj.data.uv_layers) == 0, "failed to strip UVs; test is vacuous"

irradiance.prepare_object_bake_uv(obj)
assert BAKE_UV_LAYER_NAME in obj.data.uv_layers, (
    "prepare_object_bake_uv left a UV-less mesh without a bake UV"
)

# ... and the object must actually survive into the atlas rather than being demoted.
plan = adapter.build_atlas_plan(
    bpy.context.scene, [obj],
    {"atlas_texel_density": 1500.0, "atlas_tile_min": 16, "atlas_tile_max": 512,
     "atlas_texel_soft_max": 500000},
)
assert obj.name in plan.texels, f"object demoted from the atlas; texels={list(plan.texels)}"
print("UVLESS_MESH_REACHES_ATLAS_OK")
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "UVLESS_MESH_REACHES_ATLAS_OK" in out.stdout, out.stdout + out.stderr
