"""A sparse wall must gain true thermal samples under localized illumination."""

from __future__ import annotations

import subprocess


def test_auto_coarse_wall_resolves_localized_heating(executable):
    code = r"""
import bpy
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from visionsim.simulate.heatsim import register, adapter
register()
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.mesh.primitive_plane_add(size=2)
wall = bpy.context.active_object
wall.name = 'wall'
material = bpy.data.materials.new('white')
material.diffuse_color = (0.8, 0.8, 0.8, 1)
material.use_nodes = True
wall.data.materials.append(material)
bpy.ops.object.light_add(type='POINT', location=(-0.6, 0, 0.4))
bpy.context.active_object.data.energy = 200
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
objects = adapter.gather_meshes(scene)
defaults = dict(initial_temperature_K=295, thermal_diffusivity_mm2_s=0.17,
                density_kg_m3=1330, specific_heat_J_kgK=880, emissivity=0.9,
                irradiance_scale=500)
solver = dict(sim_time_s=0.1, timestep_s=0.05, device='cpu', bake_samples=32,
              irradiance_texture_size=64)
fields = []
for density, tile in ((64, 16), (256, 32)):
    plan = adapter.build_atlas_plan(scene, objects, dict(render_domain='AUTO',
        atlas_texel_density=density, atlas_tile_min=tile, atlas_tile_max=tile,
        atlas_texel_soft_max=10000))
    assert 'wall' in plan.texels
    history = adapter.solve_scene(scene, defaults=defaults, solver_cfg=solver,
                                  cache_root=Path('/tmp/visionsim-coarse-wall-test'),
                                  atlas_plan=plan)
    field = history['wall'][-1]
    fields.append((plan.texels['wall']['position_mm'], field))
    assert len(field) > len(wall.data.vertices)
    assert np.isfinite(field).all()
    assert float(np.ptp(field)) > 0.5, 'localized heating lost on the sparse wall'
low_points, low_field = fields[0]
ref_points, ref_field = fields[1]
_, nearest = cKDTree(ref_points).query(low_points)
rms = float(np.sqrt(np.mean((low_field - ref_field[nearest]) ** 2)))
assert rms < 0.15, f'coarse/reference RMS={rms:.4f} K'
print('COARSE_WALL_OK', len(low_field), len(ref_field), rms)
"""
    result = subprocess.run(
        [str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False
    )
    assert "COARSE_WALL_OK" in result.stdout, result.stdout + "\n" + result.stderr
