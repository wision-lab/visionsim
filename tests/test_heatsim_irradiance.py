import subprocess


def test_bake_albedo_map_returns_varying_pixels(executable):
    code = r"""
import bpy, numpy as np
from visionsim.simulate.heatsim import register, irradiance

register()
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.mesh.primitive_grid_add(x_subdivisions=12, y_subdivisions=12, size=2.0)
obj = bpy.context.active_object

mat = bpy.data.materials.new('checker_mat')
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes.get('Principled BSDF')
checker = nt.nodes.new('ShaderNodeTexChecker')
checker.inputs['Scale'].default_value = 6.0
nt.links.new(checker.outputs['Color'], bsdf.inputs['Base Color'])
obj.data.materials.append(mat)

bpy.context.scene.render.engine = 'CYCLES'
try:
    bpy.context.scene.cycles.device = 'CPU'
    bpy.context.scene.cycles.samples = 4
except Exception:
    pass

baked = irradiance.bake_albedo_map(bpy.context.scene, obj, 128)
assert baked is not None, 'bake_albedo_map returned None'
px = baked.pixels
assert px.ndim == 3 and px.shape[2] == 3, f'bad pixel shape {px.shape}'
assert float(px.std()) > 0.1, f'expected high-contrast checker variation (std>0.1), got std={px.std()}'
mean = float(px.mean())
assert 0.0 <= mean <= 1.0, f'albedo mean out of range: {mean}'
print('ALBEDO_BAKE_OK', px.shape, round(mean, 3), round(float(px.std()), 3))
"""
    out = subprocess.run(
        [str(executable), "-b", "--python-expr", code],
        capture_output=True, text=True,
     check=False)
    assert "ALBEDO_BAKE_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_cycles_absorbed_flux_varies_with_albedo(executable, tmp_path):
    code = r"""
import bpy, numpy as np
from pathlib import Path
from visionsim.simulate.heatsim import register, adapter

register()
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.mesh.primitive_grid_add(x_subdivisions=20, y_subdivisions=20, size=2.0)
obj = bpy.context.active_object
obj.name = 'ThermalPlane'
obj.heat_simulation_enabled = True

mat = bpy.data.materials.new('checker_mat')
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes.get('Principled BSDF')
checker = nt.nodes.new('ShaderNodeTexChecker')
checker.inputs['Scale'].default_value = 6.0
nt.links.new(checker.outputs['Color'], bsdf.inputs['Base Color'])
obj.data.materials.append(mat)

bpy.ops.object.light_add(type='SUN')
bpy.context.active_object.data.energy = 10.0
world = bpy.context.scene.world
world.use_nodes = True
bg = world.node_tree.nodes.get('Background')
bg.inputs['Strength'].default_value = 1.0

defaults = dict(initial_temperature_K=295.0, thermal_diffusivity_mm2_s=0.17,
                density_kg_m3=1330.0, specific_heat_J_kgK=880.0, emissivity=0.9,
                irradiance_scale=100.0)
solver_cfg = dict(sim_time_s=0.1, timestep_s=0.05, 
                  device='cpu')
flux = adapter._compute_irradiance_cycles(bpy.context.scene, [obj], solver_cfg, defaults)[obj]
assert flux.shape == (len(obj.data.vertices),)
assert float(flux.std()) > 0.01, f'absorbed flux did not vary with checker albedo: {{flux.std()}}'
assert np.all(np.isfinite(flux))
print('VARYING_ALBEDO_OK', round(float(flux.mean()), 3), round(float(flux.std()), 3))
""".replace("{tmp}", str(tmp_path))
    out = subprocess.run(
        [str(executable), "-b", "--python-expr", code],
        capture_output=True, text=True,
     check=False)
    assert "VARYING_ALBEDO_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_all_zero_attribute_albedo_is_ignored_and_rebaked(executable):
    code = r"""
import bpy, numpy as np
from visionsim.simulate.heatsim import register, irradiance
register()
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()
bpy.ops.mesh.primitive_grid_add(x_subdivisions=12, y_subdivisions=12, size=2.0)
obj = bpy.context.active_object
mat = bpy.data.materials.new('checker'); mat.use_nodes = True
nt = mat.node_tree; bsdf = nt.nodes.get('Principled BSDF')
ck = nt.nodes.new('ShaderNodeTexChecker'); ck.inputs['Scale'].default_value = 6.0
nt.links.new(ck.outputs['Color'], bsdf.inputs['Base Color'])
obj.data.materials.append(mat)
bpy.context.scene.render.engine = 'CYCLES'
try: bpy.context.scene.cycles.device = 'CPU'; bpy.context.scene.cycles.samples = 4
except Exception: pass
nv = len(obj.data.vertices)
# Stale all-zeros mesh attribute for this object must NOT be served; must re-bake.
mesh = obj.data
attr = mesh.attributes.new(name='albedo', type='FLOAT', domain='POINT')
attr.data.foreach_set('value', np.zeros(nv, dtype=np.float64))
alb = irradiance.bake_vertex_albedo(bpy.context.scene, obj, texture_size=128)
assert alb is not None, 'albedo absent'
assert float(alb.std()) > 0.05, f'zeros attribute was served instead of re-baking (std={alb.std()})'
print('ZERO_ATTR_IGNORED_OK', round(float(alb.mean()),3), round(float(alb.std()),3))
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code],
                         capture_output=True, text=True, check=False)
    assert "ZERO_ATTR_IGNORED_OK" in out.stdout, out.stdout + "\n" + out.stderr
