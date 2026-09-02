from __future__ import annotations

import subprocess

# ---------------------------------------------------------------------------
# Shared synthetic-scene builder: a POINTS-domain FEM plate sitting just under
# a DIRICHLET_SOURCE box. The box's bottom face footprint matches the plate's
# extent and sits a few mm above it, so cross-object point pairs are *closer*
# than same-object neighbor spacing -- guaranteeing the point-cloud kNN
# Laplacian links plate points to the hot reservoir (see adapter._combine /
# solver._build_matrices POINTS branch: the Laplacian is a spatial graph over
# ALL combined points, irrespective of which mesh they came from).
# ---------------------------------------------------------------------------
_SCENE_SETUP = r"""
import bpy
from visionsim.simulate.heatsim import register
register()

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Plate: FEM participant, stable topology, starts at (just under) ambient.
bpy.ops.mesh.primitive_grid_add(x_subdivisions=6, y_subdivisions=6, size=0.4)
plate = bpy.context.active_object
plate.name = 'Plate'
plate.heat_simulation_enabled = True
plate.heat_sim_material.initial_temperature_K = 295.0
plate.heat_sim_material.thermal_diffusivity_mm2_s = 50.0

# Box: Dirichlet reservoir at 369 K, bottom face directly above the plate.
bpy.ops.mesh.primitive_cube_add(size=0.4, location=(0.0, 0.0, 0.205))
box = bpy.context.active_object
box.name = 'Box'
box.heat_simulation_enabled = True
box.heat_sim_material.thermal_role = 'DIRICHLET_SOURCE'
box.heat_sim_material.dirichlet_temperature_K = 369.0
"""

_DEFAULTS = """
defaults = dict(initial_temperature_K=295.0, thermal_diffusivity_mm2_s=0.17,
                density_kg_m3=1330.0, specific_heat_J_kgK=880.0, emissivity=0.9,
                irradiance_scale=100.0)
solver_cfg = dict(domain='POINTS', laplacian_backend='ROBUST', device='cpu')
"""


def test_animated_solve_heats_plate_toward_dirichlet_source(executable, tmp_path):
    code = (
        _SCENE_SETUP
        + _DEFAULTS
        + """
import numpy as np
from pathlib import Path
from visionsim.simulate.heatsim import adapter

bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 5

history, frames = adapter.solve_scene_animated(
    bpy.context.scene, defaults=defaults, solver_cfg=solver_cfg,
    cache_root=Path(r'{tmp_path}'),
    frame_start=1, frame_end=5, every_n=1, substeps_per_frame=4,
)

assert frames == [1, 2, 3, 4, 5], frames
assert 'Plate' in history, list(history.keys())
assert 'Box' not in history, 'Dirichlet source must not be recorded'

plate_hist = np.asarray(history['Plate'])
n_plate_verts = len(plate.data.vertices)
assert plate_hist.shape == (5, n_plate_verts), plate_hist.shape

mean_frame1 = float(plate_hist[0].mean())
mean_frame5 = float(plate_hist[-1].mean())
assert np.isfinite(plate_hist).all(), 'non-finite temperatures in plate history'
assert mean_frame5 > mean_frame1, (mean_frame1, mean_frame5)

print('ANIMATED_SOLVE_OK', round(mean_frame1, 4), round(mean_frame5, 4))
"""
    ).replace("{tmp_path}", str(tmp_path))
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "ANIMATED_SOLVE_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_animated_solve_survives_dirichlet_vertex_count_change(executable, tmp_path):
    """The box gains a Subdivision Surface modifier whose level is keyframed to
    step up mid-run, so its *evaluated* vertex count changes between frames.
    This must not crash, and the plate's (FEM-participant, stable-topology)
    history must stay a consistent ``(n_frames, n_plate_verts)`` shape."""
    code = (
        _SCENE_SETUP
        + """
mod = box.modifiers.new('Subsurf', 'SUBSURF')
mod.levels = 0
mod.keyframe_insert(data_path='levels', frame=1)
mod.levels = 3
mod.keyframe_insert(data_path='levels', frame=3)

# Finding 6: the whole point of this test is that the box's EVALUATED vertex
# count actually changes mid-run; assert that directly so the test can't
# silently pass if the resize never triggered (e.g. the Subsurf modifier
# didn't evaluate, or the keyframe didn't take effect).
def _eval_vert_count(obj, frame):
    bpy.context.scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return len(obj.evaluated_get(depsgraph).data.vertices)


n_box_frame1 = _eval_vert_count(box, 1)
n_box_frame5 = _eval_vert_count(box, 5)
assert n_box_frame1 != n_box_frame5, (n_box_frame1, n_box_frame5)
"""
        + _DEFAULTS
        + """
import numpy as np
from pathlib import Path
from visionsim.simulate.heatsim import adapter

bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 5

history, frames = adapter.solve_scene_animated(
    bpy.context.scene, defaults=defaults, solver_cfg=solver_cfg,
    cache_root=Path(r'{tmp_path}'),
    frame_start=1, frame_end=5, every_n=1, substeps_per_frame=4,
)

n_plate_verts = len(plate.data.vertices)
plate_hist = np.asarray(history['Plate'])
assert plate_hist.shape == (5, n_plate_verts), plate_hist.shape
assert np.isfinite(plate_hist).all(), 'non-finite temperatures after a topology change'

print('ANIMATED_RESIZE_OK', plate_hist.shape)
"""
    ).replace("{tmp_path}", str(tmp_path))
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "ANIMATED_RESIZE_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_write_frame_attributes_dirichlet_fallback_uses_reservoir_temperature(executable, tmp_path):
    """When ``write_frame_attributes`` hits its fallback branch (object absent
    from ``history``, e.g. a topology-changing Dirichlet liquid whose evaluated
    vertex count no longer matches), a ``DIRICHLET_SOURCE`` object must stamp
    its reservoir temperature (``dirichlet_temperature_K``), not the ambient
    ``initial_temperature_K`` default. A ``FEM_PARTICIPANT`` hitting the same
    fallback keeps stamping the ambient default (unchanged behavior)."""
    code = (
        _SCENE_SETUP
        + _DEFAULTS
        + """
from visionsim.simulate.heatsim import adapter

# Empty history => both Plate (FEM_PARTICIPANT) and Box (DIRICHLET_SOURCE)
# hit the fallback branch.
adapter.write_frame_attributes(bpy.context.scene, {}, -1, defaults)

assert box['heatsim_default_temperature'] == 369.0, box['heatsim_default_temperature']
assert plate['heatsim_default_temperature'] == 295.0, plate['heatsim_default_temperature']

print('DIRICHLET_FALLBACK_OK')
"""
    )
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "DIRICHLET_FALLBACK_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_thermal_write_frame_advances_animated_field(executable, tmp_path):
    """Task 5 integration test: drive the real ``BlenderService`` methods end-to-end.

    After an animated ``exposed_prepare_thermal`` solve, ``exposed_set_current_frame`` +
    ``_thermal_write_frame`` (the per-frame render hook used by ``exposed_render_frame``)
    must write an increasingly hot ``sim_temperature`` onto the plate as later frames are
    requested, since the transient field keeps evolving toward the Dirichlet reservoir.
    The Dirichlet box itself is absent from the animated history, so it must still hit
    the Task 4 reservoir-temperature fallback rather than ambient.
    """
    code = (
        _SCENE_SETUP
        + r"""
import bpy
import numpy as np
from visionsim.simulate.blender import BlenderService

blend_path = r'{tmp_path}/animated_test.blend'
root_path = r'{tmp_path}'
bpy.ops.wm.save_as_mainfile(filepath=blend_path)

service = BlenderService()
service.exposed_initialize(blend_path, root_path)

service.exposed_prepare_thermal(
    animated=True,
    domain='POINTS',
    laplacian_backend='ROBUST',
    device='cpu',
    frame_start=1,
    frame_end=5,
    substeps_per_frame=4,
    every_n_frames=1,
    initial_temperature_K=295.0,
    thermal_diffusivity_mm2_s=0.17,
    density_kg_m3=1330.0,
    specific_heat_J_kgK=880.0,
    emissivity=0.9,
    irradiance_scale=100.0,
)

assert service._thermal_animated_history is not None, 'animated history was not stored on the service'
assert service._thermal_animated_frames == [1, 2, 3, 4, 5], service._thermal_animated_frames
assert 'Plate' in service._thermal_animated_history, list(service._thermal_animated_history.keys())

plate_obj = bpy.data.objects['Plate']


def plate_mean():
    attr = plate_obj.data.attributes['sim_temperature']
    vals = np.zeros(len(attr.data), dtype=np.float32)
    attr.data.foreach_get('value', vals)
    return float(vals.mean())


service.exposed_set_current_frame(1)
service._thermal_write_frame(1)
mean_early = plate_mean()

service.exposed_set_current_frame(5)
service._thermal_write_frame(5)
mean_late = plate_mean()

assert mean_late > mean_early, (mean_early, mean_late)

# The Dirichlet box has no per-frame history entry, so the write must fall
# through to the reservoir-temperature fallback (Task 4), not ambient.
box_obj = bpy.data.objects['Box']
assert box_obj['heatsim_default_temperature'] == 369.0, box_obj['heatsim_default_temperature']

print('ANIMATED_RENDER_WRITE_OK', round(mean_early, 4), round(mean_late, 4))
"""
    ).replace("{tmp_path}", str(tmp_path))
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "ANIMATED_RENDER_WRITE_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_simulate_for_pose_pins_dirichlet_every_substep(executable, tmp_path):
    """Finding 1 regression: ``HeatSimFEM.simulate_for_pose`` must re-pin
    ``dirichlet_indices`` to ``dirichlet_values`` after EVERY substep's CG
    solve, not just on the initial state or the final returned row. Isolates
    the pin logic itself (no scene/adapter involved) on a tiny synthetic
    point cloud so a regression here can't hide behind adapter-level
    clamping.
    """
    code = r"""
import numpy as np
from types import SimpleNamespace
from visionsim.simulate.heatsim.solver import HeatSimFEM

n = 32
rng = np.random.default_rng(0)
points = rng.uniform(-10.0, 10.0, size=(n, 3)).astype(np.float64)

gen_params = SimpleNamespace(
    device='cpu', RHO=1330.0 / 1e9, C=880.0, K=0.17, NUM_FRAME_DELTA=0.05 * 60.0,
)
sim_params = SimpleNamespace(
    sim_radiation=True, sim_convection=False, add_tikhonov_reg=False,
    sim_time=0.0, record_time=0.0,
)
fem = HeatSimFEM(gen_params, sim_params, laplacian_domain='POINTS', laplacian_backend='ROBUST')

u0 = np.full(n, 295.0, dtype=np.float64)
boundary_mask = np.zeros(n, dtype=bool)
irradiance = np.zeros(n, dtype=np.float64)
density = np.full(n, 1330.0 / 1e9, dtype=np.float64)
specific_heat = np.full(n, 880.0, dtype=np.float64)
tdiff = np.full(n, 50.0, dtype=np.float64)
emissivity = np.full(n, 0.9, dtype=np.float64)

dirichlet_indices = [0, 5, 17]
dirichlet_values = [369.0, 369.0, 369.0]

states = fem.simulate_for_pose(
    points, None, boundary_mask, u0, irradiance, tdiff, density, specific_heat, emissivity,
    num_substeps=4, dt=0.05 / 4,
    dirichlet_indices=dirichlet_indices, dirichlet_values=dirichlet_values,
)

assert states.shape == (4, n), states.shape
for s in range(4):
    row = states[s, dirichlet_indices]
    assert np.all(row == 369.0), (s, row)

# Sanity: the pin isn't a no-op that happens to match because nothing moved --
# a non-pinned vertex must actually be free to evolve toward the hot nodes.
assert states[-1, 1] != 295.0, 'non-pinned vertex did not evolve at all'

print('DIRICHLET_SUBSTEP_PIN_OK')
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "DIRICHLET_SUBSTEP_PIN_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_animated_solve_more_substeps_does_not_lower_plate_mean(executable, tmp_path):
    """Finding 1 regression (coarse, end-to-end): pre-fix, the external
    post-solve Dirichlet clamp only overwrote the RETURNED states array, so
    the reservoir nodes drifted down *within* a frame across substeps (the
    Dirichlet<->FEM coupling weight in ``mv`` keeps pulling them toward the
    FEM neighbors between re-pins) and the plate under-heated -- MORE
    substeps made this WORSE. With the per-substep pin, raising
    ``substeps_per_frame`` must not lower the plate's final mean temperature.
    """
    code = (
        _SCENE_SETUP
        + _DEFAULTS
        + """
import numpy as np
from pathlib import Path
from visionsim.simulate.heatsim import adapter

bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 5


def final_plate_mean(substeps, cache_tag):
    history, frames = adapter.solve_scene_animated(
        bpy.context.scene, defaults=defaults, solver_cfg=solver_cfg,
        cache_root=Path(r'{tmp_path}') / cache_tag,
        frame_start=1, frame_end=5, every_n=1, substeps_per_frame=substeps,
    )
    return float(np.asarray(history['Plate'])[-1].mean())


mean_2 = final_plate_mean(2, 'substeps_2')
mean_8 = final_plate_mean(8, 'substeps_8')

assert mean_8 >= mean_2, (mean_2, mean_8)

print('SUBSTEP_MONOTONIC_OK', round(mean_2, 4), round(mean_8, 4))
"""
    ).replace("{tmp_path}", str(tmp_path))
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "SUBSTEP_MONOTONIC_OK" in out.stdout, out.stdout + "\n" + out.stderr
