"""Tests for the thermal preview colormap: the global temperature range helper
and the inferno-stop compositor node group (see
docs/superpowers/specs/2026-07-07-visionsim-thermal-parity-design.md §9).

Both run inside ``blender --background`` (adapter imports bpy; the node group
builds a compositor tree), matching the pattern in ``test_heatsim_irradiance``.
"""

import subprocess


def test_global_temperature_range_is_robust_to_outliers(executable):
    code = r"""
import numpy as np
from visionsim.simulate.heatsim import adapter

# A realistic scene: the bulk sits near ambient with a modest warm spread, plus a
# handful of artifact-hot vertices (2000 K, well under 1% of the data) that must
# NOT dictate the colormap.
final = np.full(1000, 295.0)
final[:100] = np.linspace(296.0, 310.0, 100)   # genuine warm spread, up to ~310 K
final[-3:] = 2000.0                             # 0.3% runaway artifact vertices
hist = {'room': np.stack([np.full(1000, 295.0), final])}   # (timesteps, vertices)

tmin, tmax = adapter.global_temperature_range(hist, default_K=295.0)
assert abs(tmin - 295.0) < 1e-6, tmin          # floored at default / P1 of the ambient bulk
assert tmax < 320.0, tmax                       # P99 rejects the 2000 K outliers
assert tmax > 300.0, tmax                       # but still covers the genuine warm spread

# empty history -> (default, default + 1)
lo, hi = adapter.global_temperature_range({}, 295.0)
assert (lo, hi) == (295.0, 296.0), (lo, hi)

# near-uniform field -> span widened to >= 1 K (no colour collapse)
lo, hi = adapter.global_temperature_range({'o': np.full((2, 4), 300.0)}, 300.0)
assert lo == 300.0 and (hi - lo) >= 1.0, (lo, hi)

print('RANGE_OK', round(tmin, 3), round(tmax, 3))
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "RANGE_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_thermal_preview_node_group_uses_inferno_stops_and_range(executable):
    code = r"""
import bpy  # noqa: F401
from visionsim.simulate.nodes import thermal_preview_node_group

ng = thermal_preview_node_group(tmin=295.0, tmax=297.0)

# MapRange wired to [tmin, tmax] -> [0, 1]
mr = ng.nodes['TempNormalize']
assert abs(mr.inputs[1].default_value - 295.0) < 1e-6, mr.inputs[1].default_value
assert abs(mr.inputs[2].default_value - 297.0) < 1e-6, mr.inputs[2].default_value

# Inferno ramp: 11 stops from near-black to pale yellow (heat-sim _INFERNO_STOPS)
ramp = ng.nodes['InfernoRamp'].color_ramp
assert len(ramp.elements) == 11, len(ramp.elements)
lo, hi = ramp.elements[0], ramp.elements[10]
assert abs(lo.position - 0.0) < 1e-6 and abs(hi.position - 1.0) < 1e-6
assert lo.color[0] < 0.02 and lo.color[2] < 0.02, tuple(lo.color)   # near-black low end
assert hi.color[0] > 0.9 and hi.color[1] > 0.9, tuple(hi.color)     # pale-yellow high end

# Regression guard: the old turbo low stop was blue-ish (0.190, 0.072, 0.232)
assert abs(lo.color[0] - 0.18995) > 1e-3, 'colormap is still turbo, not inferno'

print('INFERNO_NODEGROUP_OK')
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "INFERNO_NODEGROUP_OK" in out.stdout, out.stdout + "\n" + out.stderr
