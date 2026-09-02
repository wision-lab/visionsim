from __future__ import annotations

import subprocess


def test_object_thermal_props_register(executable):
    code = (
        "import bpy;"
        "from visionsim.simulate.heatsim import register;"
        "register();"
        "o=bpy.data.objects.new('o', bpy.data.meshes.new('m'));"
        "o.heat_sim_material.emissivity=0.7;"
        "assert abs(o.heat_sim_material.emissivity-0.7)<1e-6;"
        "assert hasattr(o,'heat_simulation_enabled');"
        "print('THERMAL_PROPS_OK')"
    )
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True, check=False)
    assert "THERMAL_PROPS_OK" in out.stdout, out.stderr
