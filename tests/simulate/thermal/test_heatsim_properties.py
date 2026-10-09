from __future__ import annotations

from tests.simulate.thermal.heatsim_test_service import SERVICE_PATH, call_service
from visionsim.simulate.blender import BlenderClient


def test_object_thermal_props_register(executable):
    with BlenderClient.spawn(executable=executable, timeout=60, service=SERVICE_PATH) as client:
        result = call_service(client, "thermal_props")

    assert abs(result["emissivity"] - 0.7) < 1e-6
    assert result["has_enabled_prop"] is True
