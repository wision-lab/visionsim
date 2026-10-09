from __future__ import annotations

import inspect
from dataclasses import asdict, fields

import pytest

from visionsim.simulate.blender import BlenderService
from visionsim.simulate.config import ThermalConfig


@pytest.mark.parametrize("field,value", [
    ("sim_time_s", 0.0),
    ("timestep_s", -1.0),
    ("atlas_texel_density", float("nan")),
    ("emissivity", 1.1),
    ("render_domain", "MESH"),
])
def test_thermal_config_rejects_invalid_inputs(field, value):
    with pytest.raises(ValueError):
        ThermalConfig(**{field: value})

_ATLAS_FIELDS = (
    "render_domain",
    "atlas_texel_density",
    "atlas_tile_min",
    "atlas_tile_max",
    "atlas_texel_soft_max",
)


def test_thermal_config_atlas_fields_dispatch_parity():
    """The 5 atlas fields must exist on ThermalConfig with the spec'd defaults, and every
    one of the three thermal-dispatch entry points must accept them with the SAME default
    (not just the same name) -- ``**asdict(config.thermal)`` drives all three uniformly, so a
    mismatched default would silently diverge from the config's own default whenever a caller
    omits the field.
    """
    cfg = ThermalConfig()
    assert cfg.render_domain == "AUTO"
    assert cfg.atlas_texel_density == 1500.0
    assert cfg.atlas_tile_min == 16
    assert cfg.atlas_tile_max == 512
    assert cfg.atlas_texel_soft_max == 500_000

    conf_defaults = {f.name: f.default for f in fields(ThermalConfig) if f.name in _ATLAS_FIELDS}
    assert set(conf_defaults) == set(_ATLAS_FIELDS)

    for method in (
        BlenderService.exposed_prepare_thermal,
        BlenderService.exposed_heatsim_solve,
        BlenderService.exposed_include_thermal,
    ):
        sig_params = inspect.signature(method).parameters
        for name in _ATLAS_FIELDS:
            assert name in sig_params, f"{method.__name__} is missing {name!r}"
            assert sig_params[name].default == conf_defaults[name], (
                f"{method.__name__}.{name} default {sig_params[name].default!r} != "
                f"ThermalConfig.{name} default {conf_defaults[name]!r}"
            )

    # asdict(ThermalConfig()) must round-trip through every key too (belt-and-suspenders,
    # mirrors test_heatsim_assignments_integration.test_asdict_dispatch_matches_both_service_signatures).
    keys = set(asdict(cfg))
    for method in (BlenderService.exposed_prepare_thermal, BlenderService.exposed_heatsim_solve):
        missing = keys - (set(inspect.signature(method).parameters) - {"self"})
        assert not missing, f"{method.__name__} is missing {sorted(missing)}"
