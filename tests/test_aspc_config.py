"""Unit tests for ASPC configuration assembly and fallback resolution."""

from pathlib import Path

import pytest

from visionsim.emulate.aspc.utils import prepare_config, ureg

ALL_CLI_PARAMS = [
    "min_depth",
    "max_depth",
    "n_bins",
    "bin_width",
    "pixel_fov_list",
    "vignette",
    "n_pulses",
    "dead_time",
    "free_running",
    "fast_sim",
    "active_enabled",
    "active_wavelength",
    "active_pulse_repetition",
    "active_pulse_width",
    "active_avg_watts",
    "active_pulse_shape",
    "ambient_enabled",
    "ambient_temperature",
    "ambient_light_conditions",
    "ambient_lambda_pass",
    "ambient_delta_lambda",
    "ambient_intensity",
    "sensor_size",
    "sensor_pixel_pitch",
    "sensor_f_number",
    "sensor_fov",
]


class TestASPCConfigNoCLIAndNoYaml:
    def test_default_config_population_warns(self, recwarn):
        config = prepare_config({}, config_file=None)

        assert config["histogrammer"]["min_depth"] == 0 * ureg.meter
        assert config["histogrammer"]["max_depth"] == 10 * ureg.meter
        assert config["histogrammer"]["n_bins"] == 672
        assert config["active_source"]["pulsed_laser"]["pulse_repetition"] == 10 * ureg.megahertz
        assert config["sensor"]["size"] == (1080, 1920)

        # Assert UserWarning emitted listing defaulted parameters
        warnings_text = " ".join([str(w.message) for w in recwarn.list if issubclass(w.category, UserWarning)])
        assert "fell back to default values" in warnings_text
        for param in ALL_CLI_PARAMS:
            assert param in warnings_text


class TestASPCConfigJustYaml:
    def test_yaml_only_overrides_and_warnings(self, tmp_path: Path, recwarn):
        yaml_content = """
histogrammer:
  min_depth: 1.0
  max_depth: 8.0
  n_bins: 500
  bin_width: 0.03
active_source:
  pulsed_laser:
    pulse_repetition: 10.0
"""
        yaml_file = tmp_path / "aspc_config.yaml"
        yaml_file.write_text(yaml_content)

        config = prepare_config({}, config_file=yaml_file)

        assert config["histogrammer"]["min_depth"] == 1.0 * ureg.meter
        assert config["histogrammer"]["max_depth"] == 8.0 * ureg.meter
        assert config["histogrammer"]["n_bins"] == 500
        assert config["histogrammer"]["bin_width"] == 0.03 * ureg.meter

        warnings_text = " ".join([str(w.message) for w in recwarn.list if issubclass(w.category, UserWarning)])
        
        # YAML scalar missing Pint unit -> emits warning
        assert "YAML parameter 'min_depth' (histogrammer.min_depth) is missing explicit Pint units" in warnings_text
        
        # Parameter not specified in YAML -> listed under default fallback warnings
        assert "dead_time" in warnings_text


class TestASPCConfigJustCLI:
    def test_cli_only_overrides(self, recwarn):
        user_args = {
            "min_depth": 2.0,
            "max_depth": 12.0,
            "n_bins": 400,
            "bin_width": 0.037474057,
            "active_pulse_repetition": 10.0,
        }

        config = prepare_config(user_args, config_file=None)

        assert config["histogrammer"]["min_depth"] == 2.0 * ureg.meter
        assert config["histogrammer"]["max_depth"] == 12.0 * ureg.meter
        assert config["histogrammer"]["n_bins"] == 400

        fallback_warnings = [str(w.message) for w in recwarn.list if "fell back to default values" in str(w.message)]
        assert len(fallback_warnings) == 1
        assert "min_depth" not in fallback_warnings[0]
        assert "max_depth" not in fallback_warnings[0]
        assert "dead_time" in fallback_warnings[0]


class TestASPCConfigBothCLIAndYaml:
    def test_cli_overrides_yaml_all_params_given(self, tmp_path: Path, recwarn):
        yaml_content = """
histogrammer:
  min_depth: !Quantity '0.5 meter'
  max_depth: !Quantity '5.0 meter'
  n_bins: 100
  bin_width: !Quantity '0.1 meter'
  pixel_fov_list: [[0, 1.0, 0, 1.0]]
  vignette: false
  n_pulses: 5000
  dead_time: !Quantity '5.0 nanosecond'
  free_running: true
  fast_sim: false
active_source:
  pulsed_laser:
    enabled: true
    wavelength: !Quantity '850.0 nanometer'
    pulse_repetition: !Quantity '10.0 megahertz'
    pulse_width: !Quantity '4.0 nanosecond'
    avg_watts: !Quantity '0.001 watt'
    pulse_shape: "gaussian"
ambient_source:
  sun:
    enabled: true
    temperature: !Quantity '5500.0 kelvin'
    light_conditions: "BRIGHT_SUNLIGHT"
    lambda_pass: !Quantity '500.0 nanometer'
    delta_lambda: !Quantity '12.0 nanometer'
    intensity: !Quantity '1000.0 watt / meter**2'
sensor:
  size: [512, 512]
  pixel_pitch: !Quantity '5.0 micrometer'
  f_number: 1.8
  fov: [45.0, 45.0]
"""
        yaml_file = tmp_path / "full_config.yaml"
        yaml_file.write_text(yaml_content)

        user_args = {
            "min_depth": 1.0,
            "max_depth": 10.0,
            "n_bins": 672,
            "bin_width": 0.02230658,
            "pixel_fov_list": [[0, 1.0, 0, 1.0]],
            "vignette": True,
            "n_pulses": 20000,
            "dead_time": 15.0,
            "free_running": False,
            "fast_sim": True,
            "active_enabled": True,
            "active_wavelength": 940.0,
            "active_pulse_repetition": 10.0,
            "active_pulse_width": 6.0,
            "active_avg_watts": 0.000000007,
            "active_pulse_shape": "gaussian",
            "ambient_enabled": True,
            "ambient_temperature": 5778.0,
            "ambient_light_conditions": "BRIGHT_SUNLIGHT",
            "ambient_lambda_pass": 550.0,
            "ambient_delta_lambda": 10.0,
            "ambient_intensity": 3.828e26,
            "sensor_size": (1080, 1920),
            "sensor_pixel_pitch": 10.0,
            "sensor_f_number": 1.4,
            "sensor_fov": (90.5, 59.14),
        }

        config = prepare_config(user_args, config_file=yaml_file)

        # CLI overrides YAML
        assert config["histogrammer"]["min_depth"] == 1.0 * ureg.meter
        assert config["histogrammer"]["max_depth"] == 10.0 * ureg.meter
        assert config["histogrammer"]["vignette"] is True
        assert config["histogrammer"]["n_pulses"] == 20000

        # No parameters defaulted -> no fallback warnings
        fallback_warnings = [w for w in recwarn.list if "fell back to default values" in str(w.message)]
        assert len(fallback_warnings) == 0


class TestASPCConfigStructureAndErrors:
    def test_custom_dict_input_raises_value_error(self):
        user_args = {"min_depth": {"invalid": "dict_structure"}}
        with pytest.raises(ValueError, match="Expected a scalar, list, or Pint Quantity, got dictionary"):
            prepare_config(user_args, config_file=None)