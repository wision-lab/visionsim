"""Unit tests for ASPC parameter unit validation and conflict detection."""

import pytest

from visionsim.emulate.aspc.utils import _validate_config, prepare_config, ureg


class TestASPCUnitValidation:
    def test_all_physical_quantities_have_pint_units(self):
        user_args = {
            "min_depth": 1.0,
            "max_depth": 10.0,
            "bin_width": 0.02230658,
            "dead_time": 10.0,
            "active_wavelength": 940.0,
            "active_pulse_repetition": 10.0,
            "active_pulse_width": 6.0,
            "active_avg_watts": 1e-3,
            "ambient_temperature": 5778.0,
            "ambient_lambda_pass": 550.0,
            "ambient_delta_lambda": 10.0,
            "sensor_pixel_pitch": 10.0,
            "sensor_fov": (90.5, 59.14),
        }
        config = prepare_config(user_args, config_file=None)

        hist = config["histogrammer"]
        active = config["active_source"]["pulsed_laser"]
        ambient = config["ambient_source"]["sun"]
        sensor = config["sensor"]

        assert hist["min_depth"].dimensionality == ureg.meter.dimensionality
        assert hist["max_depth"].dimensionality == ureg.meter.dimensionality
        assert hist["bin_width"].dimensionality == ureg.meter.dimensionality
        assert hist["dead_time"].dimensionality == ureg.second.dimensionality

        assert active["wavelength"].dimensionality == ureg.meter.dimensionality
        assert active["pulse_repetition"].dimensionality == ureg.hertz.dimensionality
        assert active["pulse_width"].dimensionality == ureg.second.dimensionality
        assert active["avg_watts"].dimensionality == ureg.watt.dimensionality

        assert ambient["temperature"].dimensionality == ureg.kelvin.dimensionality
        assert ambient["lambda_pass"].dimensionality == ureg.meter.dimensionality
        assert ambient["delta_lambda"].dimensionality == ureg.meter.dimensionality

        assert sensor["pixel_pitch"].dimensionality == ureg.meter.dimensionality
        for fov_axis in sensor["fov"]:
            assert fov_axis.dimensionality == ureg.degree.dimensionality

    def test_incompatible_unit_dimensionality_raises_value_error(self):
        user_args = {"min_depth": 5.0 * ureg.second}
        with pytest.raises(ValueError, match="Dimensionality mismatch for parameter"):
            prepare_config(user_args, config_file=None)

    def test_non_quantity_object_raises_type_error(self):
        invalid_config = {
            "histogrammer": {"min_depth": "invalid_string_instead_of_quantity"},
        }
        with pytest.raises(TypeError, match="expected a Pint Quantity"):
            _validate_config(invalid_config)


class TestASPCConflictValidation:
    def test_negative_min_depth_raises(self):
        user_args = {"min_depth": -1.0}
        with pytest.raises(ValueError, match="Invalid min_depth: value must be >= 0m"):
            prepare_config(user_args, config_file=None)

    def test_min_depth_greater_than_or_equal_max_depth_raises(self):
        user_args = {"min_depth": 10.0, "max_depth": 5.0}
        with pytest.raises(ValueError, match="max_depth .* must be strictly greater than min_depth"):
            prepare_config(user_args, config_file=None)

    def test_max_depth_exceeds_max_resolvable_depth_raises(self):
        # 10 MHz pulse repetition -> max resolvable depth ~ 14.989 m
        user_args = {"active_pulse_repetition": 10.0, "max_depth": 20.0}
        with pytest.raises(ValueError, match="exceeds maximum resolvable depth"):
            prepare_config(user_args, config_file=None)

    def test_conflicting_bin_width_and_n_bins_raises(self):
        user_args = {
            "active_pulse_repetition": 10.0,
            "n_bins": 100,
            "bin_width": 1.0,
        }
        with pytest.raises(ValueError, match="bin_width .* contradicts implied bin width"):
            prepare_config(user_args, config_file=None)