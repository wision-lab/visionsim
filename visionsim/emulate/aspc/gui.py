import math
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets
from ruamel.yaml import YAML

from visionsim.emulate.aspc.histogrammers import HistConfig, Histogrammer
from visionsim.emulate.aspc.sensors import SPADSensor
from visionsim.emulate.aspc.sources import LightConditions, PulsedLaser, Sun
from visionsim.emulate.aspc.utils import (
    irradiance_photons,
    preproc_albedo_intensity_depth_frames,
    tof2depth,
    ureg,
    yaml_constructor,
)


def _get_light_conditions_from_string(condition_str: str) -> LightConditions:
    return getattr(LightConditions, condition_str)


class HistogramCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(6, 4))
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot_series(self, y, title: str, ylabel: str):
        self.ax.clear()
        self.ax.plot(y)
        self.ax.set_title(title)
        self.ax.set_xlabel("Time Bins")
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True)
        self.draw_idle()


class ImageCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(6, 4))
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def show_image(self, img, title: str):
        self.ax.clear()
        self.ax.imshow(img, cmap="gray")
        self.ax.set_title(title)
        self.ax.axis("off")
        self.draw_idle()


class ControlPanel(QtWidgets.QWidget):
    valuesChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QFormLayout(self)

        self.n_bins = QtWidgets.QSpinBox()
        self.n_bins.setRange(32, 4096)
        self.n_bins.setValue(1000)
        layout.addRow("n_bins", self.n_bins)

        self.frequency = QtWidgets.QDoubleSpinBox()
        self.frequency.setRange(1.0, 200.0)
        self.frequency.setDecimals(3)
        self.frequency.setValue(10.0)  # MHz
        layout.addRow("laser frequency (MHz)", self.frequency)

        self.avg_watts = QtWidgets.QDoubleSpinBox()
        self.avg_watts.setRange(1e-6, 10.0)
        self.avg_watts.setDecimals(6)
        self.avg_watts.setValue(1.0)  # W
        layout.addRow("laser avg power (W)", self.avg_watts)

        self.pulse_width = QtWidgets.QDoubleSpinBox()
        self.pulse_width.setRange(1e-5, 1e3)
        self.pulse_width.setDecimals(6)
        self.pulse_width.setSingleStep(0.1)
        self.pulse_width.setValue(0.01)  # ns
        layout.addRow("pulse width (ns)", self.pulse_width)

        self.active_enabled = QtWidgets.QCheckBox()
        self.active_enabled.setChecked(True)
        layout.addRow("active enabled", self.active_enabled)

        self.ambient_enabled = QtWidgets.QCheckBox()
        self.ambient_enabled.setChecked(True)
        layout.addRow("ambient enabled", self.ambient_enabled)

        self.fov_index = QtWidgets.QSpinBox()
        self.fov_index.setRange(1, 1)
        self.fov_index.setValue(1)
        layout.addRow("FOV index", self.fov_index)

        # Additional controls
        self.pulse_shape = QtWidgets.QComboBox()
        self.pulse_shape.addItems(["gaussian", "square", "custom"])
        layout.addRow("pulse shape", self.pulse_shape)

        self.bin_width = QtWidgets.QDoubleSpinBox()
        self.bin_width.setRange(0.0, 10.0)
        self.bin_width.setDecimals(9)
        self.bin_width.setValue(0.0)  # meters; 0 => auto from frequency/n_bins
        layout.addRow("bin width (m)", self.bin_width)

        self.pixel_pitch = QtWidgets.QDoubleSpinBox()
        self.pixel_pitch.setRange(0.1, 1000.0)
        self.pixel_pitch.setDecimals(3)
        self.pixel_pitch.setValue(20.0)  # micrometers
        layout.addRow("pixel pitch (µm)", self.pixel_pitch)

        self.f_number = QtWidgets.QDoubleSpinBox()
        self.f_number.setRange(0.5, 32.0)
        self.f_number.setDecimals(3)
        self.f_number.setValue(1.4)
        layout.addRow("f-number (f/N)", self.f_number)

        self.dead_time_ns = QtWidgets.QDoubleSpinBox()
        self.dead_time_ns.setRange(0.0, 1e6)
        self.dead_time_ns.setDecimals(3)
        self.dead_time_ns.setValue(0.0)
        self.dead_time_ns.setSuffix(" ns")
        layout.addRow("dead time", self.dead_time_ns)

        self.free_running = QtWidgets.QCheckBox()
        self.free_running.setChecked(False)
        layout.addRow("free-running mode", self.free_running)

        # Connect signals
        for w in [
            self.n_bins,
            self.frequency,
            self.avg_watts,
            self.pulse_width,
            self.pulse_shape,
            self.bin_width,
            self.pixel_pitch,
            self.f_number,
            self.active_enabled,
            self.ambient_enabled,
            self.fov_index,
            self.dead_time_ns,
            self.free_running,
        ]:
            if isinstance(w, QtWidgets.QAbstractSpinBox):
                w.valueChanged.connect(self.valuesChanged.emit)
            elif isinstance(w, QtWidgets.QCheckBox):
                w.stateChanged.connect(self.valuesChanged.emit)
            elif isinstance(w, QtWidgets.QComboBox):
                w.currentIndexChanged.connect(self.valuesChanged.emit)

    def read_params(self):
        return {
            "n_bins": int(self.n_bins.value()),
            "frequency_mhz": float(self.frequency.value()),
            "avg_watts": float(self.avg_watts.value()),
            "pulse_width_ns": float(self.pulse_width.value()),
            "pulse_shape": str(self.pulse_shape.currentText()),
            "bin_width_m": float(self.bin_width.value()),
            "pixel_pitch_um": float(self.pixel_pitch.value()),
            "f_number": float(self.f_number.value()),
            "active_enabled": bool(self.active_enabled.isChecked()),
            "ambient_enabled": bool(self.ambient_enabled.isChecked()),
            "fov_index": int(self.fov_index.value()) - 1,
            "dead_time_ns": float(self.dead_time_ns.value()),
            "free_running": bool(self.free_running.isChecked()),
        }


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config_path: str = "visionsim/emulate/aspc/config.yaml"):
        super().__init__()
        self.setWindowTitle("Active-SPC Histogram Live Viewer")

        # Central widget
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        hbox = QtWidgets.QHBoxLayout(central)

        # Left: controls
        self.controls = ControlPanel()
        hbox.addWidget(self.controls, 0)

        # Right: plots (histogram + FOV image + pulse shape)
        right_box = QtWidgets.QVBoxLayout()
        self.canvas = HistogramCanvas()
        self.img_canvas = ImageCanvas()
        self.pulse_canvas = HistogramCanvas()
        right_box.addWidget(self.canvas, 1)
        right_box.addWidget(self.img_canvas, 1)
        right_box.addWidget(self.pulse_canvas, 1)
        hbox.addLayout(right_box, 1)

        # Status
        self.status_label = QtWidgets.QLabel("")
        self.statusBar().addWidget(self.status_label)

        # Load config and data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config_path = config_path
        self.yaml = YAML()
        safe_builtins = {"__builtins__": {"list": list, "dict": dict, "tuple": tuple}, "np": np, "math": math}
        self.yaml.Constructor.add_constructor(tag="!Quantity", constructor=yaml_constructor(ureg.Quantity))
        self.yaml.Constructor.add_constructor(tag="!expr", constructor=yaml_constructor(eval, safe_builtins))
        self.yaml.Constructor.add_constructor(tag="!file", constructor=yaml_constructor(self.config_path))

        self.config = self.yaml.load(open(self.config_path))

        self.data_dir = Path("examples/renders/scene1/")
        self.albedo_frames, self.intensity_frames, self.depth_frames = preproc_albedo_intensity_depth_frames(
            root=self.data_dir,
            device=self.device,
            config=self.config,
            start_idx=0,
            num_frames=1,
            requires_grad=False,
        )

        # Prepare sensor (fixed from config)
        self.sensor_config = dict(self.config["sensor"])  # copy
        self.sensor = SPADSensor(**self.sensor_config)

        # Prepare histogrammer (from config)
        self.hist_config = HistConfig(**self.config["histogrammer"])
        self.histogrammer = Histogrammer(self.hist_config)

        # Prepare FOV masks (from config)
        _, img_rows, img_cols = self.depth_frames.shape
        empty_mask = torch.zeros((img_rows, img_cols), dtype=torch.float32, device=self.device)
        self.fov_masks = self.histogrammer.get_perpixel_fov_masks(
            empty_mask, self.histogrammer.pixel_fov_list, device=self.device
        )
        self.controls.fov_index.setMaximum(max(1, len(self.fov_masks)))
        if self.hist_config.dead_time_s is not None:
            self.controls.dead_time_ns.setValue(float(self.histogrammer.dead_time_s.magnitude) * 1e9)
        if self.hist_config.free_running is not None:
            self.controls.free_running.setChecked(bool(self.histogrammer.free_running))

        # Debounce updates
        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(150)
        self._update_timer.timeout.connect(self.update_histogram)

        self.controls.valuesChanged.connect(self._schedule_update)

        # Initial compute
        self.update_histogram()

    def _schedule_update(self):
        self._update_timer.start()

    def _build_active_source(self, params):
        # Convert GUI params to expected units
        active_cfg = dict(self.config["active_source"]["pulsed_laser"])  # copy baseline
        active_cfg["frequency"] = params["frequency_mhz"] * 1e6 * ureg.hertz
        active_cfg["avg_watts"] = params["avg_watts"] * ureg.watts
        active_cfg["pulse_width"] = params["pulse_width_ns"] * 1e-9 * ureg.seconds
        active_cfg["pulse_shape"] = params["pulse_shape"]
        active_cfg["enabled"] = params["active_enabled"]
        active_enabled = active_cfg.pop("enabled")
        return active_enabled, PulsedLaser(**active_cfg) if active_enabled else None

    def _build_ambient_source(self, params):
        ambient_cfg = dict(self.config["ambient_source"]["sun"])  # copy baseline
        ambient_cfg["light_conditions"] = _get_light_conditions_from_string(ambient_cfg["light_conditions"])
        ambient_cfg["enabled"] = params["ambient_enabled"]
        ambient_enabled = ambient_cfg.pop("enabled")
        return ambient_enabled, Sun(**ambient_cfg) if ambient_enabled else None

    def update_histogram(self):
        params = self.controls.read_params()
        try:
            n_bins = params["n_bins"]

            # Rebuild sensor with updated optics
            sensor_cfg = dict(self.sensor_config)
            sensor_cfg["pixel_pitch"] = params["pixel_pitch_um"] * 1e-6 * ureg.meter
            sensor_cfg["f_number"] = params["f_number"] * ureg.dimensionless
            self.sensor = SPADSensor(**sensor_cfg)

            active_enabled, active_source = self._build_active_source(params)
            ambient_enabled, ambient_source = self._build_ambient_source(params)

            num_pixels = self.sensor.w * self.sensor.h

            # If active disabled, create zero irradiance to still show ambient effect
            if active_enabled:
                radiance = active_source.get_scene_radiance(
                    self.albedo_frames, self.depth_frames, num_pixels, self.sensor.omega
                )
                irradiance = (radiance * torch.pi / 4 * (1 / self.sensor.f_number) ** 2).to(irradiance_photons) * (
                    self.sensor.pixel_pitch.to(ureg.meter)
                ) ** 2
                irradiance = torch.tensor(irradiance.magnitude, dtype=torch.float32, device=self.device)
            else:
                irradiance = torch.zeros_like(self.depth_frames, dtype=torch.float32, device=self.device)

            if ambient_enabled:
                ambient_radiance = ambient_source.get_scene_radiance(
                    self.sensor.omega,
                    self.albedo_frames,
                    active_source.frequency if active_source else 10.0 * ureg.megahertz,
                )
                ambient_irradiance = (ambient_radiance * torch.pi / 4 * (1 / self.sensor.f_number) ** 2).to(
                    irradiance_photons
                ) * (self.sensor.pixel_pitch.to(ureg.meter)) ** 2
                offsets = torch.tensor(ambient_irradiance.magnitude, dtype=torch.float32, device=self.device)
            else:
                offsets = torch.zeros_like(self.depth_frames, dtype=torch.float32, device=self.device)

            # Transients
            transients, ambient_offsets = self.histogrammer.calculate_transients(
                irradiance,
                self.depth_frames,
                offsets,
                self.fov_masks,
                n_bins,
                (active_source.max_resolvable_depth.magnitude if active_source else 10.0),
                self.sensor_config["fov"],
                self.histogrammer.pixel_fov_list,
                self.sensor.w,
                self.sensor.h,
                self.sensor.omega,
            )

            # Arrival rates
            if active_source:
                if params["bin_width_m"] > 0.0:
                    bin_width = params["bin_width_m"] * ureg.meter
                else:
                    bin_width = 2 * tof2depth(1 / active_source.frequency) / n_bins
                x_kernel, irf = active_source.get_kernel(bin_width, None)
                # Plot pulse shape in GUI (temporary)
                x_magnitude = x_kernel.magnitude if hasattr(x_kernel, "magnitude") else x_kernel
                self.pulse_canvas.ax.clear()
                self.pulse_canvas.ax.plot(x_magnitude, irf)
                self.pulse_canvas.ax.set_title("Pulse Kernel")
                self.pulse_canvas.ax.set_xlabel("Depth (m)")
                self.pulse_canvas.ax.set_ylabel("Amplitude")
                self.pulse_canvas.ax.grid(True)
                self.pulse_canvas.draw_idle()
            else:
                # Minimal IRF to avoid failure
                irf = np.ones(n_bins, dtype=np.float32)
                # Clear pulse shape plot when active source is disabled
                self.pulse_canvas.ax.clear()
                self.pulse_canvas.ax.set_title("Pulse Kernel (Active Source Disabled)")
                self.pulse_canvas.ax.set_xlabel("Depth (m)")
                self.pulse_canvas.ax.set_ylabel("Amplitude")
                self.pulse_canvas.draw_idle()
            irf_tensor = torch.tensor(irf, dtype=torch.float32, device=self.device)
            arrival_rates = self.histogrammer.calculate_arrival_rates(irf_tensor, transients, ambient_offsets, n_bins)

            # Select FOV
            idx = max(0, min(params["fov_index"], len(arrival_rates) - 1))
            y = arrival_rates[idx].detach().cpu().numpy()
            self.canvas.plot_series(y, title=f"Photon Arrival Rates (FOV {idx + 1})", ylabel="Rate (photons/bin)")

            # FOV image (albedo masked)
            fov_mask_np = self.fov_masks[idx].detach().cpu().numpy()
            albedo_np = self.albedo_frames[0].detach().cpu().numpy()
            img = albedo_np * fov_mask_np
            self.img_canvas.show_image(img, title=f"Albedo × FOV Mask (FOV {idx + 1})")

            self.status_label.setText(
                f"n_bins={n_bins} | freq={params['frequency_mhz']} MHz | Pavg={params['avg_watts']} W | pulse={params['pulse_width_ns']} ns | "
                f"bin_w={params['bin_width_m']} m | pp={params['pixel_pitch_um']} µm | f/{params['f_number']} | shape={params['pulse_shape']} | "
                f"dead_time={params['dead_time_ns']} ns | free_run={params['free_running']}"
            )
        except Exception as e:
            self.status_label.setText(f"Error: {e}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1100, 900)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
