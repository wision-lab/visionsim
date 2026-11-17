from collections import OrderedDict


class SensorBase:
    """Base class for camera sensors with unit-aware perspective projections."""

    def __init__(
        self,
        *,
        size: tuple[int, int] = (1080, 1920),
        fov=(90.5, 90.5),
    ):
        self.h, self.w = size
        self.num_pixels = self.h * self.w
        self.fov = fov

        # Parameter names list
        self._param_names = [
            "w",
            "h",
            "fov",
            "num_pixels"
        ]

    @property
    def params(self):
        return OrderedDict((i, getattr(self, i)) for i in self._param_names)


class SPADSensor(SensorBase):
    def __init__(self, **sensor_base_kwargs):
        super().__init__(**sensor_base_kwargs)
        self._sensor_base_kwargs = sensor_base_kwargs