from types import SimpleNamespace

from visionsim.simulate.blender import BlenderService

DEVICES = [SimpleNamespace(name="gpu", type="CUDA"), SimpleNamespace(name="cpu", type="CPU")]


def names(devices):
    return [device.name for device in devices]


def test_selects_gpu_and_cpu_when_requested():
    assert names(BlenderService._select_devices(DEVICES, "CUDA", use_cpu=True)) == ["gpu", "cpu"]


def test_selects_only_gpu_without_cpu():
    assert names(BlenderService._select_devices(DEVICES, "CUDA", use_cpu=False)) == ["gpu"]


def test_cpu_type_is_not_duplicated():
    assert names(BlenderService._select_devices(DEVICES, "CPU", use_cpu=True)) == ["cpu"]


def test_none_backend_selects_cpu():
    assert names(BlenderService._select_devices(DEVICES, "NONE", use_cpu=True)) == ["cpu"]
