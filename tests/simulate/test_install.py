from __future__ import annotations

import json
import logging

import pytest

import visionsim
from visionsim import simulate
from visionsim.simulate import _install_args


class _FakeDistribution:
    def __init__(self, payload: str | None) -> None:
        self.payload = payload

    def read_text(self, name: str) -> str | None:
        return self.payload


@pytest.fixture
def not_installed_from_checkout(monkeypatch):
    """Make the running distribution look like a wheel install with no local origin."""
    monkeypatch.setattr(simulate.metadata, "distribution", lambda name: _FakeDistribution(None))


def patch_direct_url(monkeypatch, payload: dict | str) -> None:
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr(simulate.metadata, "distribution", lambda name: _FakeDistribution(raw))


def test_source_checkout_installs_from_path(tmp_path):
    package = tmp_path / "visionsim"
    package.mkdir()
    (tmp_path / "pyproject.toml").write_text("")

    assert _install_args(package_dir=package) == [tmp_path.as_posix()]


def test_editable_source_checkout_sets_flag(tmp_path):
    package = tmp_path / "visionsim"
    package.mkdir()
    (tmp_path / "pyproject.toml").write_text("")

    assert _install_args(editable=True, package_dir=package) == ["--editable", tmp_path.as_posix()]


def test_local_non_editable_install_recovers_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("")
    patch_direct_url(monkeypatch, {"dir_info": {}, "url": checkout.as_uri()})

    package = tmp_path / "site-packages" / "visionsim"
    package.mkdir(parents=True)

    assert _install_args(package_dir=package) == [checkout.as_posix()]


def test_remote_direct_url_falls_back_to_installed_version(tmp_path, monkeypatch):
    patch_direct_url(
        monkeypatch,
        {"archive_info": {"hash": "sha256=deadbeef"}, "url": "https://files.pythonhosted.org/visionsim-0.2.0.whl"},
    )
    package = tmp_path / "site-packages" / "visionsim"
    package.mkdir(parents=True)

    assert _install_args(package_dir=package) == [f"--version={visionsim.__version__}"]


def test_missing_checkout_falls_back_to_installed_version(tmp_path, monkeypatch):
    patch_direct_url(monkeypatch, {"dir_info": {}, "url": (tmp_path / "gone").as_uri()})
    package = tmp_path / "site-packages" / "visionsim"
    package.mkdir(parents=True)

    assert _install_args(package_dir=package) == [f"--version={visionsim.__version__}"]


def test_corrupt_direct_url_falls_back_to_installed_version(tmp_path, monkeypatch):
    patch_direct_url(monkeypatch, "{not json")
    package = tmp_path / "site-packages" / "visionsim"
    package.mkdir(parents=True)

    assert _install_args(package_dir=package) == [f"--version={visionsim.__version__}"]


def test_wheel_install_falls_back_to_installed_version(tmp_path, not_installed_from_checkout):
    package = tmp_path / "site-packages" / "visionsim"
    package.mkdir(parents=True)

    assert _install_args(package_dir=package) == [f"--version={visionsim.__version__}"]


def test_editable_wheel_install_raises(tmp_path, not_installed_from_checkout):
    package = tmp_path / "site-packages" / "visionsim"
    package.mkdir(parents=True)

    with pytest.raises(ValueError, match="not running from a source checkout"):
        _install_args(editable=True, package_dir=package)


def test_explicit_path_overrides_detection(tmp_path):
    package = tmp_path / "site-packages" / "visionsim"
    package.mkdir(parents=True)
    source = tmp_path / "checkout"
    source.mkdir()

    assert _install_args(path=source, package_dir=package) == [source.as_posix()]


def test_version_takes_precedence_over_path(tmp_path):
    assert _install_args(version="1.2.3", path=tmp_path) == ["--version=1.2.3"]


def test_version_and_editable_are_exclusive():
    with pytest.raises(ValueError, match="both"):
        _install_args(version="1.2.3", editable=True)


def test_torch_index_url_is_forwarded(tmp_path):
    package = tmp_path / "visionsim"
    package.mkdir()
    (tmp_path / "pyproject.toml").write_text("")

    assert _install_args(package_dir=package, torch_index_url="https://example.invalid/cpu") == [
        tmp_path.as_posix(),
        "--torch-index-url=https://example.invalid/cpu",
    ]


def test_logs_checkout_install_source(tmp_path, caplog):
    source = tmp_path / "checkout"
    source.mkdir()
    (source / "pyproject.toml").write_text("")

    with caplog.at_level(logging.INFO, logger="visionsim.simulate"):
        _install_args(path=source, package_dir=tmp_path / "unused")

    assert f"Installing visionsim from {source}" in caplog.text


def test_logs_editable_install_source(tmp_path, caplog):
    source = tmp_path / "checkout"
    source.mkdir()
    (source / "pyproject.toml").write_text("")

    with caplog.at_level(logging.INFO, logger="visionsim.simulate"):
        _install_args(editable=True, path=source, package_dir=tmp_path / "unused")

    assert f"Installing visionsim in editable mode from {source}" in caplog.text


def test_logs_pypi_fallback(tmp_path, caplog, not_installed_from_checkout):
    package = tmp_path / "site-packages" / "visionsim"
    package.mkdir(parents=True)

    with caplog.at_level(logging.INFO, logger="visionsim.simulate"):
        _install_args(package_dir=package)

    assert f"Installing visionsim {visionsim.__version__} from PyPI" in caplog.text
