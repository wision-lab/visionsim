from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import visionsim

from . import install

_log = logging.getLogger(__name__)


def _local_install_source(package_dir: str | os.PathLike | None = None) -> Path | None:
    """Return the local source tree the running distribution was installed from, if any.

    Editable installs point at the checkout through the package's parent directory. Other local
    installs copy the package into ``site-packages``, so the checkout is recovered from the
    ``file:`` URL pip records for directory installs (PEP 610, ``direct_url.json``). Wheel and
    sdist installs record no directory and return None.

    Args:
        package_dir (str | os.PathLike | None, optional): Installed package directory. Defaults to
            the directory of the imported ``visionsim`` package.

    Returns:
        Path | None: The checkout root, or None when the distribution was not installed from one.
    """
    candidates = [Path(package_dir or visionsim.__path__[0]).parent]

    try:
        direct_url = json.loads(metadata.distribution("visionsim").read_text("direct_url.json") or "{}")
    except (metadata.PackageNotFoundError, ValueError):
        direct_url = {}

    # `file:` URLs cover both `pip install .` and locally built wheels; only directories qualify.
    url = direct_url.get("url") if isinstance(direct_url, dict) else None
    if isinstance(url, str) and urlparse(url).scheme == "file":
        candidates.append(Path(url2pathname(urlparse(url).path)))

    return next((path for path in candidates if (path / "pyproject.toml").is_file()), None)


def _install_args(
    *,
    editable: bool = False,
    version: str | None = None,
    path: str | os.PathLike | None = None,
    torch_index_url: str | None = None,
    package_dir: str | os.PathLike | None = None,
) -> list[str]:
    """Build the arguments handed to ``install.py`` inside Blender's runtime.

    The install source is chosen in this order: an explicit ``version``, an explicit ``path``, the
    checkout the running distribution was installed from, then the running version from PyPI. A
    wheel install lives directly in ``site-packages``, which pip cannot install from, hence the
    PyPI fallback.

    Args:
        editable (bool, optional): Install the source checkout in editable mode. Defaults to False.
        version (str | None, optional): Version to install from PyPI. Defaults to None.
        path (str | os.PathLike | None, optional): Explicit source tree to install. Defaults to None.
        torch_index_url (str | None, optional): PyTorch wheel index. Defaults to None.
        package_dir (str | os.PathLike | None, optional): Installed package directory, used to detect
            a source checkout. Defaults to the directory of the imported ``visionsim`` package.

    Returns:
        list[str]: Arguments to pass after ``--`` to ``install.py``.

    Raises:
        ValueError: raised if ``version`` and ``editable`` are both set, or if an editable install
            is requested from a non-checkout install.
    """
    if version and editable:
        raise ValueError("Cannot specify both `version` and `editable`.")

    source = Path(path) if path is not None else _local_install_source(package_dir)

    if version:
        _log.info(f"Installing visionsim {version} from PyPI into Blender's runtime.")
        args = [f"--version={version}"]
    elif source is None:
        if editable:
            raise ValueError(
                "Cannot install an editable copy: visionsim is not running from a source checkout. "
                "Pass `path` explicitly, or omit `editable`."
            )
        _log.info(f"Installing visionsim {visionsim.__version__} from PyPI into Blender's runtime.")
        args = [f"--version={visionsim.__version__}"]
    elif editable:
        _log.info(f"Installing visionsim in editable mode from {source} into Blender's runtime.")
        # Windows-native separators trip up Blender's bundled pip.
        args = ["--editable", source.as_posix()]
    else:
        _log.info(f"Installing visionsim from {source} into Blender's runtime.")
        args = [source.as_posix()]

    if torch_index_url:
        args.append(f"--torch-index-url={torch_index_url}")
    return args


def install_dependencies(
    executable: str | os.PathLike | None = None,
    editable: bool = False,
    version: str | None = None,
    path: str | None = None,
    torch_index_url: str | None = None,
) -> subprocess.CompletedProcess:
    """Install additional packages into blender`s runtime.

    Args:
        executable (str | os.PathLike | None, optional): Path to the blender executable to use. Defaults to one on PATH.
        editable: (bool, optional): If set, install current visionsim as editable in blender. Only works if
            visionsim is already installed as editable locally.
        version (str | None, optional): The version of visionsim to install. Setting this is akin to specifying the version
            when pip installing. If set, a fresh copy from PyPI will be installed inside blender's runtime environment,
            which might not match the currently installed version. Defaults to None (use currently installed version).
        path (str | None, optional): The path to the visionsim source code to install, only used when --version is not set.
        torch_index_url (str | None, optional): PyTorch wheel index to install torch from, e.g.
            ``https://download.pytorch.org/whl/cpu``. Defaults to None (whatever PyPI resolves to).
    """
    args = _install_args(
        editable=editable,
        version=version,
        path=path,
        torch_index_url=torch_index_url,
    )

    if executable and Path(executable).is_file():
        cmd = [os.fspath(executable)]
    else:
        cmd = shlex.split(str(executable or "blender"))
    cmd += ["-b", "--python-use-system-env", "--python", str(install.__file__), "--"]
    return subprocess.run(cmd + args, stdout=sys.stdout, stderr=subprocess.STDOUT, text=True, check=True)
