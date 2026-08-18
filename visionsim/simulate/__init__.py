from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import visionsim

from . import install


def install_dependencies(
    executable: str | os.PathLike | None = None,
    editable: bool = False,
    version: str | None = None,
    path: str | None = None,
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
    """
    if version and editable:
        raise ValueError("Cannot specify both version and editable.")

    cmd = f"{executable or 'blender'} -b --python-use-system-env --python '{install.__file__}' -- "
    path = path or Path(visionsim.__path__[0]).parent.as_posix()

    if version:
        cmd += f"--version={version}"
    else:
        cmd += f"--editable {path}" if editable else f"{path}"
    return subprocess.run(shlex.split(cmd), stdout=sys.stdout, stderr=subprocess.STDOUT, text=True, check=True)
