from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import visionsim

from . import install


def install_dependencies(
    executable: str | os.PathLike | None = None, editable: bool = False, version: str | None = None
) -> subprocess.CompletedProcess:
    """Install additional packages into blender`s runtime.

    Args:
        executable (str | os.PathLike | None, optional): Path to the blender executable to use. Defaults to one on PATH.
        editable: (bool, optional): If set, install current visionsim as editable in blender. Only works if
            visionsim is already installed as editable locally.
        version (str | None, optional): The version of visionsim to install. Setting this is akin to specifying the version
            when pip installing. If set, a fresh copy from PyPI will be installed inside blender's runtime environment,
            which might not match the currently installed version. Defaults to None (use currently installed version).
    """
    if version and editable:
        raise ValueError("Cannot specify both version and editable")
    cmd = f"{executable or 'blender'} -b --python-use-system-env --python '{install.__file__}' -- "
    if version:
        cmd += f"--version={version}"
    elif editable:
        cmd += f"--editable='{Path(visionsim.__path__[0]).parent.as_posix()}'"
    else:
        cmd += f"--path={Path(visionsim.__path__[0]).parent.as_posix()}"
    return subprocess.run(shlex.split(cmd), stdout=sys.stdout, stderr=subprocess.STDOUT, universal_newlines=True)
