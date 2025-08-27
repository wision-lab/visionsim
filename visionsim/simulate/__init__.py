from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import visionsim

from . import install


def install_dependencies(
    executable: str | os.PathLike | None = None, editable: bool = False
) -> subprocess.CompletedProcess:
    """Install additional packages into blender`s runtime.

    Args:
        executable (str | os.PathLike | None, optional): Same as `BlenderServer.spawn`. Defaults to None.
        editable: (bool, optional): If set, install current visionsim as editable in blender. Only works if
            visionsim is already installed as editable locally.
    """
    cmd = f"{executable or 'blender'} -b --python-use-system-env --python {install.__file__} -- "
    cmd += f"--version={visionsim.__version__}" if not editable else f"--editable={Path(visionsim.__path__[0]).parent}"
    return subprocess.run(shlex.split(cmd), stdout=sys.stdout, stderr=subprocess.STDOUT, universal_newlines=True)
