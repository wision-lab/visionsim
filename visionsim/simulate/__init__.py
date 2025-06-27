import os
import shlex
import subprocess
import sys
from pathlib import Path

import visionsim

from . import install


def install_dependencies(executable: str | os.PathLike | None = None) -> subprocess.CompletedProcess:
    """Install additional packages into blender`s runtime.

    Args:
        executable (str | os.PathLike | None, optional): Same as `BlenderServer.spawn`. Defaults to None.
    """
    cmd = f"{executable or 'blender'} -b --python {install.__file__} -- {visionsim.__version__}"
    return subprocess.run(shlex.split(cmd), stdout=sys.stdout, stderr=subprocess.STDOUT, universal_newlines=True)
