import os
import shlex
import subprocess
import sys

try:
    # These are blender specific modules which aren't easily installed but
    # are loaded in when this script is ran from blender.
    import bpy  # type: ignore
except ImportError:
    bpy = None


def install_dependencies(executable: str | os.PathLike | None = None) -> subprocess.CompletedProcess:
    """Install additional packages into blender`s runtime.

    Args:
        executable (str | os.PathLike | None, optional): Same as `BlenderServer.spawn`. Defaults to None.
    """
    cmd = f"{executable or 'blender'} -b --python {__file__}"
    return subprocess.run(shlex.split(cmd), stdout=sys.stdout, stderr=subprocess.STDOUT, universal_newlines=True)


if __name__ == "__main__":
    # TODO: add CLI to specify version using -- to pass args to python, move this logic out of the init file!

    if bpy is None:
        sys.exit()

    commands = [
        f"{sys.executable} -m ensurepip",
        f"{sys.executable} -m pip install rpyc",
        f"{sys.executable} -m pip install --no-dependencies visionsim",
    ]

    try:
        print("Attempting to auto install dependencies into blender's runtime...")

        for cmd in commands:
            subprocess.run(shlex.split(cmd), stdout=sys.stdout, stderr=subprocess.STDOUT, universal_newlines=True)
    except subprocess.CalledProcessError:
        print(
            "Some dependencies are needed to run this script. To install it so that "
            "it is accessible from blender, you need to pip install it "
            "into blender's python interpreter like so:\n"
        )
        for cmd in commands:
            print("$", cmd)
