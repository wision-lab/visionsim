import argparse
import shlex
import subprocess
import sys

try:
    # These are blender specific modules which aren't easily installed but
    # are loaded in when this script is ran from blender.
    import bpy  # type: ignore
except ImportError:
    bpy = None


if __name__ == "__main__":
    # This file should only execute from _within_ blender's runtime to install missing deps.
    # It is called via `install_dependencies` in `__init__.py``.
    if sys.version_info < (3, 9, 0):
        raise RuntimeError("Please use newer blender version with a python version of at least 3.9.")

    if bpy is None:
        sys.exit()

    # Get script specific arguments
    try:
        index = sys.argv.index("--") + 1
    except ValueError:
        index = len(sys.argv)

    parser = argparse.ArgumentParser("Install dependencies into blender's runtime.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--version", type=str)
    group.add_argument("--editable", type=str)
    args, unknown = parser.parse_known_args(sys.argv[index:])

    module_spec = f"visionsim=={args.version}" if args.version else f"--editable {args.editable}"
    commands = [
        f"{sys.executable} -m ensurepip",
        f"{sys.executable} -m pip install -U pip",
        f"{sys.executable} -m pip install rpyc",
        f"{sys.executable} -m pip install --no-warn-script-location --force-reinstall --no-dependencies --verbose {module_spec}",
    ]

    try:
        print("Attempting to auto install dependencies into blender's runtime...")
        print("\n".join(commands))
        outputs = [
            subprocess.run(shlex.split(cmd), stdout=sys.stdout, stderr=subprocess.STDOUT, universal_newlines=True)
            for cmd in commands
        ]
    except subprocess.CalledProcessError:
        print(
            "Some dependencies are needed to run this script. To install it so that "
            "it is accessible from blender, you need to pip install it "
            "into blender's python interpreter like so:\n"
        )
        for cmd in commands:
            print("$", cmd)
