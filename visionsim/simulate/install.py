import argparse
import subprocess
import sys
from pathlib import Path

try:
    # These are blender specific modules which aren't easily installed but
    # are loaded in when this script is ran from blender.
    import bpy  # type: ignore
except ImportError:
    bpy = None


if __name__ == "__main__":
    # This file should only execute from _within_ blender's runtime to install missing deps.
    # It is called via `install_dependencies` in `__init__.py``.

    if bpy is None:
        sys.exit()

    # Get script specific arguments
    try:
        index = sys.argv.index("--") + 1
    except ValueError:
        index = len(sys.argv)

    parser = argparse.ArgumentParser("Install dependencies into blender's runtime.")
    parser.add_argument("--version", type=str)
    parser.add_argument("--editable", action="store_true")
    parser.add_argument("path", type=str, nargs="?")
    args, unknown = parser.parse_known_args(sys.argv[index:])

    if args.version:
        if args.editable or args.path:
            parser.error("If --version is set, neither path or --editable can be set.")
        module_spec = [f"visionsim=={args.version}"]
    elif args.editable:
        if not args.path:
            parser.error("If --editable is set, path must be set.")
        module_spec = ["--editable", args.path]
    elif args.path:
        module_spec = [args.path]
    else:
        parser.error("Either --version or path must be provided.")

    base_cmd = [Path(sys.executable).as_posix(), "-m"]

    print(f"Blender Python executable: {sys.executable}", flush=True)
    print(f"Blender Python path: {sys.path}", flush=True)

    # torch wheel source is platform-specific.  On Linux/Windows we pull the GPU
    # build from the cu124 index (matches NVIDIA driver >=520 / CUDA 12.x; adjust the
    # index URL for other drivers, e.g. cu118 for older cards).  macOS has no cu124
    # wheels, so we install the default wheel there (MPS/CPU) - otherwise pip would
    # fail to resolve and (with check=True) abort the whole post-install.
    if sys.platform == "darwin":
        torch_cmd = base_cmd + ["pip", "install", "torch"]
    else:
        torch_cmd = base_cmd + ["pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu124"]

    # NOTE: the core visionsim install precedes the torch/scipy/robust_laplacian
    # step so that a torch/index hiccup can never block the base package setup
    # (scipy/robust_laplacian are not on the PyTorch index, so they go via PyPI).
    commands = [
        base_cmd + ["ensurepip"],
        base_cmd + ["pip", "install", "-U", "pip"],
        base_cmd + ["pip", "install", "rpyc", "peewee", "typing-extensions"],
        base_cmd
        + ["pip", "install", "--no-warn-script-location", "--force-reinstall", "--no-dependencies", "--verbose"]
        + module_spec,
        torch_cmd,
        base_cmd + ["pip", "install", "scipy", "robust_laplacian"],
    ]

    try:
        print("Attempting to auto install dependencies into blender's runtime...", flush=True)
        for cmd in commands:
            print(f"Running: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, stdout=sys.stdout, stderr=subprocess.STDOUT, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\nInstallation failed with error: {e}", flush=True)
        print(
            "\nSome dependencies are needed to run this script. To install it so that "
            "it is accessible from blender, you need to pip install it "
            "into blender's python interpreter like so:\n"
        )
        for cmd in commands:
            print("$", " ".join(cmd))
        raise
