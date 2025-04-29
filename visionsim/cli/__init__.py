from __future__ import annotations

import glob
import inspect
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import tyro
from natsort import natsorted

from . import blender, dataset, emulate, ffmpeg, interpolate, transforms

# Exposed for tests
_cli_modules = [blender, dataset, emulate, ffmpeg, interpolate, transforms]


def _validate_directories(
    input_dir: str | os.PathLike, output_dir: str | os.PathLike | None = None, pattern: str | None = None
):
    input_dir = Path(input_dir).resolve()

    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise RuntimeError(f"Input directory {input_dir} does not exist.")

    if pattern:
        # Pattern might be ffmpeg-style like "frames_%06d.png", convert to "frames_*.png".
        pattern = re.sub(r"(%\d+d)", "*", pattern)
        if not (in_files := glob.glob(str(input_dir / pattern))):
            raise FileNotFoundError(f"No files matching {pattern} found in {input_dir}.")
        in_files = natsorted(in_files)
        return input_dir, output_dir, in_files
    return input_dir, output_dir


def _run(
    command: list[str] | str,
    shell: bool = False,
    echo: bool = False,
    log_path: str | os.PathLike | None = None,
    text: bool = True,
    check: bool = False,
):
    """Execute a command and return an object with the result and failure status."""

    if echo:
        print(command)

    # shlex the command if we don't want to run in shell
    if not shell and isinstance(command, str):
        command = shlex.split(command)

    # Either Pipe output or save to a file
    if log_path:
        Path(log_path).mkdir(parents=True, exist_ok=True)
        log_out = Path(log_path).resolve() / "out.log"
        log_err = Path(log_path).resolve() / "err.log"

        with open(str(log_out), "w") as f_out:
            with open(str(log_err), "w") as f_err:
                return subprocess.run(
                    command,
                    shell=shell,
                    check=check,
                    stdout=f_out,
                    stderr=f_err,
                    text=text,
                )
    else:
        return subprocess.run(
            command,
            shell=shell,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=text,
        )


def main():
    cli_dict = {}

    for module in _cli_modules:
        current_module = sys.modules[module.__name__]
        module_name = current_module.__name__.split(".")[-1]
        cli_dict.update(
            {
                f"{module_name}.{func_name}": func
                for func_name, func in inspect.getmembers(current_module, inspect.isfunction)
                if func.__module__ == module.__name__ and not func_name.startswith("_")
            }
        )

    tyro.extras.subcommand_cli_from_dict(cli_dict)
