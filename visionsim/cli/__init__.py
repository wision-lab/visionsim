from __future__ import annotations

import glob
import inspect
import logging
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import tyro
from natsort import natsorted
from rich.logging import RichHandler
from rich.traceback import install
from typing_extensions import overload

from . import blender, dataset, emulate, ffmpeg, interpolate, transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logging.getLogger("PIL").setLevel(logging.WARNING)
_log = logging.getLogger("rich")
install(suppress=[tyro])


# Exposed for tests
_cli_modules = [blender, dataset, emulate, ffmpeg, interpolate, transforms]


@overload
def _validate_directories(
    input_dir: str | os.PathLike, output_dir: None = None, pattern: str = ""
) -> tuple[Path, None, list[str]]: ...


@overload
def _validate_directories(
    input_dir: str | os.PathLike, output_dir: str | os.PathLike = "", pattern: str = ""
) -> tuple[Path, Path, list[str]]: ...


@overload
def _validate_directories(
    input_dir: str | os.PathLike, output_dir: None = None, pattern: str | None = None
) -> tuple[Path, None]: ...


@overload
def _validate_directories(
    input_dir: str | os.PathLike, output_dir: str | os.PathLike = "", pattern: str | None = None
) -> tuple[Path, Path]: ...


def _validate_directories(
    input_dir: str | os.PathLike, output_dir: str | os.PathLike | None = None, pattern: str | None = None
) -> tuple[Path, Path | None] | tuple[Path, Path | None, list[str]]:
    input_path = Path(input_dir).resolve()

    if output_dir is not None:
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    if not input_path.exists():
        raise RuntimeError(f"Input directory {input_path} does not exist.")

    if pattern:
        # Pattern might be ffmpeg-style like "frames_%06d.png", convert to "frames_*.png".
        pattern = re.sub(r"(%\d+d)", "*", pattern)
        if not (in_files := glob.glob(str(input_path / pattern))):
            raise FileNotFoundError(f"No files matching {pattern} found in {input_path}.")
        in_files = natsorted(in_files)
        return input_path, output_path, in_files
    return input_path, output_path


def _run(
    command: list[str] | str,
    shell: bool = False,
    echo: bool = False,
    log_path: str | os.PathLike | None = None,
    text: bool = True,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Execute a command and return an object with the result and failure status."""

    if echo:
        _log.debug(f"Running command: {command}")

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


def post_install(executable: str | os.PathLike | None = None, editable: bool = False):
    """Install additional dependencies

    Args:
        executable (str | os.PathLike | None, optional): Path to Blender executable. Defaults to one found on $PATH.
        editable: (bool, optional): If set, install current visionsim as editable in blender. Only works if
            visionsim is already installed as editable locally.
    """
    from visionsim.simulate import install_dependencies

    if _run(f"{executable or 'blender'} --version", shell=True).returncode != 0:
        raise RuntimeError(
            "No blender installation found on path! Please make sure it is discoverable, or specify executable."
        )
    install_dependencies(executable, editable=editable)


def main():
    cli_dict = {"post-install": post_install}

    for module in _cli_modules:
        current_module = sys.modules[module.__name__]
        module_name = current_module.__name__.split(".")[-1]
        cli_dict.update(
            {
                f"{module_name}.{func_name}".replace("_", "-"): func
                for func_name, func in inspect.getmembers(current_module, inspect.isfunction)
                if func.__module__ == module.__name__ and not func_name.startswith("_")
            }
        )

    tyro.extras.subcommand_cli_from_dict(cli_dict)
