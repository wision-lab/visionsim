from __future__ import annotations

import contextlib
import glob
import inspect
import io
import logging
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal, get_args, overload

import tyro
from natsort import natsorted
from rich.logging import RichHandler
from rich.traceback import install

from . import blender, dataset, emulate, ffmpeg, interpolate, transforms

logging.basicConfig(
    level=os.environ.get("VSIM_LOG_LEVEL", "INFO").upper(),
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
    log_path: str | os.PathLike | None = None,
    text: bool = True,
    hide: bool = False,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Execute a command and return an object with the result and failure status."""
    _log.debug(f"Running command: {command}")

    # shlex the command if we don't want to run in shell
    if not shell and isinstance(command, str):
        command = shlex.split(command)

    # Either Pipe output or save to a file
    if log_path:
        Path(log_path).mkdir(parents=True, exist_ok=True)
        log_out = Path(log_path).resolve() / "out.log"
        log_err = Path(log_path).resolve() / "err.log"

        with open(str(log_out), "w") as f_out, open(str(log_err), "w") as f_err:
            return subprocess.run(
                command,
                shell=shell,
                check=check,
                stdout=f_out,
                stderr=f_err,
                text=text,
            )
    else:
        stdout = subprocess.PIPE if hide else None
        stderr = subprocess.PIPE if hide else None

        return subprocess.run(
            command,
            shell=shell,
            check=check,
            stdout=stdout,
            stderr=stderr,
            text=text,
        )


def post_install(
    executable: str | os.PathLike | None = None,
    editable: bool = False,
    version: str | None = None,
    path: str | None = None,
    torch_index_url: str | None = None,
):
    """Install additional dependencies

    Args:
        executable (str | os.PathLike | None, optional): Path to Blender executable. Defaults to one found on $PATH.
        editable (bool, optional): If set, install current visionsim as editable in blender. Only works if
            visionsim is already installed as editable locally.
        version (str | None, optional): The version of visionsim to install. Setting this is akin to specifying the version
            when pip installing. If set, a fresh copy from PyPI will be installed inside blender's runtime environment,
            which might not match the currently installed version. Defaults to None (use currently installed version).
        path (str | None, optional): The path to the visionsim source code to install, only used when --version is not set.
        torch_index_url (str | None, optional): PyTorch wheel index to install torch from, e.g.
            ``https://download.pytorch.org/whl/cpu`` for a machine without CUDA. Defaults to None (whatever PyPI resolves to).
    """
    from visionsim.simulate import install_dependencies

    if _run(f"{executable or 'blender'} --version", shell=True, hide=True).returncode != 0:
        raise RuntimeError(
            "No blender installation found on path! Please make sure it is discoverable, or specify executable."
        )
    install_dependencies(executable, editable=editable, version=version, path=path, torch_index_url=torch_index_url)


_CompletionShell = Literal["bash", "zsh", "fish"]
_COMPLETION_SHELLS: tuple[_CompletionShell, ...] = get_args(_CompletionShell)
# Keep in sync with [project.scripts] in pyproject.toml.
_COMPLETION_PROGS = ("visionsim", "vsim")


def _completion_dir(shell: _CompletionShell) -> Path:
    if shell == "bash":
        base = os.environ.get("BASH_COMPLETION_USER_DIR")
        root = Path(base) if base else Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
        return root / "bash-completion" / "completions"
    if shell == "zsh":
        return Path.home() / ".zfunc"
    if shell == "fish":
        return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "fish" / "completions"
    raise ValueError(f"Unsupported shell: {shell}")


def _completion_filename(shell: _CompletionShell, prog: str) -> str:
    if shell == "bash":
        return prog
    if shell == "zsh":
        return f"_{prog}"
    if shell == "fish":
        return f"{prog}.fish"
    raise ValueError(f"Unsupported shell: {shell}")


def _detect_shell() -> _CompletionShell:
    shell = Path(os.environ.get("SHELL", "")).name
    if shell not in _COMPLETION_SHELLS:
        raise ValueError(
            f"Could not detect a supported shell from $SHELL {shell!r}; pass --shell one of {_COMPLETION_SHELLS}."
        )
    return shell


def _completion_script(shell: _CompletionShell, prog: str) -> str:
    # tyro only exposes generation through its CLI flag, which prints and then exits.
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            tyro.extras.subcommand_cli_from_dict(_cli_dict(), prog=prog, args=["--tyro-write-completion", shell, "-"])
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    script = buffer.getvalue()

    # tyro prints multi-line help text verbatim, so descriptions become extra completion
    # candidates that break prefix matching (bash computes an empty common prefix).
    script = script.replace(
        "    for completion in completions:\n        print(completion)",
        "    for completion in completions:\n        print(*completion.splitlines())",
    )

    if shell == "fish":
        # tyro 1.0.16 passes the spec as a single -c argument, which exceeds Linux's
        # 128 KiB per-argument limit for this CLI; feed it through stdin instead.
        script = script.replace('$python_cmd -c "$pycode"', "printf '%s' \"$pycode\" | $python_cmd -")

    return script


def completion(
    mode: Literal["install", "uninstall"] = "install",
    shell: _CompletionShell | None = None,
    path: str | os.PathLike | None = None,
):
    """Install or remove tab-completion scripts for the current shell.

    Args:
        mode (Literal["install", "uninstall"], optional): Whether to write or delete completion scripts.
            Defaults to "install".
        shell (Literal["bash", "zsh", "fish"] | None, optional): Shell to generate completions for.
            Defaults to the basename of ``$SHELL``.
        path (str | os.PathLike | None, optional): Directory to write completion scripts to instead of the
            shell's standard user completion directory. Defaults to None.
    """
    if sys.platform == "win32":
        raise RuntimeError("Shell completion is not supported on Windows.")

    shell = shell or _detect_shell()
    target_dir = Path(path) if path is not None else _completion_dir(shell)

    if mode == "uninstall":
        for prog in _COMPLETION_PROGS:
            target = target_dir / _completion_filename(shell, prog)
            if target.exists():
                target.unlink()
                _log.info(f"Removed completion script {target}")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    for prog in _COMPLETION_PROGS:
        target = target_dir / _completion_filename(shell, prog)
        target.write_text(_completion_script(shell, prog))
        _log.info(f"Wrote completion script {target}")

    if shell == "zsh":
        _log.info("Ensure ~/.zfunc is on fpath: fpath+=~/.zfunc; autoload -Uz compinit && compinit")


def _cli_dict() -> dict[str, Any]:
    cli_dict: dict[str, Any] = {"post-install": post_install, "completion": completion}

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
    return cli_dict


def main():
    # Capture bare script name to keep help text short and accurate
    script, *_ = sys.argv
    tyro.extras.subcommand_cli_from_dict(_cli_dict(), prog=Path(script).name)
