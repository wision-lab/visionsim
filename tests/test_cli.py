import inspect
import sys
from pathlib import Path

import pytest
from docstring_parser import parse_from_object

from visionsim.cli import _cli_modules, _run


@pytest.mark.skipif("win" in sys.platform, reason="No autocomplete on windows")
def test_completions(tmpdir):
    # Note: If this test fails, it most likely means one of the arguments of a CLI method is
    #   not annotated properly. Logs will be saved to the temp dir, and should show what's going on.
    shell_name = Path(_run("echo $SHELL", shell=True, check=True).stdout.strip()).stem

    if shell_name not in ("bash", "zsh", "tcsh"):
        pytest.skip(f"Unsupported shell, got {shell_name}, expected on of 'bash', 'zsh', 'tcsh'.")

    _run(rf"visionsim --tyro-write-completion {shell_name} {tmpdir}/visionsim", check=True, shell=True, log_path=tmpdir)


@pytest.mark.parametrize("module", _cli_modules)
def test_help_is_full(module):
    for func_name, func in inspect.getmembers(module, inspect.isfunction):
        if func.__module__ == module.__name__ and not func_name.startswith("_"):
            docs = parse_from_object(func)
            documented_params = set(param.arg_name for param in docs.params if param.description)
            all_params = set(inspect.getfullargspec(func).args)

            assert documented_params == all_params
