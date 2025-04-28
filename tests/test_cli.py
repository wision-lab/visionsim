import sys
from pathlib import Path

import pytest

from visionsim.cli.common import _run


@pytest.mark.skipif(sys.platform.contains("win"), reason="No autocomplete on windows")
def test_completions(tmpdir):
    # Note: If this test fails, it most likely means one of the arguments of a CLI method is
    #   not annotated properly. Logs will be saved to the temp dir, and should show what's going on.
    shell_name = Path(_run("echo $SHELL", shell=True, check=True).stdout.strip()).stem

    if shell_name not in ("bash", "zsh", "tcsh"):
        pytest.skip(f"Unsupported shell, got {shell_name}, expected on of 'bash', 'zsh', 'tcsh'.")

    _run(rf"visionsim --tyro-write-completion {shell_name} {tmpdir}/visionsim", check=True, shell=True, log_path=tmpdir)


# def test_list_has_all_tasks():
#     # Note: We cannot mock the context or runner here as we are if visionsim cli
#     #   was properly installed and is on PATH. Also, we need to set `in_stream`
#     #   to False otherwise invoke reads from stdin and interferes with pytest.
#     c = Context()
#     tasks = ns.task_names.keys()
#     result = _run(c, "visionsim --list", hide=True, in_stream=False)
#     assert all(t in result.stdout.strip() for t in tasks)


# @pytest.mark.parametrize("task", ns.task_names.keys())
# def test_task_has_doc(task):
#     assert ns[task].__doc__


# @pytest.mark.parametrize("task", ns.task_names.keys())
# def test_task_help_has_defaults(task):
#     help_dict = dict((ns[task].help or {}).items())
#     params = {name: (p.kind, p.default) for name, p in ns[task].argspec(ns[task].body).parameters.items()}

#     for name, (kind, default) in params.items():
#         # We should be able to check if kind == KEYWORD, but for some
#         # reason they all show up as POSITIONAL_OR_KEYWORD...
#         if default is not inspect.Parameter.empty:
#             assert "default" in help_dict[name].lower()


# @pytest.mark.parametrize("task", ns.task_names.keys())
# def test_task_help_is_full(task):
#     help_dict = dict((ns[task].help or {}).items())
#     func_args = set(ns[task].argspec(ns[task].body).parameters.keys())

#     assert all(help_dict.values())
#     assert set(help_dict.keys()) == func_args
