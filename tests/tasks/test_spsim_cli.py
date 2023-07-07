import pytest
from invoke.context import Context

from spsim.tasks import ns
from spsim.tasks.common import _run


def test_list_has_all_tasks():
    # Note: We cannot mock the context or runner here as we are if spsim cli
    #   was properly installed and is on PATH. Also, we need to set `in_stream`
    #   to False otherwise invoke reads from stdin and interferes with pytest.
    c = Context()
    tasks = ns.task_names.keys()
    result = _run(c, "spsim --list", hide=True, in_stream=False)
    assert all(t in result.stdout.strip() for t in tasks)


@pytest.mark.parametrize("task", ns.task_names.keys())
def test_task_has_doc(task):
    assert ns[task].__doc__


@pytest.mark.parametrize("task", ns.task_names.keys())
def test_task_help_is_full(task):
    # Copy help dict as `get_arguments` consumes it.
    help_dict = dict((ns[task].help or {}).items())
    func_args = set(arg.name for arg in ns[task].get_arguments())

    assert all(help_dict.values())
    assert set(help_dict.keys()) == func_args
