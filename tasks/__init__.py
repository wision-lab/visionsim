"""Tasks for maintaining the project."""

from __future__ import annotations

import ast
import copy
import fnmatch
import glob
import os
import platform
import shlex
import shutil
import sys
import webbrowser
from pathlib import Path

from invoke import Collection, Task, task
from rich.console import Console

console = Console()
ROOT_DIR = Path(__file__).parent.parent.resolve()
TEST_DIR = ROOT_DIR / "tests"
SOURCE_DIR = ROOT_DIR / "visionsim"
EXAMPLE_DIR = ROOT_DIR / "examples"
SCRIPTS_DIR = ROOT_DIR / "scripts"
TASKS_DIR = ROOT_DIR / "tasks"
COVERAGE_FILE = ROOT_DIR / ".coverage"
COVERAGE_DIR = ROOT_DIR / "htmlcov"
COVERAGE_REPORT = COVERAGE_DIR / "index.html"
DOCS_DIR = ROOT_DIR / "docs"
DOCS_INDEX = DOCS_DIR / "build" / "html" / "index.html"
PYTHON_DIRS = [str(d) for d in [SOURCE_DIR, TEST_DIR, EXAMPLE_DIR, SCRIPTS_DIR, DOCS_DIR, TASKS_DIR]]


def _delete_file(file, except_patterns=None):
    """Delete a file, or a directory tree, printing what is removed.

    Args:
        file: Path to remove. Files are unlinked, directories are removed
            recursively.
        except_patterns: If given, prune directories in place instead of removing
            them, keeping any file matching one of these glob patterns.
    """
    if os.path.isfile(file):
        console.print(f"Removing file {file}.")
        os.remove(file)
    elif os.path.isdir(file):
        if except_patterns is None:
            console.print(f"Removing directory {file}.")
            shutil.rmtree(file, ignore_errors=True)
        else:
            console.print(f"Purging directory {file}.")
            for dirpath, dirnames, filenames in os.walk(file):
                for filename in filenames:
                    file = os.path.join(dirpath, filename)
                    if any(fnmatch.fnmatch(file, pattern) for pattern in except_patterns):
                        console.print(f"\tKeeping file {file}.")
                    else:
                        console.print(f"\tRemoving file {file}.")
                        os.remove(file)
                if not dirnames and not filenames:
                    console.print(f"\tRemoving directory {file}.")
                    shutil.rmtree(dirpath, ignore_errors=True)


def _delete_pattern(pattern):
    """Delete every file whose path matches ``pattern``, recursively.

    Args:
        pattern: Glob passed to :func:`glob.glob` with ``recursive=True``.
    """
    for file in glob.glob(os.path.join("**", pattern), recursive=True):
        _delete_file(file)


def _run(c, command, **kwargs):
    """Run a shell command through invoke, allocating a pty where supported.

    Args:
        c: The invoke context.
        command: Shell command to run.
        **kwargs: Forwarded to ``c.run``.

    Returns:
        The invoke ``Result``.
    """
    return c.run(command, pty=platform.system() != "Windows", **kwargs)


@task
def format(c, check=False):
    """Format code and sort imports with ruff.

    Checks ``visionsim/``, ``tests/``, ``examples/``, ``scripts/``, ``docs/`` and
    ``tasks/``, plus any top-level ``*.py`` files.

    Args:
        check: Report what would change without writing anything, exiting non-zero
            if any file is unformatted or has unsorted imports. Used by the
            ``inv-format`` pre-commit hook; omit it to actually rewrite the files.
    """
    python_dirs_string = " ".join(PYTHON_DIRS + glob.glob(os.path.join(ROOT_DIR, "*.py")) + [__file__])
    fix = "" if check else "--fix"
    _run(c, f"ruff check --select I {fix} {python_dirs_string}")
    _run(c, f"ruff format {'--check' if check else ''} {python_dirs_string}")


@task
def lint(c):
    """Lint the code with ruff, including import order.

    Reports issues without modifying any files, so it is safe to run in CI. Use
    ``format`` to fix what it finds.
    """
    _run(c, f"ruff check --extend-select I {' '.join(PYTHON_DIRS)} {__file__}")


@task(iterable=["paths"])
def test(c, executable=None, paths=None):
    """Run the test suite with pytest.

    Passes ``-s`` so test output, including anything printed by a failing test,
    goes straight to the terminal instead of being captured, and reports the
    slowest tests since the Blender fixtures dominate the runtime.

    Args:
        executable: Path to Blender executable. Defaults to ``$VSIM_BLENDER`` if
            set, otherwise to a Blender found on $PATH. The environment variable
            exists so the pre-commit hook can pin a specific build without a shell.
        paths: Optional test files or directories to run, relative to the repo
            root. May be given more than once. Defaults to the whole ``tests/``
            tree.
    """
    executable = executable or os.environ.get("VSIM_BLENDER")
    targets = [shlex.quote(str(ROOT_DIR / path)) for path in paths or []] or [shlex.quote(str(TEST_DIR))]
    command = f"pytest -s --durations=0 -c {shlex.quote(str(ROOT_DIR / 'pyproject.toml'))} {' '.join(targets)}"
    if executable:
        command += f" --executable {shlex.quote(executable)}"
    _run(c, command)


@task
def test_stubs(c):
    """Check the generated blender type stubs against the real module with stubtest.

    Catches drift between ``visionsim/simulate/blender.pyi`` and the attributes
    the process actually exposes; regenerate with ``generate-stubs`` when it fails.
    """
    _run(c, "stubtest visionsim.simulate.blender --concise --ignore-disjoint-bases")


@task
def type_check(c):
    """Type-check the package and the development tasks with mypy.

    Covers ``visionsim/`` and everything under ``tasks/``. The tests and the
    dashboards are not checked.
    """
    flags = "--follow-untyped-imports" if sys.version_info < (3, 10, 0) else ""
    _run(c, f"mypy {SOURCE_DIR} {TASKS_DIR} {flags}")


@task
def precommit(c):
    """Run every pre-commit hook against all files, not just staged ones."""
    _run(c, "pre-commit run --all-files")


@task
def coverage(c):
    """Run the test suite under coverage and open the HTML report.

    Measures ``visionsim/`` only, prints the terminal summary, writes
    ``htmlcov/`` and opens ``htmlcov/index.html`` in the browser.
    """
    _run(c, f"coverage run --source {SOURCE_DIR} -m pytest")
    _run(c, "coverage report")
    _run(c, "coverage html")
    webbrowser.open(COVERAGE_REPORT.as_uri())


@task
def build_docs(c, preview=False):
    """Build the Sphinx documentation, regenerating the API and CLI stubs first.

    Run ``pip install -e .[dev]`` (or equivalent) beforehand, otherwise docstring
    edits will not show up in the generated pages. Pass ``--preview`` to open the
    result in the browser when the build succeeds.
    """
    with c.cd(ROOT_DIR):
        api_exclude = ["visionsim/interpolate/rife", "visionsim/simulate/compat.py", "visionsim/simulate/nodes"]
        # We have to do this for all the new changes in the docs to be reflected
        console.print(
            '[yellow]Make sure to run "pip install -e .[dev]" or equivalent to make sure docstring changes are reflected!'
        )
        # Generate API, CLI docs
        _run(c, "sphinx-apidoc -f --remove-old -o docs/source/apidocs visionsim " + " ".join(api_exclude))
        Path("docs/source/apidocs/modules.rst").unlink()

    with c.cd(DOCS_DIR):
        _run(c, "make html")

    if preview:
        webbrowser.open(DOCS_INDEX.as_uri())


@task
def generate_stubs(c):
    """Regenerate ``visionsim/simulate/blender.pyi`` from the blender module.

    Writes a stub with stubgen, rewrites it so ``BlenderClient`` methods mirror the
    ``exposed_*`` methods on ``BlenderService`` (with per-client return types), and
    formats the result. The stubs give autocomplete for attributes that only exist
    inside a running blender process; ``test-stubs`` checks them against it.
    """
    source_path = "visionsim/simulate/blender.py"
    stub_path = "visionsim/simulate/blender.pyi"
    _run(c, f"stubgen {source_path} --include-docstrings --include-private -o .")

    class FixStub(ast.NodeTransformer):
        def __init__(self, path, ignores_from: str | None = None):
            if ignores_from:
                with open(ignores_from) as f:
                    root = ast.parse(f.read(), ignores_from, type_comments=True)

                self.type_ignored_mods = {
                    n.name.split(".")[0]
                    for node in ast.walk(root)
                    if isinstance(node, (ast.Import, ast.ImportFrom))
                    for n in node.names
                    if any(
                        ignore.lineno in range(node.lineno, (node.end_lineno or node.lineno) + 1)
                        for ignore in root.type_ignores
                    )
                }
            else:
                self.type_ignored_mods = set()
            self.type_ignores: set[int] = set()
            self.classes: dict[str, ast.ClassDef] = {}

            with open(path) as f:
                self.root = ast.parse(code := f.read(), path, type_comments=True)
                self.code = code
            type_check_only = ast.ImportFrom(module="typing", names=[ast.alias(name="type_check_only")], level=0)
            self.root.body.insert(0, type_check_only)
            ast.increment_lineno(self.root)
            super().__init__()

        def visit_Import(self, node):
            self.generic_visit(node)

            if any(ignored in n.name for ignored in self.type_ignored_mods for n in node.names):
                self.type_ignores.add(node.lineno - 1)

            # Remove redundant (from Y) import X as X
            # Convert to (from Y) import X
            for name in node.names:
                if name.asname and name.asname == name.name:
                    name.asname = None
            return node

        def visit_ImportFrom(self, node):
            return self.visit_Import(node)

        def visit_ClassDef(self, node):
            self.generic_visit(node)

            # Requires BlenderService to precede every BlenderClient(s) in source order
            if "BlenderClient" in node.name and "BlenderService" in self.classes:
                methods = {n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)}

                for child in ast.walk(self.classes["BlenderService"]):
                    if isinstance(child, ast.FunctionDef) and child.name.startswith("exposed_"):
                        name = child.name.replace("exposed_", "")

                        if name not in methods:
                            child = copy.deepcopy(child)
                            child.decorator_list = [ast.Name(id="type_check_only", ctx=ast.Load())]
                            child.name = name

                            if (
                                node.name == "BlenderClients"
                                and child.returns is not None
                                and not (isinstance(child.returns, ast.Constant) and child.returns.value is None)
                            ):
                                # BlenderClients wraps every return type in tuple[...], skipping None
                                child.returns = ast.Subscript(
                                    value=ast.Name(id="tuple", ctx=ast.Load()),
                                    slice=ast.Tuple(elts=[child.returns]),
                                    ctx=ast.Load(),
                                )
                            node.body.append(child)

            self.classes[node.name] = node
            return node

        def transform(self):
            tree = self.visit(self.root)
            code = ast.unparse(tree).splitlines()

            for ignore in self.type_ignores:
                code[ignore] += "  #type: ignore"
            return "\n".join(code)

    code = FixStub(stub_path, ignores_from=source_path).transform()

    with open(stub_path, "w") as f:
        f.write(code)

    _run(c, f"ruff check --select I --fix {stub_path}")
    _run(c, f"ruff check --extend-select I --fix {stub_path}")
    _run(c, f"ruff format {stub_path}")


@task
def clean_build(c):
    """Remove build artifacts: ``build/``, ``dist/``, ``.eggs/`` and ``*.egg-info``."""
    _delete_file("build/")
    _delete_file("dist/")
    _delete_file(".eggs/")
    _delete_pattern("*.egg-info")
    _delete_pattern("*.egg")


@task
def clean_python(c):
    """Remove compiled Python: ``__pycache__``, ``*.pyc``, ``*.pyo`` and editor backups."""
    _delete_pattern("__pycache__")
    _delete_pattern("*.pyc")
    _delete_pattern("*.pyo")
    _delete_pattern("*~")


@task
def clean_tests(c):
    """Remove test artifacts: ``.coverage``, ``htmlcov/`` and ``.pytest_cache``."""
    _delete_file(COVERAGE_FILE)
    _delete_file(COVERAGE_DIR)
    _delete_pattern(".pytest_cache")


@task
def clean_docs(c):
    """Remove the built documentation by running ``make clean`` in ``docs/``."""
    with c.cd(DOCS_DIR):
        _run(c, "make clean")


@task(pre=[clean_build, clean_python, clean_tests, clean_docs])
def clean(c):
    """Remove build, Python, test and docs artifacts, then clear the ruff cache.

    Runs ``clean-build``, ``clean-python``, ``clean-tests`` and ``clean-docs``
    first.
    """
    _run(c, "ruff clean")


from . import figures

# Defining `ns` at module scope would otherwise shadow the top-level tasks, so every
# Task defined above is re-registered by name first.
ns = Collection()
for _obj in list(globals().values()):
    if isinstance(_obj, Task):
        ns.add_task(_obj)
ns.add_collection(Collection.from_module(figures, name="figures"))
