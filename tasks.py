"""
Tasks for maintaining the project.

Execute 'inv[oke] --list' for a list of dev tasks.
"""

import fnmatch
import glob
import json
import os
import platform
import shutil
import webbrowser
from functools import partial
from pathlib import Path

import numpy as np
from invoke import task
from rich.progress import Progress

from visionsim.simulate.blender import BlenderClient

ROOT_DIR = Path(__file__).parent
TEST_DIR = ROOT_DIR / "tests"
SOURCE_DIR = ROOT_DIR / "visionsim"
EXAMPLE_DIR = ROOT_DIR / "examples"
SCRIPTS_DIR = ROOT_DIR / "scripts"
COVERAGE_FILE = ROOT_DIR / ".coverage"
COVERAGE_DIR = ROOT_DIR / "htmlcov"
COVERAGE_REPORT = COVERAGE_DIR / "index.html"
PYTHON_DIRS = [str(d) for d in [SOURCE_DIR, TEST_DIR, EXAMPLE_DIR, SCRIPTS_DIR]]
DOCS_DIR = ROOT_DIR / "docs"
DOCS_INDEX = DOCS_DIR / "build" / "html" / "index.html"
DOCS_STATIC = DOCS_DIR / "source" / "_static"


def _delete_file(file, except_patterns=None):
    if os.path.isfile(file):
        print(f"Removing file {file}...")
        os.remove(file)
    elif os.path.isdir(file):
        if except_patterns is None:
            print(f"Removing directory {file}...")
            shutil.rmtree(file, ignore_errors=True)
        else:
            print(f"Purging directory {file}...")
            for dirpath, dirnames, filenames in os.walk(file):
                # Remove regular files, ignore directories
                for filename in filenames:
                    file = os.path.join(dirpath, filename)
                    if any(fnmatch.fnmatch(file, pattern) for pattern in except_patterns):
                        print(f"\tKeeping file {file}...")
                    else:
                        print(f"\tRemoving file {file}...")
                        os.remove(file)
                # Remove empty directories
                if not dirnames and not filenames:
                    print(f"\tRemoving directory {file}...")
                    shutil.rmtree(dirpath, ignore_errors=True)


def _delete_pattern(pattern):
    for file in glob.glob(os.path.join("**", pattern), recursive=True):
        _delete_file(file)


def _run(c, command):
    return c.run(command, pty=platform.system() != "Windows")


@task
def format(c):
    """Format code (and sort imports)"""
    python_dirs_string = " ".join(PYTHON_DIRS + glob.glob(os.path.join(ROOT_DIR, "*.py")))
    _run(c, f"ruff check --select I --fix {python_dirs_string}")
    _run(c, f"ruff format {python_dirs_string}")


@task
def lint(c):
    """Lint code with ruff"""
    _run(c, f"ruff check --extend-select I {' '.join(PYTHON_DIRS)}")


@task
def test(c):
    """Run tests"""
    _run(c, "pytest")


@task
def precommit(c):
    """Run pre-commit hooks"""
    _run(c, "pre-commit run --all-files")


@task
def coverage(c):
    """Create coverage report"""
    _run(c, f"coverage run --source {SOURCE_DIR} -m pytest")
    _run(c, "coverage report")

    # Build a local report
    _run(c, "coverage html")
    webbrowser.open(COVERAGE_REPORT.as_uri())


@task
def build_docs(c, preview=False, full=False):
    """Confirm docs can be built"""
    if full:
        # Create examples from the quick start guide
        Path("cache/quickstart").mkdir(exist_ok=True, parents=True)
        if not Path("data/lego.blend").exists():
            print("File `data/lego.blend` not found, you can get it by running the command:")
            print("gdown https://drive.google.com/file/d/1XPkeA0ENljAjk4D9PGHpMzzr3GF_rECa/view?usp=drive_link --fuzzy")
            return

        with (
            BlenderClient.spawn(timeout=30, executable="flatpak run --die-with-parent org.blender.Blender") as client,
            Progress() as progress,
        ):
            client.initialize(Path("data/lego.blend").resolve(), Path("cache/quickstart/lego-gt/").resolve())
            client.unbind_camera()

            for frame, theta in enumerate(np.linspace(0, 2 * np.pi, 100, endpoint=False)):
                client.position_camera(location=[5 * np.cos(theta), 5 * np.sin(theta), 1], look_at=[0, 0, 0])
                client.set_camera_keyframe(frame)
            client.set_animation_range(start=0, stop=100)
            client.move_keyframes(scale=5.0)
            client.set_resolution((320, 320))
            task = progress.add_task("Rendering lego.blend...")
            transforms = client.render_animation(update_fn=partial(progress.update, task))

            with open(str(Path("cache/quickstart/lego-gt/").resolve() / "transforms.json"), "w") as f:
                json.dump(transforms, f, indent=2)

        _run(
            c,
            f"gifski $(ls -1a cache/quickstart/lego-gt/frames/*.png | sed -n '1~5p') --fps 25 -o {DOCS_STATIC}/lego-gt-preview.gif",
        )

        _run(c, "visionsim interpolate.frames cache/quickstart/lego-gt/ -o cache/quickstart/lego-interp/ -n=32")

        _run(
            c,
            "visionsim emulate.rgb cache/quickstart/lego-interp/ -o cache/quickstart/lego-rgb25fps/ --chunk-size=160 --readout-std=0 --force",
        )
        _run(
            c, f"gifski cache/quickstart/lego-rgb25fps/frames/*.png --fps 25 -o {DOCS_STATIC}/lego-rgb25fps-preview.gif"
        )

        _run(
            c,
            "visionsim emulate.spad cache/quickstart/lego-interp/ -o cache/quickstart/lego-spc4kHz/ --mode=img --force",
        )
        _run(
            c,
            f"gifski $(ls -1a cache/quickstart/lego-spc4kHz/frames/*.png | sed -n '1~160p') --fps 25 -o {DOCS_STATIC}/lego-spc4kHz-preview.gif",
        )

        # Create interpolation examples
        Path("cache/interpolation").mkdir(exist_ok=True, parents=True)
        for i, n in enumerate((25, 50, 100, 200)):
            _run(
                c,
                f"visionsim blender.render data/lego.blend cache/interpolation/lego-{n:04}/ --num-frames={n} --width=320 --height=320",
            )
            _run(
                c,
                f"visionsim interpolate.frames cache/interpolation/lego-{n:04}/ -o cache/interpolation/lego{n:04}-interp/ -n={int(64 / 2**i)}",
            )
            _run(
                c,
                f"gifski $(ls -1a cache/interpolation/lego{n:04}-interp/frames/*.png | sed -n '1~8p') --fps 25 -o {DOCS_STATIC}/lego{n:04}-interp.gif",
            )
    # Run autodocs
    with c.cd(ROOT_DIR):
        # TODO: Make this a project configuration
        api_exclude = ["visionsim/tasks", "visionsim/interpolate/rife"]
        # We have to do this for all the new changes in the docs to be reflected
        print('\033[93m' + "Make sure to run pip install -e . or equivalent to make sure docstring changes are reflected" + '\033[0m')
        # Generate API docs
        _run(c, "sphinx-apidoc -f --remove-old -o docs/source/apidocs visionsim " + " ".join(api_exclude))
        # Generate CLI docs
        _run(c, "sphinx-apidoc -f --remove-old -o docs/source/clidocs visionsim/tasks")
    with c.cd(DOCS_DIR):
        _run(c, "make html")

    if preview:
        webbrowser.open(DOCS_INDEX.as_uri())


@task
def clean_build(c):
    """Clean up files from package building"""
    _delete_file("build/")
    _delete_file("dist/")
    _delete_file(".eggs/")
    _delete_pattern("*.egg-info")
    _delete_pattern("*.egg")


@task
def clean_python(c):
    """Clean up python file artifacts"""
    _delete_pattern("__pycache__")
    _delete_pattern("*.pyc")
    _delete_pattern("*.pyo")
    _delete_pattern("*~")


@task
def clean_tests(c):
    """Clean up files from testing"""
    _delete_file(COVERAGE_FILE)
    _delete_file(COVERAGE_DIR)
    _delete_pattern(".pytest_cache")


@task
def clean_docs(c):
    """Clean up docs build"""
    with c.cd(DOCS_DIR):
        _run(c, "make clean")


@task(pre=[clean_build, clean_python, clean_tests, clean_docs])
def clean(c):
    """Runs all clean sub-tasks"""
    _run(c, "ruff clean")
