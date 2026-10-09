Development
===========

Editable Install
----------------

We recommend `uv <https://docs.astral.sh/uv/>`_ for development. It resolves against the
committed ``uv.lock`` and runs every tool in the project environment, so you never have to
activate a virtualenv or worry about which interpreter owns ``invoke``. Clone the repository,
navigate to it and run::

    uv sync --all-groups

Then prefix every command with ``uv run --all-groups``, for example::

    uv run --all-groups inv test

``--all-groups`` is not optional: the toolchain (``invoke``, ruff, mypy, pytest) lives in the
``dev`` group and the documentation build in ``docs``, so a bare ``uv run invoke`` may not find
``invoke`` at all.

If you would rather not type ``uv run --all-groups`` for every command, source the environment
instead and drop the prefix::

    source .venv/bin/activate
    inv test

``uv sync`` creates ``.venv`` in the project root, so activating it directly works the same way.
The ``uv run`` form is preferable in scripts and CI because it needs no activation step and picks
up the lockfile each time.

If you cannot use uv, install into an environment you have activated yourself. The package must be
installed **editable** either way, because Blender is told to install visionsim from this
interpreter's import path::

    pip install -e . --group dev --group docs

Similarly, to install visionsim in an editable manner within Blender's runtime, you can do the following::

    uv run --all-groups visionsim post-install --editable

Each Blender version ships its own Python interpreter and site-packages, so ``post-install`` must
be re-run for every version you want to render or test with. The tests do this automatically for
the ``--executable`` they are given.

| 

Running Tests
-------------

We use pytest for testing. The whole suite runs from the project root with ``inv test``; since
the toolchain lives in the development group, invoke it through uv like so::

    uv run --all-groups inv test --executable=<path-to-blender>

To run only part of the suite, pass ``--paths`` after ``inv test`` (repeatable, resolved relative
to the repo root) or call pytest directly::

    uv run --all-groups inv test --paths tests/simulate/test_render.py --executable=<path-to-blender>
    uv run --all-groups pytest tests/simulate/test_render.py -rP --executable=<path-to-blender>

``inv test`` reports the slowest tests by default, which is useful because the Blender fixtures
dominate the runtime. Prefer it over raw pytest when the Blender-side dependencies may be stale:
the test fixtures re-run ``install_dependencies`` themselves, so raw ``pytest`` skips that repair
and can fail on version skew between the client and Blender's Python.

To ensure that there's no conflicts due to different versions of the libraries between the server/client sides, a editable ``post-install`` task is run when starting the tests.

The ``-rP`` option is also helpful for seeing any stdout messages that are otherwise hidden. 

Some tests (``tests/simulate/test_playblast.py``) exercise Blender's viewport renderer, which opens a real window and needs a GL context, so they are skipped when no display is set. To run them headlessly, install ``xvfb`` and start a virtual display before the test command::

    DISPLAY="" WAYLAND_DISPLAY="" xvfb-run -a --server-args="-screen 0 1920x1080x24" \
        uv run --all-groups pytest tests/simulate/test_playblast.py

| 

Running CI Locally
------------------

You can run the CI locally using the `ACT CLI <https://github.com/nektos/act>`_, or use it's `vscode extension <https://sanjulaganepola.github.io/github-local-actions-docs/>`_ as a front end. Using the CLI, you can run all workflows that trigger on a push using `act push`. The following command will run the workflows and, if they fail, open an interactive shell into the latest container::
    
    act push || docker exec -it `docker ps -q | head -n1` bash  

|

Building the Documentation
--------------------------

In the project root, with visionsim installed with the dev dependencies, run::

    inv clean build-docs --preview

Sphinx does not run any figure recipe. Every documentation page that has generated figures gets its own task under the ``figures`` namespace, whose tree mirrors the documentation tree below ``docs/source`` (dots separate subdirectories). Only that page's figures are rebuilt, and only those that are stale, so re-running a page task is cheap::

    inv --list                     # figures.quick-start, figures.sections.interpolation, ...
    inv figures.quick-start
    inv figures.sections.interpolation
    inv figures.sections.sensors.itof
    inv figures.tutorials.playblast

Using ``--dry-run`` reports what would happen without writing anything, and ``--force`` rebuilds the page's figures and their dependencies.

A page is one dependency graph. Its module lists only the figures the page owns, and each figure names the intermediates it needs through ``requires``. Shared intermediates, such as the rendered and interpolated datasets several quick-start figures read, live in ``_page.INTERMEDIATES`` and are provisioned at most once per invocation. Each node is attempted once. A node whose requirement failed is skipped with one line naming that requirement, so a failed render is reported once at its root instead of failing again for every dependent. Cycles and unknown requirement names are rejected when the task loads, so ``inv --list`` fails instead of a run half-way through.

Figures and intermediates that render need ``cache/lego.blend``, obtained manually or with `gdown <https://github.com/wkentaro/gdown>`_ using the ``--fuzzy --folder`` command in ``examples/README.md``; without it the blender command fails and the node is reported as skipped.

Some tasks use ``gifski`` to encode preview gif which is not a Python dependency. Install with ``cargo install gifski``, or a package from https://gif.ski.

Paths resolve from the repository root, so a linked worktree starts with an empty ``cache/`` and re-renders everything, even though the main checkout already holds the datasets. Sharing one cache avoids that::

    ln -s /path/to/main/checkout/cache cache

|

Dev tools
---------

We're using `invoke <https://docs.pyinvoke.org/en/stable/>`_ to manage common development and housekeeping tasks.

Make sure you have invoke installed then you can run any of the following `tasks` from the project root:

.. command-output:: invoke --list

It's also recommended using the pre-commit hook that will lint/test/clean the code before every commit. For this make sure that `invoke` and `pre-commit` are installed (via pip) and then install the pre-hooks with::

    pre-commit install

See `pre-commit <https://pre-commit.com/#intro>`_ for more.

| 

Release Process
---------------

To prepare for a new release, first ensure all tests, linting, formatting and typing checks pass, and update the documentation and version numbers accordingly. Then you'll need to build the new source distribution and push it to PyPI using twine. 

The up-to-date source on this is the `python package authority <https://packaging.python.org/en/latest/tutorials/packaging-projects>`_, but you'll have to first build the source distribution using::

    python -m build

Then upload it to PyPI with twine::

    python -m twine upload dist/*
