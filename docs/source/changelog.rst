Changelog
=========

All notable changes to VisionSIM are documented here. The format follows
`Keep a Changelog <https://keepachangelog.com/en/1.1.0/>`_ and the project uses
`Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
^^^^^

- Indirect time-of-flight (iToF) sensor emulation, with a tutorial and figure task.
- An event camera (DVS) emulation tutorial covering the CLI, output format, Python API and parameter figures.
- Standalone ``render-playblast`` CLI for fast viewport previews, plus a playblast tutorial.
- HDR, stereo, light-pass, and large-dataset tutorials backed by per-page figure tasks.
- A grouped tutorials index (Sensors, Rendering & Simulation, Applications) and new
  camera-models, optimize-rate, changelog and troubleshooting pages.
- A varying-focal-lengths tutorial with a rendered contrazoom preview, and an IMU trajectory
  preview built from five integrated runs.
- Materials, diffuse, specular, and position/point-map render passes, with a ``dataset.to-pointcloud`` CLI.
- ``transforms.db`` metadata backed by peewee, alongside the existing ``transforms.json``.
- Pre-commit format check, mypy configuration, and type checking for the package and tasks.

Changed
^^^^^^^

- Rebuilt the documentation around per-page figure tasks, sphinx-design grids for figure rows,
  and UTF-8 reading of doc sources.
- Blender 5 support for nodes and fcurves, with the Blender version selected through
  ``VSIM_BLENDER`` or ``--executable``.
- Dataset format evolution (#26): material and segmentation refactor, transforms coordinate
  convention fix, and dataset paths resolved through a glob pattern.
- Interpolation gained scale decay, a stall watchdog with debug dump, and rate/resolution
  defaults tuned for long jobs.
- IMU emulation converts ``transforms.json`` pose matrices out of the OpenGL camera frame
  by default, since those matrices map OpenGL camera coordinates directly into Blender's
  world axes and are not plain rotations. ``--no-convert-coords`` emulates them as-is, and
  the coordinate helpers were renamed to ``camcoord_opengl_to_blender`` and
  ``camcoord_blender_to_opengl``.
- Split CI into separate docs and tests workflows, build the Docker image on push, and run
  viewport tests under a virtual display.

Fixed
^^^^^

- Event generation blur and pattern handling, ffprobe output capture, and RIFE weight caching.
- Camera-change warnings and material-preview connections, plus docstring backtick and math
  escaping errors.

[0.1.0-alpha2] - 2025-11-05
---------------------------

Added
^^^^^

- ``post-install`` task that installs VisionSIM editable into Blender's own Python, so the
  Blender side and the client share versions.
- Read the Docs build configuration, a ``vsim`` alias, and dashed CLI names.
- Docstrings for the ``simulate`` module, typing stubs for dynamic attributes, and passive SPC
  documentation.
- Package metadata, build system, and ``CITATION.cff``.

Changed
^^^^^^^

- Moved render-job and config code out of the CLI and scripts into modules so the Blender
  client pool imports cleanly.
- Reworked the render and ``mkdatasets`` CLIs, replaced ``tqdm`` with ``rich``, raised the
  default log level to ``INFO``, and surfaced the EXR codec as an option.
- Extended Blender compatibility down to 2.9, assigned the port when spawning a
  ``BlenderServer``, and added a connection timeout.
- Autoscaling now validates that a CUDA device exists, and tdigest was replaced with a faster
  digest.

Fixed
^^^^^

- Import errors in the Blender client pool, a missing render import, and normals are now
  enforced as RGB channels in EXR files.
- CI and editable-install fixes.

[0.1.0-alpha] - 2025-04-29
--------------------------

Added
^^^^^

- Blender client/server API with a render pool, so renders run across several Blender processes.
- Ground-truth outputs for depth, normals, segmentation, and forward/backward optical flow,
  with debug visualizations.
- Sensor simulation for IMUs and a v2e-based event camera.
- RIFE interpolation with automatic weight download, spline-based pose interpolation with
  derivatives, and an extended-interpolation example.
- ``Dataset`` abstraction with schema validation, ``transforms.json`` metadata, and
  numpy writers.
- Emulation and colorization CLIs, type annotations throughout, and Sphinx docs with a
  quickstart and examples.

Changed
^^^^^^^

- Renamed the project from ``spsim`` to ``visionsim`` and moved to a modern ``pyproject`` layout.
- Migrated the CLI to Tyro.
- Standardized depth to a single channel and stopped tonemapping ground-truth outputs.

Fixed
^^^^^

- Tonemapping and sRGB conversion, and segmentation colorization.
- Removed the dependency on ``spsim.io``.
