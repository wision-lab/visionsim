Configuration Files
===================

Some commands take a large group of related options, exposed as ``--config.*``. Those
commands also accept ``--config.load PATH`` and ``--config.dump PATH``:

- ``--config.load`` reads a YAML or JSON file and uses its values as defaults for the
  group.
- ``--config.dump`` writes the effective group to a YAML file before the command runs.

This is useful for reproducible jobs, for sharing a base configuration, and for
invocations too long to repeat. The command line stays authoritative: any ``--config.*``
flag overrides the file.

Where it applies
----------------

The options belong to a command's configuration argument, not to the CLI globally. For
example, ``visionsim blender.render-animation`` and ``visionsim blender.render-playblast``
take a render configuration, so ``--config.load`` and ``--config.dump`` accept a
``RenderConfig`` file. Commands without such an argument do not accept the options.

The file format
---------------

The file holds the group's fields directly, without a ``config:`` wrapper. Nested options
are mappings and option names use underscores, mirroring the field names behind the
flags:

.. code-block:: yaml

    jobs: 4
    include_frames: true
    include_depths: true
    frames:
      file_format: PNG
      bit_depth: 16
    flows:
      direction: both

.. code-block:: bash

    visionsim blender.render-animation scene.blend output/ \
        --config.load render.yaml --config.jobs 8

The command above renders with eight jobs; every other value comes from ``render.yaml``.
Values are parsed exactly like their command-line counterparts, so types, choices and
formats such as ``max_job_vram: 4GB`` behave the same. YAML and JSON are both accepted.

Dumping the effective configuration
-----------------------------------

``--config.dump PATH`` writes the merged group to ``PATH`` before the command starts,
which makes a working invocation easy to reuse as a starting point. YAML is used unless
``PATH`` ends in ``.json``. ``load`` and ``dump`` are command-line options and cannot be
set in the file itself. The dump reflects the parsed input; derived values such as
autoscaled job counts are applied later by the job. Parent directories are created.

.. tip::
    Set ``VSIM_LOG_LEVEL=DEBUG`` to print the resolved configuration before rendering.
    The log shows the merged file and command-line values, so you can check a file
    without writing a dump.

.. note::
    ``include_all`` enables every output and takes precedence over the individual
    ``include_*`` options, whether they come from a file or the command line. Use the
    explicit flags when only a subset of the outputs is wanted.
