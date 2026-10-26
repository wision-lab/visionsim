Configuration Files
===================

Render commands take many ``--config.*`` options. Instead of repeating them on every
invocation, write them to a YAML or JSON file and load it with ``--config.load``:

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

The file mirrors the ``--config.*`` options without their ``config.`` prefix, with
hyphens written as underscores (``--config.include-depths`` becomes ``include_depths``);
nested options are written as mappings. Flags passed on the command line override the
file, so the command above uses eight jobs while keeping the other options.

Passing ``--config.dump PATH`` writes the merged configuration to ``PATH`` before the
command starts, which makes a working invocation easy to reuse as a starting point.
``load`` and ``dump`` are command-line options and cannot be set in the file itself.
YAML and JSON are both accepted.

.. note::
    ``include_all`` enables every output and takes precedence over the individual
    ``include_*`` options, whether they come from a file or the command line. Use the
    explicit flags when only a subset of the outputs is wanted.
