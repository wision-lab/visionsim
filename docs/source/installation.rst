Installation
============

First, you'll need:

* `Blender <https://www.blender.org/download/>`_ >= 3.6, to render new views. 
* `FFmpeg <https://ffmpeg.org/download.html>`_, for visualizations. 


Make sure Blender and ffmpeg are on your PATH.

Then you can **install the latest stable release** via `pip <https://pip.pypa.io>`_::
    
    pip install visionsim


Finally, to install additional dependencies into Blender's runtime, you can run the following::

    visionsim post-install


We currently support **Python 3.10+**. Users still on Python 3.9 or older are
urged to upgrade.

|

Using a Custom Blender Version
------------------------------

By default, visionsim looks for a ``blender`` executable on your ``PATH``. If
you have several Blender versions installed, or Blender isn't on your ``PATH``
(for instance when installed via Flatpak or as a standalone tarball), point
visionsim at a specific install instead.

Every command that spawns Blender takes a ``--config.executable`` option with
the path to the Blender binary to use::

    visionsim blender.render-animation scene.blend output/ \
        --config.executable=/opt/blender-4.2/blender

When Blender comes from a package manager, the value can be the full invocation
command. For Flatpak::

    visionsim blender.render-animation scene.blend output/ \
        --config.executable="flatpak run --die-with-parent org.blender.Blender"

``post-install`` takes the same path via ``--executable``::

    visionsim post-install --executable=/opt/blender-4.2/blender

Each Blender version ships its own Python interpreter and site-packages, so
dependencies must be installed into every version you render with. Run
``post-install`` once per custom executable. Blender's torch comes from PyPI by
default; on a machine without CUDA, or one that needs a specific CUDA build, pass the
matching wheel index::

    visionsim post-install --executable=/opt/blender-4.2/blender \
        --torch-index-url=https://download.pytorch.org/whl/cpu

In Python, set the ``executable`` field of
:class:`RenderConfig <visionsim.simulate.config.RenderConfig>`, or pass
``executable`` to :class:`BlenderServer <visionsim.simulate.blender.BlenderServer>`
and :class:`BlenderClient <visionsim.simulate.blender.BlenderClient>`.

|

Autocompletion
--------------

Tab completion is provided by `Tyro <https://brentyi.github.io/tyro/tab_completion/>`_ and
can be installed for your current shell with::

    visionsim completion

This writes completion scripts for ``visionsim`` and ``vsim`` into the standard user
completion directory of the shell named by ``$SHELL``. Supported shells are bash, zsh,
and fish; pass ``--shell`` to choose one explicitly, or ``--path`` to write somewhere
else. Re-run the command after upgrading VisionSIM, since subcommands and options change
between releases.

Bash and fish load the standard directory automatically. Zsh reads completion functions
from ``fpath``, so add ``~/.zfunc`` in ``.zshrc``::

    fpath+=~/.zfunc
    autoload -Uz compinit && compinit

Remove the scripts again with::

    visionsim completion --mode uninstall
