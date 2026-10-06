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


We currently support **Python 3.9+**. Users still on Python 3.8 or older are
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

    $ visionsim blender.render-animation scene.blend output/ \
        --config.executable=/opt/blender-4.2/blender

When Blender comes from a package manager, the value can be the full invocation
command. For Flatpak::

    $ visionsim blender.render-animation scene.blend output/ \
        --config.executable="flatpak run --die-with-parent org.blender.Blender"

``post-install`` takes the same path via ``--executable``::

    $ visionsim post-install --executable=/opt/blender-4.2/blender

Each Blender version ships its own Python interpreter and site-packages, so
dependencies must be installed into every version you render with. Run
``post-install`` once per custom executable.

In Python, set the ``executable`` field of
:class:`RenderConfig <visionsim.simulate.config.RenderConfig>`, or pass
``executable`` to :class:`BlenderServer <visionsim.simulate.blender.BlenderServer>`
and :class:`BlenderClient <visionsim.simulate.blender.BlenderClient>`.

|

Autocompletion
--------------

The auto-complete functionality is provided by `Tyro <https://brentyi.github.io/tyro/tab_completion/>`_, and can be activated per terminal as follows.

|

Bash Support
^^^^^^^^^^^^

First, find and make directory for local completions::

    completion_dir=${BASH_COMPLETION_USER_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/bash-completion}/completions/
    mkdir -p $completion_dir

Next, write completion script::

    visionsim --tyro-write-completion bash ${completion_dir}/visionsim

|

ZSH Support
^^^^^^^^^^^

First, make directory for local completions::

    mkdir -p ~/.zfunc

Next, write completion script::

    visionsim --tyro-write-completion zsh ~/.zfunc/_visionsim

Finally, add the following lines to `.zshrc` file to add `.zfunc` to the function search path::

    fpath+=~/.zfunc
    autoload -Uz compinit && compinit
