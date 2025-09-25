Installation
============

First, you'll need:

* `Blender <https://www.blender.org/download/>`_ >= 3.3.1, to render new views. 
* `FFmpeg <https://ffmpeg.org/download.html>`_, for visualizations. 


Make sure Blender and ffmpeg are on your PATH.

Then you can **install the latest stable release** via `pip <https://pip.pypa.io>`_::
    
    $ pip install visionsim


Finally, to install additional dependencies into Blender's runtime, you can run the following::

    $ visionsim post-install


We currently support **Python 3.9+**. Users still on Python 3.8 or older are
urged to upgrade.

|

Autocompletion
--------------

The auto-complete functionality is provided by `Tyro <https://brentyi.github.io/tyro/tab_completion/>`_, and can be activated per terminal as follows.

|

Bash Support
^^^^^^^^^^^^

First, find and make directory for local completions::

    $ completion_dir=${BASH_COMPLETION_USER_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/bash-completion}/completions/
    $ mkdir -p $completion_dir

Next, write completion script::

    $ visionsim --tyro-write-completion bash ${completion_dir}/visionsim

|

ZSH Support
^^^^^^^^^^^

First, make directory for local completions::

$ mkdir -p ~/.zfunc

Next, write completion script::

$ visionsim --tyro-write-completion zsh ~/.zfunc/_visionsim

Finally, add the following lines to `.zshrc` file to add `.zfunc` to the function search path::

    $ fpath+=~/.zfunc
    $ autoload -Uz compinit && compinit
