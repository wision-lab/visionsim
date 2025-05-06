Installation
============

First, you'll need a few dependencies:

* `Blender <https://www.blender.org/download/>`_ >= 3.3.1, to render new views. 
* `FFmpeg <https://ffmpeg.org/download.html>`_, for visualizations. 


Make sure Blender and ffmpeg are on your PATH.

You can install VisionSIM like so:: 
    
    $ pip install visionsim 


The first time you use the renderer, it may ask you to install additional packages into blender's runtime. 

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
