Autocompletion
==============

The auto-complete functionality is provided by `pyinvoke <https://docs.pyinvoke.org/en/stable/invoke.html#shell-tab-completion>`_, and can be activated per terminal like so::

$ source <(visionsim --print-completion-script bash)


or by creating a file that can be sourced from your `~/.bashrc`::

    $ visionsim --print-completion-script bash > ~/.visionsim-completion.sh

    # Place in ~/.bashrc:
    $ source ~/.visionsim-completion.sh

The same can be done in other shells such as `zsh`, `fish`. 
