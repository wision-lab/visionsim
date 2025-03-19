Autocompletion
==============

The auto-complete functionality is provided by `pyinvoke <https://docs.pyinvoke.org/en/stable/invoke.html#shell-tab-completion>`_, and can be activated per terminal like so::

$ source <(spsim --print-completion-script bash)


or by creating a file that can be sourced from your `~/.bashrc`::

    $ spsim --print-completion-script bash > ~/.spsim-completion.sh

    # Place in ~/.bashrc:
    $ source ~/.spsim-completion.sh

The same can be done in other shells such as `zsh`, `fish`. 
