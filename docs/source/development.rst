Development
===========

Editable Install
----------------

To install visionsim locally in an editable state with all required development dependencies, clone the repository, navigate to it and run::
    
    $ pip install -e ".[dev]"


Dev tools
---------

We're using `invoke <https://docs.pyinvoke.org/en/stable/>`_ to manage common development and housekeeping tasks.

Make sure you have invoke installed then you can run any of the following `tasks` from the project root:

.. command-output:: invoke --list

It's also recommended using the pre-commit hook that will lint/test/clean 
the code before every commit. For this make sure that `invoke` and `pre-commit` are 
installed (via pip) and then install the pre-hooks with::

    $ pre-commit install

See `pre-commit <https://pre-commit.com/#intro>`_ for more.
