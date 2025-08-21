Development
===========

Editable Install
----------------

To install visionsim locally in an editable state with all required development dependencies, clone the repository, navigate to it and run::
    
    $ pip install -e ".[dev]"

Similarly, to install visionsim in an editable manner within Blender's runtime, you can do the following::

    $ visionsim post-install --editable

| 

Running Tests
-------------

We use pytest for testing, all tests can be ran directly using the pytest CLI from the project's root, or equivalently using `inv test`. When running the tests, you can optionally pass in the name of a specific test file/test function and path to a install of Blender to test with, like so:: 

    $ pytest tests/test_simulate.py --executable=<path-to-blender>

To ensure that there's no conflicts due to different versions of the libraries between the server/client sides, a editable ``post-install`` task is run when starting the tests.

The ``-rP`` option is also helpful for seeing any stdout messages that are otherwise hidden. 

| 

Building the Documentation
--------------------------

In the project root, with visionsim installed with the dev dependencies, run::

    $ inv clean build-docs --preview

|

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
