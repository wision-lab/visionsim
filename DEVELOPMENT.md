# Environment

You'll need blender and FFmpeg installed, as well as this repository cloned and installed with the extra dev-dependencies:
```shell
pip install -e ".[dev]"
```

# Dev tools

We're using [invoke](https://docs.pyinvoke.org/en/stable/), and can be activated per terminal like so:
 to manage common development and housekeeping tasks.

Make sure you have invoke installed then you can run any of the following "tasks":

```
$ inv --list
Available tasks:

  clean          Runs all clean sub-tasks
  clean-build    Clean up files from package building
  clean-python   Clean up python file artifacts
  clean-tests    Clean up files from testing
  coverage       Create coverage report
  format         Format code
  lint           Lint code with ruff
  precommit      Run pre-commit hooks
  test           Run tests
```


It's also recommended using the pre-commit hook that will lint/test/clean 
the code before every commit. For this make sure that `invoke` and `pre-commit` are 
installed (via pip) and then install the pre-hooks with: 

```
pre-commit install
```

You can also run the pre-hooks without having to create a commit like so:

```
pre-commit run --all-files
```

See [pre-commit](https://pre-commit.com/#intro) for more.
