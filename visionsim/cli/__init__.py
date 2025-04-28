import inspect
import sys

import tyro

from . import blender, dataset, emulate, ffmpeg, interpolate, transforms


def main():
    cli_dict = {}
    cli_modules = [blender, dataset, emulate, ffmpeg, interpolate, transforms]

    for module in cli_modules:
        current_module = sys.modules[module.__name__]
        module_name = current_module.__name__.split(".")[-1]
        cli_dict.update(
            {
                f"{module_name}.{func_name}": func
                for func_name, func in inspect.getmembers(current_module, inspect.isfunction)
                if func.__module__ == module.__name__ and not func_name.startswith("_")
            }
        )

    tyro.extras.subcommand_cli_from_dict(cli_dict)
