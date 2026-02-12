import inspect
import re
import warnings

import peewee
import pytest
from docstring_parser import parse_from_object

from visionsim.cli import _cli_modules
from visionsim.dataset import dataset, models
from visionsim.interpolate import pose
from visionsim.simulate import blender, install, job, schema


def get_public_members(obj, module=None):
    """Recursively get all public classes, methods thereof and functions"""
    if not module:
        module = obj.__name__

    for _, child_obj in inspect.getmembers(obj, lambda c: getattr(c, "__module__", None) == module):
        name = getattr(child_obj, "__name__", getattr(child_obj, "attrname", None))

        if name.startswith("_") and not name.startswith("__"):
            continue
        elif inspect.isclass(child_obj):
            yield child_obj
            yield from get_public_members(child_obj, module)
        elif inspect.isfunction(child_obj) or inspect.ismethod(child_obj):
            yield child_obj


@pytest.mark.parametrize(
    "obj",
    [
        pytest.param(m, id=".".join([mod.__name__, m.__qualname__]))
        for mod in _cli_modules + [blender, install, job, schema, dataset, models, pose]
        for m in get_public_members(mod)
    ],
)
def test_docstrings(obj):
    if inspect.isclass(obj) and issubclass(obj, peewee.DoesNotExist):
        # Peewee dynamically creates error classes per-table
        return

    docs = parse_from_object(obj)

    assert obj.__doc__, "Missing docstring."

    if matches := re.findall(r"(?<!:|`)`(?!`)[\w \.]+`(?!_|`)", obj.__doc__):
        warnings.warn(
            f"Found improper backtick usage in {obj.__qualname__}'s docstring. "
            f"Expected either sphinx-style links or double-ticks for literals, got {', '.join(matches)}"
        )

    if inspect.isclass(obj):
        # We expect the __init__ to have the arg's docstring, so we early out.
        return

    if not sum(1 for line in obj.__doc__.splitlines() if not line.strip()):
        # Skip further validation for short descriptions
        return

    documented_params = set(param.arg_name.strip("*") for param in docs.params if param.description)
    assert all("_description_" not in param.description for param in docs.params)

    sig = inspect.signature(obj)
    all_params = set(sig.parameters.keys())

    for item in ("self", "cls"):
        if item in all_params:
            all_params.remove(item)

    if sig.return_annotation:
        # Methods decorated with @contextmanager will yield a value, so the return type is of that value, not
        # the Iterator[T] that the func is annotated with. Hacky, but works well enough.
        if "@contextmanager\n" in inspect.getsource(obj):
            return_annotation = sig.return_annotation.removeprefix("Iterator[").removesuffix("]")
        else:
            return_annotation = sig.return_annotation
        doc_returns = docs.returns or docs.yields

        if return_annotation != "None" and return_annotation != "Self":
            assert doc_returns, "Missing docstring return info."

            # # Does this issue still occur??
            # if doc_returns.type_name is None:
            #     # Return types using | cause issues, so here we just match the type string verbatim instead.
            #     # See: https://github.com/rr-/docstring_parser/issues/81
            #     print(obj.__doc__[obj.__doc__.index("Returns:"):])
            #     assert return_annotation in obj.__doc__[obj.__doc__.index("Returns:"):]
            # else:
            assert return_annotation == doc_returns.type_name, "Missing or incorrect docstring return type."

    assert documented_params == all_params, "Not all arguments are in docstring."
