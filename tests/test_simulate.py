import inspect
import itertools
import re
import warnings

import numpy as np
import OpenEXR
import pytest
from docstring_parser import parse_from_object

from visionsim.dataset import IMG_SCHEMA, read_and_validate
from visionsim.simulate import blender


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


@pytest.mark.parametrize("obj", [pytest.param(m, id=m.__qualname__) for m in get_public_members(blender)])
def test_docstrings(obj):
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

        if return_annotation != "None" and return_annotation != "Self":
            assert docs.returns, "Missing docstring return info."
            assert return_annotation == docs.returns.type_name, "Missing or incorrect docstring return type."

    assert documented_params == all_params, "Not all arguments are in docstring."


def test_render_layout(cube_dataset):
    assert (cube_dataset / "transforms.json").exists()

    for subdir in ["frames", "depths", "normals", "flows", "segmentations"]:
        subdir = cube_dataset / subdir
        assert subdir.exists()
        assert len(list(subdir.glob("*.png"))) == 5
        assert len(list(subdir.glob("*.exr"))) in (0, 5)


@pytest.mark.parametrize(
    "subdir, channels", [("depths", ["V"]), ("normals", ["RGB"]), ("flows", ["RGBA"]), ("segmentations", ["V"])]
)
def test_groundtruth_exrs(cube_dataset, subdir, channels):
    for file in cube_dataset.glob(f"{subdir}/*.exr"):
        with OpenEXR.File(str(file)) as f:
            # Before v4.3.0 exr's couldn't be single channel, they were saved as
            # RGB with duplicated channels. Yet, in 4.2 and 4.1, they still get saved as "V"
            if channels == ["V"] and "V" not in f.channels():
                assert "RGB" in f.channels()
                data = f.channels()["RGB"].pixels.transpose(2, 0, 1)
                assert all(np.allclose(a, b) for a, b in itertools.pairwise(data))
                channels = ["RGB"]
            else:
                assert list(f.channels().keys()) == channels

            for channel in channels:
                assert np.issubdtype(f.channels()[channel].pixels.dtype, np.floating)


def test_transforms_schema(cube_dataset):
    read_and_validate(path=cube_dataset / "transforms.json", schema=IMG_SCHEMA)
