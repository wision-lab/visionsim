from __future__ import annotations

import copy
import json

from jsonschema import ValidationError, validate
from typing_extensions import Any

# Schema for nerfstudio-style transforms.json file, which uses a folder of imgs.
# Note: The command `visionsim blender.render-animation` will output a compatible transform file,
#   but with added information such as:
#   "angle": {"type": "number"},  #  FoV in x axis in radians, same as `camera_angle_x` or `angle_x`.
#   "shift_x": {"type": "number"},  #  Offset from center, in x axis, of sensor where optical axis lies
#   "shift_y": {"type": "number"},  #  Offset from center, in y axis, of sensor where optical axis lies
#   "type": {"type": "string", "pattern": "PERSP"},  # Type of camera: PERSP, ORTHO or PANO. Only PERSP is supported.
IMG_FRAME_SCHEMA = {
    "type": "object",
    "properties": {
        "transform_matrix": {  # Ensure the transform matrix is a 4x4
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "number"},
            },
        },
        "file_path": {"type": "string"},  # Path to color image, relative to parent dir
        "depth_file_path": {"type": "string"},  # Path to depth image, relative to parent dir
    },
    "required": ["transform_matrix", "file_path"],
}

IMG_SCHEMA = {
    "type": "object",
    "properties": {
        "fl_x": {"type": "number"},  # Focal length in X
        "fl_y": {"type": "number"},  # Focal length in Y
        "cx": {"type": "number"},  # Center of optical axis in pixel coordinates in X
        "cy": {"type": "number"},  # Center of optical axis in pixel coordinates in Y
        "w": {"type": "number"},  # Sensor width in pixels
        "h": {"type": "number"},  # Sensor height in pixels
        "c": {"type": "number"},  # Number of output channels (i.e: RGBA = 4)
        "frames": {"type": "array", "items": IMG_FRAME_SCHEMA},
    },
    "required": ["fl_x", "fl_y", "cx", "cy", "h", "w", "c", "frames"],
}


# Schema for NPY-style transforms file.
# The main difference is that "*file_path", which was per-frame before, has been
# extracted and is now the top level path to the npy file containing all frames.
# Indexing into this npy file is done based on the frame's index in the `frames` list.
NPY_FRAME_SCHEMA: dict[str, Any] = copy.deepcopy(IMG_FRAME_SCHEMA)
NPY_FRAME_SCHEMA["properties"]["file_path"] = False  # Cannot be present!
NPY_FRAME_SCHEMA["properties"]["depth_file_path"] = False
NPY_FRAME_SCHEMA["required"] = ["transform_matrix"]

NPY_SCHEMA: dict[str, Any] = copy.deepcopy(IMG_SCHEMA)
NPY_SCHEMA["properties"]["frames"] = {"type": "array", "items": NPY_FRAME_SCHEMA}
NPY_SCHEMA["properties"]["file_path"] = {"type": "string"}  # Path to npy file containing color images
NPY_SCHEMA["properties"]["depth_file_path"] = {"type": "string"}  # Path to npy file containing depth images
NPY_SCHEMA["properties"]["bitpack"] = {"type": "boolean"}  # Whether npy file is bitpacked
NPY_SCHEMA["properties"]["bitpack_dim"] = {"type": ["number", "null"]}  # Bitpacked axis
NPY_SCHEMA["required"].append("file_path")


def read_and_validate(*, path, schema):
    """Load a json from a file and check that it complies with the provided schema."""
    with open(path, "r") as f:
        transforms = json.load(f)

    try:
        validate(schema=schema, instance=transforms)
    except ValidationError as e:
        raise ValidationError(
            f"Format not understood: Supplied transforms file ({path}) does not match expected schema."
        ) from e

    return transforms


def validate_and_write(*, path, schema, transforms):
    """Dump json to a file and check that it complies with the provided schema."""
    try:
        validate(schema=schema, instance=transforms)
    except ValidationError as e:
        raise ValidationError(
            f"Format not understood: Supplied transforms file ({path}) does not match expected schema."
        ) from e

    with (path).open("w") as f:
        json.dump(transforms, f, indent=2)
