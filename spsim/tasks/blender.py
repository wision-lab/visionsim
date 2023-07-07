import sys
from pathlib import Path

from invoke import task

from spsim.cli import modify_signature
from spsim.render import parser_config
from spsim.tasks.common import _run

# Dynamically populate arguments of the `render` task
conf = parser_config()
render_help = {arg["name"].lstrip("-").replace("-", "_"): arg["help"] for arg in conf["arguments"]}
render_help["autoexec"] = "if true, enable the execution of bundled code. default: False"
render_args = [arg["name"].lstrip("-").replace("-", "_") for arg in conf["arguments"] if "default" not in arg]
render_kwargs = {
    arg["name"].lstrip("-").replace("-", "_"): arg["default"] for arg in conf["arguments"] if "default" in arg
}
render_kwargs.pop("blend_file")


def render(c, blend_file, *args, autoexec=False, **kwargs):
    """Wrapper that calls the `blender.py` CLI within blender. All arguments
    are first verified here using invoke and then again using argparse."""
    # TODO: Enable parallelization by spinning up different workers (and blender
    #  instances), each focusing on different frames.

    # Runtime checks and gard rails
    if _run(c, "blender --version", hide=True).failed:
        raise RuntimeError("No blender installation found on path!")
    if not (blend_file := Path(blend_file).resolve()).exists():
        raise FileNotFoundError(f"Blender file {blend_file} not found.")
    if "blender.render" not in sys.argv[1]:
        raise RuntimeError("Task `blender.render` must run first if running multiple tasks simultaneously.")

    # Call `render.py` script through blender's python interpreter
    path = Path(__file__).parent.parent / "render.py"
    autoexec = "--enable-autoexec" if autoexec else "--disable-autoexec"
    cmd = f"blender --background {autoexec} --python {path} -- {' '.join(sys.argv[3:])} --blend-file={blend_file}"
    _run(c, cmd)


render_full = modify_signature(
    *render_args,
    remove_var_args=True,
    remove_var_kwargs=True,
    # TODO: Make the docstring for invoke usage only here and have a
    #   default docstring within argparse in the `parser_conf` dict.
    docstring=conf["parser"]["prog"],
    **render_kwargs,
)(render)
render = task(help=render_help, auto_shortflags=False)(render_full)


@task(
    help={
        "input_dir": "directory containing transform file or path of file",
        "infile": "name of input transform file, default: transforms_blender.json",
        "outfile": "name of output transform file, default: transforms.json",
        "aabb_scale": "scale of axis aligned bounding box, default: 16",
        "sharpness": "if supplied, compute per-image sharpness value, default: False",
        "force": "override existing transform file if in/outfile are equal, default: False",
    }
)
def to_nerf_format(
    _,
    input_dir,
    infile="transforms_blender.json",
    outfile="transforms.json",
    aabb_scale=16,
    sharpness=False,
    force=False,
):
    """Convert transform.json from blender format to nerf-style format

    Flatten "camera" entry, add "w", "h", "fl_x" and "fl_y", and convert "file_paths" to "file_path"
    """
    import json

    import numpy as np

    from spsim.colmaptools import compute_sharpness

    if Path(input_dir).is_file():
        print("Path provided is file, ignoring `infile` argument.")
        infile = Path(input_dir).name
        input_dir = Path(input_dir).parent

    transforms_in_path = (Path(input_dir) / infile).resolve()
    transforms_out_path = (Path(input_dir) / outfile).resolve()

    if not force:
        if transforms_out_path == transforms_in_path:
            raise RuntimeError("Input and output files are the same!")
        if transforms_out_path.exists():
            raise RuntimeError(f"Output file {transforms_out_path} already exists.")

    with transforms_in_path.open("r") as f:
        out = json.load(f)

    # Flatten out camera section
    camera = out.pop("camera")
    camera["cx"] = camera.get("cx", np.array(camera["intrinsics"])[0, 2])
    camera["cy"] = camera.get("cy", np.array(camera["intrinsics"])[1, 2])
    out.update(camera)

    if camera["type"] != "PERSP":
        raise NotImplementedError("Only perspective cameras are supported!")

    out["aabb_scale"] = aabb_scale
    out["w"] = 2 * (camera["cx"] - camera["shift_x"])
    out["h"] = 2 * (camera["cy"] - camera["shift_y"])
    out["fl_x"] = camera["fx"]
    out["fl_y"] = camera["fy"]

    out["frames"] = [
        {
            "file_path": frame["file_paths"][0],
            "sharpness": compute_sharpness(str(Path(input_dir) / frame["file_paths"][0])) if sharpness else None,
            "transform_matrix": frame["transform_matrix"],
        }
        for frame in out["frames"]
    ]

    with transforms_out_path.open("w") as outfile:
        json.dump(out, outfile, indent=2)
