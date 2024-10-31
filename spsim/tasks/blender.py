import sys
from pathlib import Path

from invoke import task

from spsim.cli import modify_signature
from spsim.render import parser_config
from spsim.tasks.common import _run

# Dynamically populate arguments of the `render` task
conf = parser_config()
render_help = {arg["name"].lstrip("-").replace("-", "_"): arg["help"] for arg in conf["arguments"]}
render_help["blender_path"] = "specify which blender to use if there exists multiple, default: version on system $PATH"
render_help["autoexec"] = "if true, enable the execution of bundled code. default: False"
render_args = [arg["name"].lstrip("-").replace("-", "_") for arg in conf["arguments"] if "default" not in arg]
render_kwargs = {
    arg["name"].lstrip("-").replace("-", "_"): arg["default"] for arg in conf["arguments"] if "default" in arg
}
render_kwargs.pop("blend_file")


def render(c, blend_file, *args, blender_path=None, autoexec=False, **kwargs):
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
    cmd = f"{blender_path or 'blender'} --background {autoexec} --python {path} -- {' '.join(sys.argv[3:])} --blend-file={blend_file}"
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
