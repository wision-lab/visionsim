"""Rolling-ball figures: ``docs/source/sections/blender/thermal.rst``."""

from __future__ import annotations

from shlex import quote

from ..._page import ROOT, Node, _command_recipe, cached, dataset_check, page_task, static, video_recipe

SCENE = ROOT / "assets" / "blender" / "rollingball.blend"
RENDERS = cached("rolling-ball", "renders")
FRAMES = 180


_render = _command_recipe(
    "rolling-ball-render",
    f"visionsim blender.render-animation {quote(str(SCENE))} rolling-ball/renders"
    " --config.include-thermal --config.max-samples=32"
    " --config.adaptive-threshold=0.03 --config.use-motion-blur=False"
    " --config.thermal.mode=TRANSIENT --config.thermal.device=cpu"
    " --config.thermal.render-domain=VERTEX"
    " --config.thermal.substeps-per-frame=2 --config.thermal.laplacian-update-every-steps=2"
    " --config.thermal.irradiance-scale=300 --config.thermal.bake-samples=16"
    " --config.thermal.irradiance-texture-size=256 --config.thermal.cache-root=rolling-ball/cache"
    " --config.log-dir=rolling-ball/logs{executable}{force}",
)


def _complete() -> bool:
    return all(
        dataset_check(RENDERS / name, FRAMES)()
        for name in ("frames", "temperature", "thermal_radiance", "previews/temperature")
    )


NODES = (
    Node(name="rolling-ball-render", files=(RENDERS,), recipe=_render, check=_complete),
    *(
        Node(
            name=f"rolling-ball-{name}-preview",
            files=(static(f"thermal-transient/rolling-ball-{name}-20fps.mp4"),),
            is_figure=True,
            requires=("rolling-ball-render",),
            recipe=video_recipe(
                f"rolling-ball/renders/{source}/*/*.png",
                1,
                f"thermal-transient/rolling-ball-{name}-20fps",
                size=(320, 256),
                fps=20,
            ),
        )
        for name, source in (("rgb", "frames"), ("temperature", "previews/temperature"))
    ),
)

build = page_task(NODES, "thermal", "Regenerate the rolling-ball thermal demonstration.")
