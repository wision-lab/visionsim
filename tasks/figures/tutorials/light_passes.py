"""Light passes figures: ``docs/source/tutorials/light-passes.rst``."""

from __future__ import annotations

from .._page import Node, _command_recipe, cached, encode_webp, page_task, static


def _stills(executable: str | None, force: bool = False) -> None:
    """Write the page's stills: the combined frame and each tonemapped pass.

    Args:
        executable: Ignored; this step needs no Blender.
        force: Ignored; the copies and the tonemap overwrite unconditionally.
    """
    _command_recipe(
        "light-passes-stills",
        # Give each linear EXR pass a name of its own, since every pass shares the
        # frame's filename and tonemap-frames flattens output to <output-dir>/<stem>.<ext>,
        # which would collide all six onto one file.
        "mkdir -p light-passes-named && for p in diffuse/direct diffuse/indirect diffuse/color"
        " specular/direct specular/indirect specular/color; do"
        " cp light-passes/$p/0000/090.exr light-passes-named/$(echo $p | tr / -).exr; done",
        # One call tone-maps every pass; the names chosen above keep them distinct.
        "visionsim transforms.tonemap-frames --input-dir light-passes-named"
        " --output-dir ../docs/source/_static/blender/light-passes --pattern '*.exr' --ext .webp",
    )(executable, force)

    # The combined frame is already an sRGB PNG.
    encode_webp(cached("light-passes", "frames", "0000", "090.png"), static("blender/light-passes/combined.webp"))


NODES = (
    Node(
        name="light-passes-render",
        files=(cached("light-passes", "frames"),),
        recipe=_command_recipe(
            "light-passes-render",
            "visionsim blender.render-frame country-kitchen.blend light-passes/ --frame 90"
            " --config.include-frames --config.include-diffuse-pass --config.include-specular-pass"
            "{executable}{force}",
        ),
    ),
    Node(
        name="light-passes-stills",
        files=(
            static("blender/light-passes/combined.webp"),
            static("blender/light-passes/diffuse-direct.webp"),
        ),
        is_figure=True,
        requires=("light-passes-render",),
        recipe=_stills,
    ),
)

build = page_task(NODES, "light_passes", "Regenerate the light pass example figures.")
