"""Sphinx helpers for the documentation.

Inline Tree Clipper exports so node graphs render without a fetch, and keep a percentage
``:width:`` on a video from being dropped on the way to the browser.
"""

import base64
import gzip
from pathlib import Path
from typing import ClassVar

from docutils import nodes  # type: ignore[import-untyped]
from docutils.parsers.rst import directives  # type: ignore[import-untyped]
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util.docutils import SphinxDirective
from sphinxcontrib.video import video_node

MAGIC = "TreeClipper::"


class NodeGraphDirective(SphinxDirective):
    """Render a node tree exported by the ``figures.*node-tree*`` tasks."""

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = False
    option_spec: ClassVar[dict] = {"height": directives.length_or_percentage_or_unitless}

    def run(self):
        # Relative to the document, the same way `.. image::` resolves.
        document = Path(self.state.document.current_source)
        source = (document.parent / self.arguments[0]).resolve()
        if not source.is_file():
            raise self.error(
                f"node graph export not found: {self.arguments[0]} (looked in {source};"
                " run `inv figures.sections.blender.groundtruths`)"
            )

        # Without this the page is only rebuilt when its own source changes, so a
        # regenerated export would leave the embedded graph as it was.
        self.env.note_dependency(source)

        # mtime=0 keeps the compressed payload, and therefore the page, byte-for-byte
        # reproducible across builds.
        payload = MAGIC + base64.b64encode(gzip.compress(source.read_bytes(), 9, mtime=0)).decode()

        options = f' style="height: {self.options["height"]}"' if "height" in self.options else ""
        html = f'<div class="node-graph" data-payload="{payload}"{options}></div>'
        return [nodes.raw("", html, format="html")]


class PercentVideoWidth(SphinxPostTransform):
    """Let a percentage ``:width:`` on a video reach the browser.

    ``sphinxcontrib-video`` writes its ``width`` option out as an attribute, and a
    ``<video>`` width only accepts a pixel count, so a browser drops a percentage and
    the clip keeps its natural size however the page asked for it. The extension writes
    ``figwidth`` as a style on the container instead, which a percentage survives, but
    its directive clears that option unless the video also has a caption. After the
    directive has run, move the percentage over to the option that works.
    """

    default_priority = 100

    def run(self, **kwargs):
        """Give every video with a percentage width the equivalent ``figwidth``."""
        for node in self.document.findall(video_node):
            width = node.get("width", "")
            if isinstance(width, str) and width.endswith("%"):
                node["figwidth"] = width
                node["width"] = ""


def setup(app):
    app.add_directive("node-graph", NodeGraphDirective)
    app.add_post_transform(PercentVideoWidth)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
