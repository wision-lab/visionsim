"""Documentation sources are linted here, not by Sphinx.

Sphinx only reports backtick typos or over-wide code as warnings, which the docs
build happily ignores, so the rules that back the documentation's conventions
live in this module. Every rule parses ``docs/source`` with docutils -- the same
parser Sphinx uses -- and asserts on the resulting doctree, so reST constructs
are understood the way the reader will see them rather than matched textually.
"""

from pathlib import Path

import pytest
from docutils import nodes
from docutils.core import publish_doctree

DOCS_DIR = Path(__file__).parent.parent / "docs" / "source"

# A code block wider than the rendered code column forces the docs CSS to wrap,
# or a scrollbar in a theme without it. The column is about 90 characters wide at
# Furo's default size; the slack covers the extra characters a wider window holds.
MAX_CODE_WIDTH = 100

# Directives that docutils does not know (they come from Sphinx extensions) end up
# as literal blocks showing their own markup. They are not code blocks, so they
# are skipped instead of being mistaken for one that needs wrapping.
NON_CODE_DIRECTIVES = (
    ".. literalinclude::",
    ".. program-output::",
    ".. seealso::",
)


def doctree(path):
    return publish_doctree(path.read_text(), settings_overrides={"report_level": 5, "halt_level": 5})


def code_blocks(path):
    """Yield ``(source line, lines)`` for every literal block of actual code."""
    for block in doctree(path).findall(nodes.literal_block):
        lines = block.astext().splitlines()
        if any(line.startswith(NON_CODE_DIRECTIVES) for line in lines):
            continue

        yield block.line, [line.rstrip() for line in lines if line.strip()]


def docs_files():
    return sorted(DOCS_DIR.rglob("*.rst"))


def test_docs_exist():
    assert docs_files(), f"No documentation sources found under {DOCS_DIR}"


@pytest.mark.parametrize("path", docs_files(), ids=lambda p: str(p.relative_to(DOCS_DIR)))
def test_no_prompt_in_codeblocks(path):
    offenders = [line for _, lines in code_blocks(path) for line in lines if line.lstrip().startswith("$ ")]

    assert not offenders, (
        f"{path.relative_to(DOCS_DIR)}: drop the $ prompt from literal blocks, it makes them un-copyable: {offenders}"
    )


@pytest.mark.parametrize("path", docs_files(), ids=lambda p: str(p.relative_to(DOCS_DIR)))
def test_codeblock_width(path):
    offenders = [
        f"{len(line)} chars: {line.strip()}"
        for _, lines in code_blocks(path)
        for line in lines
        if len(line) > MAX_CODE_WIDTH
    ]

    assert not offenders, (
        f"{path.relative_to(DOCS_DIR)}: literal blocks must fit in {MAX_CODE_WIDTH} rendered characters, "
        f"or the reader has to scroll horizontally. Wrap these lines: {offenders}"
    )
