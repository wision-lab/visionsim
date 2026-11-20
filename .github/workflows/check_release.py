"""Check that a release tag, the package version and the changelog agree."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parent.parent.parent


def parse_tag(tag: str) -> Version:
    """Parse a ``v``-prefixed release tag into a PEP 440 version.

    Args:
        tag (str): Git tag, for instance ``v0.2.0``.

    Returns:
        Version: Parsed version.

    Raises:
        ValueError: raised when the tag is not a ``v``-prefixed PEP 440 version.
    """
    match = re.fullmatch(r"v(.+)", tag)
    if match is None:
        raise ValueError(f"Tag {tag!r} does not start with 'v'.")
    return Version(match.group(1))


def changelog_versions(changelog: Path) -> set[Version]:
    """Collect the versions that have a release heading in the changelog.

    Args:
        changelog (Path): Path to ``changelog.rst``.

    Returns:
        set[Version]: Versions with a heading; non-version headings such as ``[Unreleased]`` are skipped.
    """
    headings = re.findall(r"^\[([^\]]+)\]", changelog.read_text(encoding="utf-8"), re.MULTILINE)
    versions = set()
    for heading in headings:
        try:
            versions.add(Version(heading))
        except InvalidVersion:
            continue
    return versions


def check(tag: str, root: Path = ROOT) -> Version:
    """Verify the tag matches ``__version__`` and has a changelog heading.

    Args:
        tag (str): Git tag being released, for instance ``v0.2.0``.
        root (Path, optional): Repository root. Defaults to the checkout containing this script.

    Returns:
        Version: The checked version.

    Raises:
        ValueError: raised when the tag, version or changelog disagree.
    """
    expected = parse_tag(tag)

    source = (root / "visionsim" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__ = "([^"]+)"', source)
    if match is None:
        raise ValueError("Could not find __version__ in visionsim/__init__.py.")
    version = Version(match.group(1))
    if version != expected:
        raise ValueError(f"visionsim.__version__ is {version}, but tag {tag} implies {expected}.")

    if expected not in changelog_versions(root / "docs/source/changelog.rst"):
        raise ValueError(f"No changelog heading for {expected}; add one before tagging.")
    return version


def main() -> None:
    """Run the check, emitting a GitHub annotation and non-zero exit on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Release tag, for instance v0.2.0.")
    args = parser.parse_args()
    try:
        version = check(args.tag)
    except ValueError as error:
        sys.exit(f"::error::{error}")
    print(f"Tag {args.tag} matches version {version} and the changelog.")


if __name__ == "__main__":
    main()
