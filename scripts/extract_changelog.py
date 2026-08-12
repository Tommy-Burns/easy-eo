#!/usr/bin/env python
"""Extract one version's section from ``CHANGELOG.md``.

The release workflow uses this to turn the changelog entry for the tag being
released into the body of the GitHub Release, so the release notes and the
changelog cannot drift apart. Failing when the section is missing is the point
as much as the extraction is: it stops a tag that nobody wrote a changelog
entry for from becoming a release.

Usage
-----
    python scripts/extract_changelog.py 0.3.0
    python scripts/extract_changelog.py v0.3.0 --output notes.md

A section runs from its ``## [<version>] - <date>`` heading to the next ``##``
heading, or to the link-reference definitions at the end of the file for the
oldest entry.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

# "## [0.3.0] - 2026-08-07", "## [0.1.0b1] - 2025-12-24", "## [Unreleased]".
VERSION_HEADING = re.compile(r"^##\s+\[(?P<version>[^\]]+)\]")

# "[0.3.0]: https://github.com/..." — the link definitions closing the file,
# which belong to the document rather than to any one section.
LINK_DEFINITION = re.compile(r"^\[[^\]]+\]:\s+\S+")


def extract(text: str, version: str) -> str:
    """Return the changelog body for one version.

    Parameters
    ----------
    text : str
        Full contents of the changelog.
    version : str
        Version to extract, without a leading ``v`` (e.g. ``"0.3.0"``).

    Returns
    -------
    str
        The section body with surrounding blank lines stripped, excluding the
        version heading itself.

    Raises
    ------
    SystemExit
        If the version has no heading, or its section carries no content.
    """
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        match = VERSION_HEADING.match(line)
        if match and match.group("version") == version:
            start = index + 1
            break

    if start is None:
        raise SystemExit(
            f"CHANGELOG.md has no '## [{version}]' section. Add the entry for "
            f"this release before tagging it."
        )

    body: list[str] = []
    for line in lines[start:]:
        if VERSION_HEADING.match(line) or LINK_DEFINITION.match(line):
            break
        body.append(line)

    section = "\n".join(body).strip()
    if not section:
        raise SystemExit(f"The '## [{version}]' section in CHANGELOG.md is empty.")
    return section


def main() -> int:
    """Extract the requested section and write it out.

    Returns
    -------
    int
        0 on success. Failures raise SystemExit with a message instead.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version",
        help="version or tag to extract, e.g. 0.3.0 or v0.3.0",
    )
    parser.add_argument(
        "--changelog",
        type=Path,
        default=CHANGELOG,
        help=f"changelog to read (default: {CHANGELOG})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="file to write the section to (default: stdout)",
    )
    args = parser.parse_args()

    version = args.version.removeprefix("v")
    if version.lower() == "unreleased":
        raise SystemExit("Refusing to extract the 'Unreleased' section as release notes.")

    section = extract(args.changelog.read_text(encoding="utf-8"), version)

    if args.output is None:
        print(section)
    else:
        args.output.write_text(section + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
