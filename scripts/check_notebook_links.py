#!/usr/bin/env python
"""Check the links in the tutorial notebooks.

A link checker pointed straight at a ``.ipynb`` file sees JSON rather than
prose, and every URL that happens to end a source line comes back with the
literal ``\\n`` escape attached — ``.../api/stac/v1`` is reported broken as
``.../api/stac/v1/n``. So this script does two things the checker cannot:

1. Writes each notebook's Markdown cells out as real Markdown, mirroring the
   source tree (``examples/a/b.ipynb`` becomes ``<output>/examples/a/b.ipynb.md``),
   for the link checker to read the web links from.
2. Resolves relative links itself and reports the ones that point nowhere. The
   extracted copy cannot be used for this — a sibling link like
   ``02_clip_and_mosaic.ipynb`` only resolves against the notebook's real
   directory — and this needs no network, so a renamed notebook is caught
   offline.

Only Markdown cells are read. URLs in code cells are endpoints the notebooks
call rather than links a reader clicks, and the notebook job in ``ci.yml``
already exercises those by running the notebooks.

Usage
-----
    python scripts/check_notebook_links.py --output .linkcheck

Exits 1 if any relative link is broken.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GLOB = "examples/**/*.ipynb"

# Markdown inline links: the target inside [text](target).
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)")

# Targets that are not paths on disk.
EXTERNAL = re.compile(r"^(https?:|mailto:|data:|#)")


def markdown_cells(notebook: Path) -> str:
    """Return a notebook's Markdown cells joined into one document.

    Parameters
    ----------
    notebook : pathlib.Path
        Path to the ``.ipynb`` file.

    Returns
    -------
    str
        The Markdown sources, separated by blank lines. Empty when the
        notebook has no Markdown cells.

    Raises
    ------
    SystemExit
        If the file is not valid notebook JSON.
    """
    try:
        content = json.loads(notebook.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise SystemExit(f"{notebook}: not valid notebook JSON: {err}") from err

    blocks = []
    for cell in content.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", [])
        # nbformat allows either a list of lines or a single string.
        blocks.append("".join(source) if isinstance(source, list) else str(source))
    return "\n\n".join(blocks)


def broken_relative_links(notebook: Path, text: str) -> list[str]:
    """Return the relative link targets in ``text`` that do not exist.

    Parameters
    ----------
    notebook : pathlib.Path
        Notebook the text came from; relative targets resolve against its
        directory, which is why this cannot run on the extracted copy.
    text : str
        The notebook's Markdown prose.

    Returns
    -------
    list of str
        Targets that resolve to no file, in the order encountered.
    """
    broken = []
    for target in MARKDOWN_LINK.findall(text):
        if EXTERNAL.match(target):
            continue
        # Drop any anchor: the file is what exists, the heading is not checked.
        path = target.split("#", 1)[0]
        if not path:
            continue
        if not (notebook.parent / path).exists():
            broken.append(target)
    return broken


def main() -> int:
    """Extract notebook prose and check its relative links.

    Returns
    -------
    int
        0 when every relative link resolves, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        help=f"notebooks to check (default: {DEFAULT_GLOB})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="directory to write the extracted Markdown into",
    )
    args = parser.parse_args()

    notebooks = args.notebooks or sorted(REPO_ROOT.glob(DEFAULT_GLOB))
    if not notebooks:
        raise SystemExit(f"No notebooks matched {DEFAULT_GLOB}.")

    written = 0
    failures = 0
    for notebook in notebooks:
        text = markdown_cells(notebook)
        if not text.strip():
            continue

        for target in broken_relative_links(notebook, text):
            failures += 1
            print(f"BROKEN {notebook}: relative link to '{target}' does not exist")

        relative = notebook.resolve().relative_to(REPO_ROOT)
        destination = args.output / relative.parent / f"{relative.name}.md"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
        written += 1

    print(f"Extracted prose from {written} of {len(notebooks)} notebooks into {args.output}")
    if failures:
        print(f"{failures} broken relative link(s).")
        return 1
    print("All relative links resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
