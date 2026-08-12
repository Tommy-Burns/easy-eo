#!/usr/bin/env python
"""Verify every "Open in Colab" link points at a notebook that exists.

These links cannot be checked over HTTP. Colab is a single-page app: it answers
200 for any URL, including a notebook path that does not exist and a repository
that does not exist, and only reports the problem in the browser once its
JavaScript has run. A link checker therefore confirms nothing about them beyond
Colab being online, which is why they are checked structurally here instead.

A Colab link embeds everything needed to check it offline::

    https://colab.research.google.com/github/<owner>/<repo>/blob/<ref>/<path>

so this verifies that ``<owner>/<repo>`` is this project, that ``<ref>`` is the
branch Colab will actually find the file on, and that ``<path>`` exists. Inside
a notebook the link must also point at that same notebook — a badge copied to a
new notebook and left unedited opens the wrong tutorial, which is the failure
this catches most often.

Usage
-----
    python scripts/check_colab_links.py

Exits 1 if any link is wrong.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# GitHub treats owner and repository names case-insensitively, and the project
# spells its own URL both ways (pyproject uses lowercase, the badges do not),
# so the comparison is casefolded rather than exact.
EXPECTED_REPO = "Tommy-Burns/easy-eo"

# The branch Colab loads the notebook from. A link to a branch that no longer
# exists fails silently in exactly the same way a bad path does.
EXPECTED_REF = "main"

# Files that carry Colab badges: the notebooks themselves, the Markdown, and
# the tutorials page in the docs.
SCAN_PATTERNS = ("*.md", "examples/**/*.md", "examples/**/*.ipynb", "docs/source/**/*.rst")

COLAB_LINK = re.compile(
    r"https://colab\.research\.google\.com/github/"
    r"(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+)/blob/(?P<ref>[^/\s]+)/(?P<path>[^\s)\"'\\>]+)"
)


def scan_files() -> list[Path]:
    """Return the files to search, de-duplicated and sorted.

    Returns
    -------
    list of pathlib.Path
        Existing files matching any pattern in SCAN_PATTERNS.
    """
    found: set[Path] = set()
    for pattern in SCAN_PATTERNS:
        found.update(path for path in REPO_ROOT.glob(pattern) if path.is_file())
    return sorted(found)


def check_file(path: Path) -> list[str]:
    """Return a message for each bad Colab link in one file.

    Parameters
    ----------
    path : pathlib.Path
        File to inspect. Read as raw text, so notebook JSON works without
        parsing: the link is checked by its structure, not by fetching it.

    Returns
    -------
    list of str
        One ``file:line: problem`` message per bad link. Empty when they all
        check out.
    """
    text = path.read_text(encoding="utf-8")
    relative = path.relative_to(REPO_ROOT).as_posix()
    problems = []

    for match in COLAB_LINK.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        where = f"{relative}:{line}"
        repo = f"{match['owner']}/{match['repo']}"
        target = match["path"]

        if repo.casefold() != EXPECTED_REPO.casefold():
            problems.append(f"{where}: points at '{repo}', expected '{EXPECTED_REPO}'")
            continue
        if match["ref"] != EXPECTED_REF:
            problems.append(
                f"{where}: points at branch '{match['ref']}', expected '{EXPECTED_REF}'"
            )
            continue
        if not (REPO_ROOT / target).is_file():
            problems.append(f"{where}: '{target}' does not exist in the repository")
            continue
        # A notebook's own badge must open that notebook, not another one.
        if path.suffix == ".ipynb" and target != relative:
            problems.append(f"{where}: badge opens '{target}' instead of this notebook")

    return problems


def main() -> int:
    """Check every Colab link in the repository.

    Returns
    -------
    int
        0 when every link is valid, 1 otherwise.
    """
    argparse.ArgumentParser(description=__doc__).parse_args()

    problems: list[str] = []
    links = 0
    for path in scan_files():
        text = path.read_text(encoding="utf-8")
        links += len(COLAB_LINK.findall(text))
        problems.extend(check_file(path))

    for problem in problems:
        print(f"BROKEN {problem}")

    if problems:
        print(f"\n{len(problems)} of {links} Colab links are wrong.")
        return 1
    print(f"All {links} Colab links point at notebooks that exist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
