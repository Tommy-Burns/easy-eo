"""Structural guards for the tutorial notebooks in ``examples/``.

CI *executes* the offline notebooks (``pytest --nbmake examples/``, see the
``notebooks`` job in ``.github/workflows/ci.yml``). That leaves three things
execution cannot check, all of which these tests cover:

* the four STAC notebooks are never executed anywhere, so nothing else would
  notice if one lost its Colab badge or grew stored outputs;
* every Colab badge hardcodes the notebook's path on ``main``, so moving or
  renaming a notebook silently breaks its badge;
* the ``requires_network`` metadata flag decides what CI runs
  (``examples/conftest.py``), so it has to agree with what the notebook says
  about itself.

Nothing here executes a notebook or touches the network - these are pure JSON
checks over the committed files.
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"

#: The branch and repository the committed Colab badges point at.
COLAB_PREFIX = "https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/"

NOTEBOOKS = sorted(EXAMPLES.rglob("*.ipynb")) if EXAMPLES.is_dir() else []

pytestmark = pytest.mark.skipif(
    not NOTEBOOKS, reason="examples/ is absent (running against an installed package)"
)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _ids(paths):
    return [str(p.relative_to(EXAMPLES)) for p in paths]


@pytest.fixture(params=NOTEBOOKS, ids=_ids(NOTEBOOKS))
def notebook(request):
    """One committed notebook, as ``(relative path, parsed JSON)``."""
    path = request.param
    return path.relative_to(REPO_ROOT).as_posix(), _load(path)


def test_declares_network_requirement(notebook):
    relpath, nb = notebook
    flag = nb["metadata"].get("easy_eo", {}).get("requires_network")

    assert isinstance(flag, bool), (
        f"{relpath} has no metadata.easy_eo.requires_network flag. It decides "
        "whether CI executes the notebook (see examples/conftest.py), so every "
        "notebook must set it explicitly."
    )

    # The flag must agree with what the notebook tells its reader in the intro.
    states_required = "Network: required" in "".join(nb["cells"][0]["source"])
    assert flag == states_required, (
        f"{relpath}: metadata says requires_network={flag} but the intro cell "
        f"{'does' if states_required else 'does not'} say 'Network: required'."
    )


def test_opens_with_a_colab_badge_for_itself(notebook):
    relpath, nb = notebook
    first = nb["cells"][0]

    assert first["cell_type"] == "markdown", f"{relpath} must open with a markdown cell"

    expected = f"]({COLAB_PREFIX}{relpath})"
    assert expected in "".join(first["source"]), (
        f"{relpath} must open with a Colab badge linking to its own path. "
        f"Expected a link ending in {expected!r} - if the notebook was moved or "
        "renamed, update the badge, examples/README.md and docs/source/tutorials.rst."
    )


def test_installs_itself_when_run_in_colab(notebook):
    relpath, nb = notebook
    setup = "".join(nb["cells"][1]["source"])

    assert nb["cells"][1]["cell_type"] == "code", (
        f"{relpath}: the cell after the title must be the Colab setup cell"
    )
    assert '"google.colab" in sys.modules' in setup and "%pip install" in setup, (
        f"{relpath}: the Colab setup cell must install easy-eo only when running "
        f"in Colab, so it stays a no-op everywhere else. Got:\n{setup}"
    )

    requires_network = nb["metadata"]["easy_eo"]["requires_network"]
    if requires_network:
        assert "stac" in setup, (
            f"{relpath} queries a STAC catalog, so its Colab setup cell must "
            'install the extra: %pip install -q "easy-eo[stac]"'
        )


def test_committed_with_its_outputs(notebook):
    relpath, nb = notebook
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    with_outputs = [c for c in code_cells if c.get("outputs")]

    # Notebooks are read on GitHub far more often than they are run, so every
    # one is committed executed - including the STAC notebooks, whose plots are
    # the only way a reader without the extra installed ever sees them.
    assert with_outputs, (
        f"{relpath} is committed without stored outputs, so it renders as a wall "
        "of code on GitHub. Run it top to bottom and commit the executed notebook."
    )


@pytest.mark.parametrize(
    "index",
    ["examples/README.md", "docs/source/tutorials.rst"],
)
def test_every_notebook_is_indexed(index):
    listing = (REPO_ROOT / index).read_text(encoding="utf-8")
    missing = [
        p.relative_to(EXAMPLES).as_posix()
        for p in NOTEBOOKS
        if p.relative_to(EXAMPLES).as_posix() not in listing
    ]
    assert not missing, f"{index} does not list: {', '.join(missing)}"
