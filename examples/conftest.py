"""Collection rules for executing the tutorial notebooks with nbmake.

Running ``pytest --nbmake examples/`` executes every notebook here top to
bottom, except those that need a live network. A notebook declares that in its
own metadata::

    "metadata": {"easy_eo": {"requires_network": true}}

which is the single source of truth: the STAC tutorials query a live catalog,
so they cannot run on CI and are stored without outputs, while the rest run
offline once ``eeo.datasets`` has cached the sample. Flagging a new notebook
therefore excludes it automatically, with no CI change. ``tests/test_notebooks.py``
checks that the flag agrees with what each notebook says about itself and with
whether its outputs are stored.
"""

import json
from pathlib import Path

_HERE = Path(__file__).parent


def _requires_network(path: Path) -> bool:
    """Return whether a notebook declares that it needs a live network."""
    metadata = json.loads(path.read_text(encoding="utf-8")).get("metadata", {})
    return bool(metadata.get("easy_eo", {}).get("requires_network", False))


collect_ignore = [
    str(path.relative_to(_HERE))
    for path in sorted(_HERE.rglob("*.ipynb"))
    if _requires_network(path)
]
