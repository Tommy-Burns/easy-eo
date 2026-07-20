"""Guard that the generated ``eeo/core/core.pyi`` stays in sync with source.

The stub is produced by ``scripts/generate_core_stub.py`` from ``core.py`` and
the decorator registry; it must never be hand-edited. This test fails if the
committed stub differs from a fresh generation, i.e. someone added or changed a
bound op (or a ``core.py`` method) without regenerating.

The generator formats its output with ``ruff``; when ruff is unavailable (it
ships in the ``dev`` extra) the test skips rather than giving a false failure.
CI installs the dev extra, so the check runs there.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR = REPO_ROOT / "scripts" / "generate_core_stub.py"


def test_core_stub_is_fresh():
    if shutil.which("ruff") is None:
        pytest.skip("ruff not on PATH; stub freshness needs the dev extra installed")

    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        "eeo/core/core.pyi is stale. Regenerate it with:\n"
        "    python scripts/generate_core_stub.py\n\n"
        f"{result.stderr}"
    )
