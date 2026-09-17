"""Run a snippet of Easy-EO in a child process with a hard memory cap.

The cap is ``RLIMIT_AS``, so an allocation past it raises ``MemoryError``
instead of the machine swapping or the kernel's OOM killer picking a victim.
Three things about that, each learned the hard way (see the WP-16 notes):

* ``RLIMIT_AS`` limits *address space*, which is always larger than the
  resident memory a tool like ``top`` reports — importing Easy-EO alone
  reserves a few hundred megabytes of it. A cap is therefore chosen with
  headroom, and what a test asserts about size is the child's own peak RSS.
* GDAL sizes its block cache once, at first use, from whatever limit is in
  force, so a cap silently shrinks it. ``GDAL_CACHEMAX`` is pinned in the
  child's environment before the limit applies, so every run gets the same
  cache regardless of the cap.
* ``ru_maxrss`` is inherited across fork *and* exec on Linux, so a child of a
  large parent reports the parent's peak. The child reads its own ``VmHWM``
  instead, which ``execve`` resets.

Only Linux is covered: ``RLIMIT_AS`` is not honoured meaningfully on macOS and
does not exist on Windows.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass

import pytest

#: GDAL block cache for every capped run, in megabytes. Pinned so the cap
#: cannot change it, and small enough to leave the cap to the code under test.
GDAL_CACHE_MB = 256

#: Prelude every snippet runs under: headless plots, and the reporting helper
#: the snippet calls to hand a result back to the parent.
_PRELUDE = """
import json, sys, matplotlib
matplotlib.use("Agg")

def peak_rss_mib():
    with open("/proc/self/status", encoding="utf-8") as status:
        for line in status:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("VmHWM missing from /proc/self/status")

def report(**fields):
    fields["peak_rss_mib"] = peak_rss_mib()
    print("__RESULT__" + json.dumps(fields))
"""


@dataclass
class CappedRun:
    """What a capped child process did."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        """True when the child ran to completion."""
        return self.returncode == 0

    @property
    def out_of_memory(self) -> bool:
        """True when the child died for want of memory rather than a bug."""
        return not self.ok and ("MemoryError" in self.stderr or "Unable to allocate" in self.stderr)

    @property
    def result(self) -> dict:
        """The fields the snippet passed to ``report()``."""
        for line in self.stdout.splitlines():
            if line.startswith("__RESULT__"):
                return json.loads(line[len("__RESULT__") :])
        raise AssertionError(f"the snippet reported nothing.\nstdout: {self.stdout}\n{self.stderr}")

    @property
    def peak_rss_mib(self) -> float:
        """The child's peak resident memory, in mebibytes."""
        return self.result["peak_rss_mib"]

    @property
    def failure(self) -> str:
        """The child's last line of stderr, for a test's failure message."""
        lines = self.stderr.strip().splitlines()
        return lines[-1] if lines else f"exited with {self.returncode}"


def requires_capping() -> None:
    """Skip the calling test where an address-space cap means nothing."""
    if not sys.platform.startswith("linux"):
        pytest.skip("RLIMIT_AS caps are only meaningful on Linux")
    if not os.path.exists("/proc/self/status"):
        pytest.skip("/proc/self/status is needed to measure the child's peak")


def run_capped(snippet: str, cap_mib: int, timeout: int = 900) -> CappedRun:
    """Run ``snippet`` in a child process limited to ``cap_mib`` of address space.

    Parameters
    ----------
    snippet : str
        Python source. It may call ``report(**fields)`` to return values, and
        the child's peak RSS is added to them.
    cap_mib : int
        Address-space cap for the child, in mebibytes.
    timeout : int, default 900
        Seconds to wait before killing the child.

    Returns
    -------
    CappedRun
        The child's exit status, output, and whatever it reported.
    """
    requires_capping()
    import resource

    def apply_cap() -> None:
        limit = cap_mib * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    # Pinned before the cap applies, so GDAL does not size its cache from it.
    env = dict(os.environ, GDAL_CACHEMAX=str(GDAL_CACHE_MB), MPLBACKEND="Agg")
    completed = subprocess.run(
        [sys.executable, "-c", _PRELUDE + textwrap.dedent(snippet)],
        capture_output=True,
        text=True,
        preexec_fn=apply_cap,  # noqa: PLW1509 - the cap must apply before exec
        env=env,
        timeout=timeout,
        check=False,
    )
    return CappedRun(completed.returncode, completed.stdout, completed.stderr)
