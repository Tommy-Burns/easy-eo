"""Local caching and checksum-verified download of sample assets.

Downloads use only the Python standard library (``urllib`` + ``hashlib``), so
:mod:`eeo.datasets` adds no runtime dependency. Files are cached under a
per-user directory and re-verified on every access, so a corrupt or partial
download is transparently repaired.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from eeo.core.exceptions import EEOError

from ._registry import BASE_URL, Asset

_CHUNK = 1 << 20  # 1 MiB streaming reads keep peak memory flat on large assets.


class DatasetError(EEOError):
    """Raised when a sample dataset cannot be fetched or verified."""


def cache_dir() -> Path:
    """Return the directory sample assets are cached in, creating it.

    Resolution order:

    1. ``EEO_DATA_DIR`` environment variable, if set (used by tests and CI).
    2. ``XDG_CACHE_HOME/easy-eo`` when ``XDG_CACHE_HOME`` is set.
    3. ``~/.cache/easy-eo`` otherwise.

    Returns
    -------
    pathlib.Path
        The cache directory. It is created (with parents) if absent.
    """
    override = os.environ.get("EEO_DATA_DIR")
    if override:
        base = Path(override)
    elif os.environ.get("XDG_CACHE_HOME"):
        base = Path(os.environ["XDG_CACHE_HOME"]) / "easy-eo"
    else:
        base = Path.home() / ".cache" / "easy-eo"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` atomically (via a temp file + rename)."""
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), suffix=".part")
    tmp = Path(tmp_name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "easy-eo"})
        with urllib.request.urlopen(request) as response, os.fdopen(tmp_fd, "wb") as out:
            shutil.copyfileobj(response, out, _CHUNK)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise DatasetError(
            f"failed to download sample data from {url}: {exc}. "
            "Check your network connection and try again."
        ) from exc
    os.replace(tmp, dest)


def ensure_asset(asset: Asset) -> Path:
    """Return the cached path of ``asset``, downloading and verifying if needed.

    A cached copy whose checksum matches is returned untouched. A missing or
    checksum-mismatched copy is (re)downloaded and re-verified; a download that
    still fails verification raises rather than returning corrupt data.

    Parameters
    ----------
    asset : Asset
        Registry asset to materialize.

    Returns
    -------
    pathlib.Path
        Path to the verified local file.

    Raises
    ------
    DatasetError
        If the download fails, or the downloaded bytes do not match the pinned
        sha256 (indicating corruption or a changed remote file).
    """
    dest = cache_dir() / asset.remote
    if dest.is_file() and _sha256(dest) == asset.sha256:
        return dest

    url = BASE_URL + asset.remote
    _download(url, dest)

    digest = _sha256(dest)
    if digest != asset.sha256:
        dest.unlink(missing_ok=True)
        raise DatasetError(
            f"checksum mismatch for {asset.remote}: expected {asset.sha256}, got "
            f"{digest}. The remote file may have changed or the download was "
            "corrupted; please retry, and report the issue if it persists."
        )
    return dest
