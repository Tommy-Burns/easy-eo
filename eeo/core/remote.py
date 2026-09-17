"""Opening rasters that are not plain local files.

GDAL reads a raster straight from a URL or an object store, and addresses a
file inside an archive through its virtual filesystems (``/vsizip/``,
``/vsitar/``, ...). Easy-EO's loaders therefore cannot assume every path is
something :func:`os.path.isfile` will confirm. This module holds the two
things they need to tell the difference: which paths are not local files, and
the GDAL configuration a remote read wants.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

import rasterio as rio

from eeo.core.types import StrPath

#: URL schemes GDAL reads directly, without a local copy.
REMOTE_SCHEMES = ("http://", "https://", "ftp://", "s3://", "gs://", "az://")

#: GDAL virtual filesystem prefixes that fetch over a network.
REMOTE_VSI_PREFIXES = (
    "/vsicurl/",
    "/vsicurl_streaming/",
    "/vsis3/",
    "/vsigs/",
    "/vsiaz/",
    "/vsioss/",
    "/vsiswift/",
    "/vsihdfs/",
)

# GDAL settings for reading a remote COG efficiently. Object stores answer a
# directory probe by listing the whole container, which costs far more than the
# read itself; HTTP/2 multiplexing and the VSI cache keep the range requests for
# the tiles we actually want.
GDAL_HTTP_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "VSI_CACHE": "TRUE",
}


def is_remote(path: StrPath) -> bool:
    """Report whether a path is read over a network rather than from disk.

    Parameters
    ----------
    path : str or path-like
        Path or URL a raster would be opened from.

    Returns
    -------
    bool
        True for an ``http(s)``/``ftp``/``s3``/``gs``/``az`` URL or a GDAL
        virtual filesystem prefix that fetches remotely, False otherwise.
    """
    text = os.fspath(path)
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return lowered.startswith(REMOTE_SCHEMES) or lowered.startswith(REMOTE_VSI_PREFIXES)


def is_gdal_path(path: StrPath) -> bool:
    """Report whether GDAL, not the filesystem, decides if a path exists.

    True for anything :func:`is_remote` covers, and for the local virtual
    filesystems — ``/vsizip/archive.zip/band.tif`` names a file inside an
    archive, which exists to GDAL but not to :func:`os.path.isfile`.

    Parameters
    ----------
    path : str or path-like
        Path or URL a raster would be opened from.

    Returns
    -------
    bool
        True when existence must be left to GDAL to determine.
    """
    text = os.fspath(path)
    if not isinstance(text, str):
        return False
    return is_remote(text) or text.lower().startswith("/vsi")


@contextlib.contextmanager
def open_env(path: StrPath) -> Iterator[Any]:
    """Apply the GDAL configuration an open of ``path`` wants.

    A remote raster is opened under :data:`GDAL_HTTP_ENV`; a local one needs
    nothing, and gets an empty context rather than a new GDAL environment.

    Parameters
    ----------
    path : str or path-like
        Path or URL about to be opened.

    Yields
    ------
    None
        Inside the context, the configuration is in force.

    Notes
    -----
    A rasterio environment is thread-local and lasts as long as the ``with``
    block, so it covers the open and any read made through the returned
    dataset on this thread. The lazy backend's reads happen later, inside
    dask's worker threads, and run under GDAL's defaults; they still issue
    range requests, which is what bounds them.
    """
    if is_remote(path):
        with rio.Env(**GDAL_HTTP_ENV):
            yield
    else:
        yield
