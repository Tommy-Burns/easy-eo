"""Attribute-addressable access to the curated sample files.

:func:`load_sample_dataset` returns a :class:`SampleDataset` whose attributes
are the individual sample files, so a file can be opened by dotted name instead
of a string key::

    from eeo.datasets import load_sample_dataset
    from eeo import load_raster

    sd = load_sample_dataset()                     # instant; no download
    scene = load_raster(sd.sentinel2_cog_stacked)  # downloads that one file
    dem = load_raster(sd.copernicus_dem)
    blue = load_raster(sd.sentinel2_blue)

Each attribute is a lazy :class:`SamplePath`: holding it touches no network, and
the file is downloaded and checksum-verified only when it is actually opened
(when :func:`eeo.load_raster` resolves the path). This is a thin convenience
layer over the string-keyed :func:`eeo.datasets.load` / :func:`eeo.datasets.fetch`
API — those remain the way to load the multi-file ``sentinel2_small_bands``
stack (with band names *and* acquisition timestamp set) and to describe or list
datasets.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import _cache
from ._registry import SAMPLE_FILES, Asset


class SamplePath(os.PathLike):
    """A lazy path to one cached sample file.

    The object is inert until its filesystem path is needed: passing it to
    :func:`eeo.load_raster`, calling :func:`os.fspath` on it, or calling
    :meth:`fetch` downloads the file (once) and verifies it against its pinned
    checksum. Because it implements :class:`os.PathLike`, it can be used
    anywhere a path string is accepted.

    Parameters
    ----------
    name : str
        The readable attribute name this file is exposed under.
    asset : Asset
        The registry asset this path resolves to.
    """

    def __init__(self, name: str, asset: Asset) -> None:
        self._name = name
        self._asset = asset
        self._resolved: Path | None = None

    @property
    def name(self) -> str:
        """The readable attribute name (e.g. ``"copernicus_dem"``)."""
        return self._name

    def fetch(self) -> Path:
        """Download (if needed), verify, and return the cached file path.

        The result is memoized, so repeated resolution (``load_raster`` checks
        existence and then opens the file) verifies the checksum only once.

        Returns
        -------
        pathlib.Path
            Path to the verified local file.

        Raises
        ------
        DatasetError
            If the download fails or the bytes fail checksum verification.
        """
        if self._resolved is None:
            self._resolved = _cache.ensure_asset(self._asset)
        return self._resolved

    def __fspath__(self) -> str:
        return str(self.fetch())

    def __str__(self) -> str:
        # Side-effect-free: show the cached path once fetched, else the target
        # filename. Never triggers a download (unlike __fspath__), so printing a
        # dataset's source in describe() stays cheap and offline.
        if self._resolved is not None:
            return str(self._resolved)
        return self._asset.remote

    def __repr__(self) -> str:
        state = "cached" if self._resolved is not None else "not fetched"
        return f"<SamplePath {self._name} -> {self._asset.remote} ({state})>"


class SampleDataset:
    """A namespace of the curated sample files, one per attribute.

    Obtain one from :func:`load_sample_dataset`. Every attribute is a
    :class:`SamplePath` — inert until opened — so constructing the namespace
    never touches the network. Attributes are declared explicitly (below) so
    editors offer autocompletion and type checkers see them.

    Attributes
    ----------
    sentinel2_stacked : SamplePath
        Sentinel-2 blue/green/red/nir as one 4-band GeoTIFF.
    sentinel2_cog_stacked : SamplePath
        Cloud-Optimized GeoTIFF variant of ``sentinel2_stacked``.
    sentinel2_blue, sentinel2_green, sentinel2_red, sentinel2_nir : SamplePath
        The four Sentinel-2 bands as separate single-band GeoTIFFs.
    copernicus_dem : SamplePath
        Copernicus GLO-30 DEM warped onto the Sentinel-2 grid (float32 metres).
    copernicus_dem_cog : SamplePath
        Cloud-Optimized GeoTIFF variant of ``copernicus_dem``.
    boundary : SamplePath
        Region-of-interest polygon (GeoPackage). A vector, not a raster: read it
        with GeoPandas (``gpd.read_file(sd.boundary)``), not ``load_raster``.
    """

    sentinel2_stacked: SamplePath
    sentinel2_cog_stacked: SamplePath
    sentinel2_blue: SamplePath
    sentinel2_green: SamplePath
    sentinel2_red: SamplePath
    sentinel2_nir: SamplePath
    copernicus_dem: SamplePath
    copernicus_dem_cog: SamplePath
    boundary: SamplePath

    def __init__(self, prefetch: bool = False) -> None:
        for attr, asset in SAMPLE_FILES.items():
            setattr(self, attr, SamplePath(attr, asset))
        if prefetch:
            for attr in SAMPLE_FILES:
                getattr(self, attr).fetch()

    def __repr__(self) -> str:
        return f"SampleDataset({', '.join(SAMPLE_FILES)})"


def load_sample_dataset(prefetch: bool = False) -> SampleDataset:
    """Return the sample files as an attribute-addressable namespace.

    Each attribute of the returned object is a lazy :class:`SamplePath` that can
    be passed straight to :func:`eeo.load_raster`; the underlying file is
    downloaded and checksum-verified on first use, then cached. Constructing the
    namespace itself performs no network access.

    Parameters
    ----------
    prefetch : bool, default False
        If ``True``, download and verify every sample file immediately (useful
        before going offline). If ``False`` (the default), each file is fetched
        lazily the first time it is opened.

    Returns
    -------
    SampleDataset
        A namespace whose attributes are the individual sample files. See its
        class documentation for the available names.

    See Also
    --------
    eeo.datasets.load : Open a dataset by string key, with band names and the
        acquisition timestamp set (and the multi-file band stack).
    eeo.datasets.fetch : Get cached file path(s) for a dataset by string key.

    Notes
    -----
    Opening a single file with ``load_raster(sd.<name>)`` reads band names from
    the file's GDAL descriptions but does *not* set the acquisition timestamp;
    use :func:`eeo.datasets.load` when you need the timestamp too.

    Examples
    --------
    >>> from eeo.datasets import load_sample_dataset
    >>> from eeo import load_raster
    >>> sd = load_sample_dataset()
    >>> scene = load_raster(sd.sentinel2_cog_stacked)  # doctest: +SKIP
    >>> dem = load_raster(sd.copernicus_dem)           # doctest: +SKIP
    """
    return SampleDataset(prefetch=prefetch)
