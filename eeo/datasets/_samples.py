"""Attribute-addressable access to the curated sample files.

:func:`load_sample_dataset` returns a :class:`SampleDataset` whose attributes
are the individual sample files, so a file is opened by dotted name — never by a
hard-coded string key::

    from eeo.datasets import load_sample_dataset
    from eeo import load_raster

    sd = load_sample_dataset()                     # instant; no download
    scene = load_raster(sd.sentinel2_cog_stacked)  # downloads that one file
    dem = load_raster(sd.copernicus_dem)
    blue = load_raster(sd.sentinel2_blue)

Each attribute is a lazy :class:`SamplePath`: holding it touches no network, and
the file is downloaded and checksum-verified only when it is actually opened
(when :func:`eeo.load_raster` resolves the path). Provenance travels with the
handle — :meth:`SamplePath.info` and :attr:`SamplePath.attribution` carry the
required Copernicus attribution.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import _cache
from ._registry import SAMPLE_FILES, SampleFile


class SamplePath(os.PathLike):
    """A lazy path to one cached sample file.

    The object is inert until its filesystem path is needed: passing it to
    :func:`eeo.load_raster`, calling :func:`os.fspath` on it, or accessing
    :attr:`path` downloads the file (once) and verifies it against its pinned
    checksum. Because it implements :class:`os.PathLike`, it can be used anywhere
    a path string is accepted (``load_raster``, ``geopandas.read_file``, ...).

    Parameters
    ----------
    name : str
        The readable attribute name this file is exposed under.
    sample : SampleFile
        The registry entry this handle resolves to (asset + provenance).
    """

    def __init__(self, name: str, sample: SampleFile) -> None:
        self._name = name
        self._sample = sample
        self._resolved: Path | None = None

    @property
    def name(self) -> str:
        """The readable attribute name (e.g. ``"copernicus_dem"``)."""
        return self._name

    @property
    def kind(self) -> str:
        """``"raster"`` or ``"vector"``."""
        return self._sample.kind

    @property
    def description(self) -> str:
        """One-line description of the file's contents."""
        return self._sample.description

    @property
    def attribution(self) -> str:
        """Data licence/attribution text that must accompany reuse."""
        return self._sample.attribution

    @property
    def path(self) -> Path:
        """The cached file path, downloading and verifying if needed."""
        return self.fetch()

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
            self._resolved = _cache.ensure_asset(self._sample.asset)
        return self._resolved

    def info(self) -> str:
        """Return a human-readable description and attribution for this file.

        Returns
        -------
        str
            A multi-line summary: name, kind, description, backing filename, and
            the data licence/attribution required for reuse.
        """
        return (
            f"{self._name} ({self._sample.kind})\n"
            f"  {self._sample.description}\n"
            f"  file: {self._sample.asset.remote}\n"
            f"  attribution: {self._sample.attribution}"
        )

    def __fspath__(self) -> str:
        return str(self.fetch())

    def __str__(self) -> str:
        # Side-effect-free: show the cached path once fetched, else the target
        # filename. Never triggers a download (unlike __fspath__), so printing a
        # dataset's source in describe() stays cheap and offline.
        if self._resolved is not None:
            return str(self._resolved)
        return self._sample.asset.remote

    def __repr__(self) -> str:
        state = "cached" if self._resolved is not None else "not fetched"
        return f"<SamplePath {self._name} -> {self._sample.asset.remote} ({state})>"


class SampleDataset:
    """A namespace of the curated sample files, one per attribute.

    Obtain one from :func:`load_sample_dataset`. Every attribute is a
    :class:`SamplePath` — inert until opened — so constructing the namespace
    never touches the network. Attributes are declared explicitly (below) so
    editors offer autocompletion and type checkers see them. The namespace is
    iterable, yielding its :class:`SamplePath` handles.

    Attributes
    ----------
    sentinel2_stacked : SamplePath
        Sentinel-2 blue/green/red/nir as one 4-band GeoTIFF.
    sentinel2_cog_stacked : SamplePath
        Cloud-Optimized GeoTIFF variant of ``sentinel2_stacked``.
    sentinel2_blue, sentinel2_green, sentinel2_red, sentinel2_nir : SamplePath
        The four Sentinel-2 bands as separate single-band GeoTIFFs.
    sentinel2_blue_cog, sentinel2_green_cog, sentinel2_red_cog, sentinel2_nir_cog : SamplePath
        Cloud-Optimized GeoTIFF variants of the four single-band files.
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
    sentinel2_blue_cog: SamplePath
    sentinel2_green: SamplePath
    sentinel2_green_cog: SamplePath
    sentinel2_red: SamplePath
    sentinel2_red_cog: SamplePath
    sentinel2_nir: SamplePath
    sentinel2_nir_cog: SamplePath
    copernicus_dem: SamplePath
    copernicus_dem_cog: SamplePath
    boundary: SamplePath

    def __init__(self, prefetch: bool = False) -> None:
        for attr, sample in SAMPLE_FILES.items():
            setattr(self, attr, SamplePath(attr, sample))
        if prefetch:
            for attr in SAMPLE_FILES:
                getattr(self, attr).fetch()

    def __iter__(self):
        """Iterate the :class:`SamplePath` handles in registry order."""
        return (getattr(self, attr) for attr in SAMPLE_FILES)

    def __len__(self) -> int:
        return len(SAMPLE_FILES)

    def __repr__(self) -> str:
        return f"SampleDataset({', '.join(SAMPLE_FILES)})"


def load_sample_dataset(prefetch: bool = False) -> SampleDataset:
    """Return the sample files as an attribute-addressable namespace.

    This is the only supported way to reach the bundled samples: each attribute
    of the returned object is a lazy :class:`SamplePath` that can be passed
    straight to :func:`eeo.load_raster`. The underlying file is downloaded and
    checksum-verified on first use, then cached; constructing the namespace
    itself performs no network access.

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

    Notes
    -----
    Opening a file with ``load_raster(sd.<name>)`` reads band names from the
    file's GDAL descriptions. Provenance and the required Copernicus attribution
    are available on each handle via :meth:`SamplePath.info` and
    :attr:`SamplePath.attribution`.

    Examples
    --------
    >>> from eeo.datasets import load_sample_dataset
    >>> from eeo import load_raster
    >>> sd = load_sample_dataset()
    >>> scene = load_raster(sd.sentinel2_cog_stacked)  # doctest: +SKIP
    >>> dem = load_raster(sd.copernicus_dem)           # doctest: +SKIP
    >>> print(sd.copernicus_dem.attribution)           # doctest: +SKIP
    """
    return SampleDataset(prefetch=prefetch)
