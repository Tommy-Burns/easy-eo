"""Curated sample-dataset registry for :mod:`eeo.datasets`.

Each entry maps a logical dataset name to the remote assets hosting it, with a
pinned sha256 so a download is verified against a checksum that ships *inside*
the package (never fetched over the same channel as the data). Assets are
served from a GitHub Release; see ``BASE_URL``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

#: Root URL of the hosted sample assets (a GitHub Release; asset paths are
#: flattened by GitHub, so every remote name here is a bare filename).
BASE_URL = "https://github.com/Tommy-Burns/easy-eo/releases/download/sample-data-v1/"

#: Acquisition time of the Sentinel-2 scene the samples are cut from.
_S2_ACQUIRED = datetime(2023, 9, 7, 10, 0, 31, tzinfo=timezone.utc)

#: Human-facing provenance/attribution shown by :func:`eeo.datasets.info` and
#: required by the data licence.
_S2_ATTRIBUTION = (
    "Contains modified Copernicus Sentinel-2 L2A data 2023 (tile T33UUP, "
    "acquired 2023-09-07), processed by ESA; accessed via Microsoft Planetary "
    "Computer. Licensed under the Copernicus open data terms."
)
_DEM_ATTRIBUTION = (
    "Contains modified Copernicus DEM GLO-30 data (© DLR e.V. 2010-2014 and "
    "© Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS "
    "by the European Union and ESA); accessed via Microsoft Planetary Computer."
)


@dataclass(frozen=True)
class Asset:
    """One downloadable file backing a dataset.

    Attributes
    ----------
    remote : str
        Filename appended to :data:`BASE_URL` to form the download URL, and the
        basename used for the cached copy.
    sha256 : str
        Expected hex digest of the file's bytes, verified after download.
    nbytes : int
        Expected size in bytes, used only for progress/reporting.
    band_name : str or None
        Name to assign this asset's band when several single-band assets are
        loaded together as one multi-band dataset; ``None`` for non-raster or
        already-multiband assets.
    """

    remote: str
    sha256: str
    nbytes: int
    band_name: str | None = None


@dataclass(frozen=True)
class Dataset:
    """A named sample dataset made of one or more :class:`Asset` files.

    Attributes
    ----------
    name : str
        Registry key passed to :func:`eeo.datasets.fetch` / ``load``.
    kind : str
        ``"raster"`` or ``"vector"``. Vectors are returned as a path by both
        ``fetch`` and ``load``; rasters are opened by ``load``.
    assets : list of Asset
        Backing files, in band order for a multi-asset raster.
    description : str
        One-line summary for :func:`eeo.datasets.info`.
    attribution : str
        Data licence/attribution text.
    timestamp : datetime.datetime or None
        Acquisition time applied to the loaded raster.
    nodata : float or int or None
        Nodata value applied when a multi-asset raster is stacked in memory.
    """

    name: str
    kind: str
    assets: list[Asset]
    description: str
    attribution: str
    timestamp: datetime | None = None
    nodata: float | int | None = None
    band_names: list[str | None] = field(default_factory=list)

    @property
    def multi(self) -> bool:
        """Whether the dataset is backed by more than one file."""
        return len(self.assets) > 1


#: Per-band single-band files (hosted as raw components). Kept as a multi-asset
#: dataset (``sentinel2_small_bands``) for workflows that want the bands as
#: separate rasters; ``sentinel2_small`` itself is the pre-stacked file below.
_S2_BANDS = [
    Asset(
        "B02.tif",
        "dc60e793fde0b2d88a43a39c1e9ebf8a32fa6215e3c72e6e60057f178e0a328e",
        1500038,
        "blue",
    ),
    Asset(
        "B03.tif",
        "4e8c9e0a4d79271607b1b9ae1c99649302c8b75dc22efb619811cc713c23bd38",
        1542821,
        "green",
    ),
    Asset(
        "B04.tif",
        "56145db8d431d03718a0e082bf5e38dae2afe49eda54f6431ef208687c1be1b0",
        1553010,
        "red",
    ),
    Asset(
        "B08.tif",
        "1f7ca601fae06ebec9a9ccaa7a255ce571f228bd6366f25c6ef90c08bcac6a2a",
        1544666,
        "nir",
    ),
]
_S2_BAND_NAMES: list[str | None] = ["blue", "green", "red", "nir"]

_DATASETS: dict[str, Dataset] = {
    "sentinel2_small": Dataset(
        name="sentinel2_small",
        kind="raster",
        assets=[
            Asset(
                "sentinel2_small.tif",
                "f57186fc62574f123bd15359af55e5efa2e654e2146da662aff2320ff8617b92",
                5750737,
            )
        ],
        description="Sentinel-2 L2A 4-band stack (blue/green/red/nir), 1024x1024 @ 10 m, EPSG:32633.",
        attribution=_S2_ATTRIBUTION,
        timestamp=_S2_ACQUIRED,
        nodata=0,
        band_names=_S2_BAND_NAMES,
    ),
    "sentinel2_small_cog": Dataset(
        name="sentinel2_small_cog",
        kind="raster",
        assets=[
            Asset(
                "sentinel2_small_cog.tif",
                "192ba37bc715a4358e60b56407fed0e2a7f5bf7b57e8a721b6ce7c57b4569f8c",
                7870231,
            )
        ],
        description="Cloud-Optimized GeoTIFF variant of sentinel2_small (for HTTP range-read demos).",
        attribution=_S2_ATTRIBUTION,
        timestamp=_S2_ACQUIRED,
        nodata=0,
        band_names=_S2_BAND_NAMES,
    ),
    "sentinel2_small_bands": Dataset(
        name="sentinel2_small_bands",
        kind="raster",
        assets=_S2_BANDS,
        description="The blue/green/red/nir bands of sentinel2_small as separate single-band files.",
        attribution=_S2_ATTRIBUTION,
        timestamp=_S2_ACQUIRED,
        nodata=0,
        band_names=_S2_BAND_NAMES,
    ),
    "dem_small": Dataset(
        name="dem_small",
        kind="raster",
        assets=[
            Asset(
                "DEM.tif",
                "4aa03df4de877570d728ace4b0c07e71cb8be79a0c0640c6de602116ad1b3f88",
                3413595,
                "elevation",
            )
        ],
        description="Copernicus GLO-30 DEM warped onto the sentinel2_small grid, float32 metres.",
        attribution=_DEM_ATTRIBUTION,
        band_names=["elevation"],
    ),
    "dem_small_cog": Dataset(
        name="dem_small_cog",
        kind="raster",
        assets=[
            Asset(
                "DEM_COG.tif",
                "c1d763c4dcbec033695ab0d73c75fdbc58d5a2c0d91eb1fe443c13d48644ee27",
                5385324,
                "elevation",
            )
        ],
        description="Cloud-Optimized GeoTIFF variant of dem_small.",
        attribution=_DEM_ATTRIBUTION,
        band_names=["elevation"],
    ),
    "sentinel2_small_boundary": Dataset(
        name="sentinel2_small_boundary",
        kind="vector",
        assets=[
            Asset(
                "roi.gpkg",
                "788a1800caed8d03ef7893de1a61ead281edd901ea049cdf969c5749c0af0744",
                98304,
            )
        ],
        description="Region-of-interest polygon (GeoPackage, EPSG:4326) inside the sentinel2_small footprint.",
        attribution=_S2_ATTRIBUTION,
    ),
}


def get_dataset(name: str) -> Dataset:
    """Return the registry entry for ``name`` or raise a listing error.

    Parameters
    ----------
    name : str
        Dataset name.

    Returns
    -------
    Dataset
        The matching registry entry.

    Raises
    ------
    KeyError
        If ``name`` is not a registered dataset. The message lists the
        available names.
    """
    try:
        return _DATASETS[name]
    except KeyError:
        available = ", ".join(sorted(_DATASETS))
        raise KeyError(f'unknown dataset "{name}"; available datasets are: {available}') from None


def available() -> list[str]:
    """Return the sorted names of every registered sample dataset.

    Returns
    -------
    list of str
        Dataset names accepted by :func:`eeo.datasets.fetch` and
        :func:`eeo.datasets.load`.
    """
    return sorted(_DATASETS)
