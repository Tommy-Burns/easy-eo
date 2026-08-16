"""Curated sample-file registry for :mod:`eeo.datasets`.

Each entry of :data:`SAMPLE_FILES` maps a readable name to one downloadable
asset with a pinned sha256 that ships *inside* the package (never fetched over
the same channel as the data), plus the provenance/attribution that must
accompany reuse. Assets are served from a GitHub Release; see :data:`BASE_URL`.

This registry is an internal detail: users reach the files through
:func:`eeo.datasets.load_sample_dataset`, never by these names as strings.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Root URL of the hosted sample assets (a GitHub Release; asset paths are
#: flattened by GitHub, so every remote name here is a bare filename).
BASE_URL = "https://github.com/Tommy-Burns/easy-eo/releases/download/sample-data-v1/"

#: Human-facing provenance/attribution required by the data licence, surfaced by
#: :meth:`eeo.datasets.SamplePath.info` / :attr:`~eeo.datasets.SamplePath.attribution`.
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
    """One downloadable file.

    Attributes
    ----------
    remote : str
        Filename appended to :data:`BASE_URL` to form the download URL, and the
        basename used for the cached copy.
    sha256 : str
        Expected hex digest of the file's bytes, verified after download.
    nbytes : int
        Expected size in bytes, used only for progress/reporting.
    """

    remote: str
    sha256: str
    nbytes: int


@dataclass(frozen=True)
class SampleFile:
    """One curated sample exposed as a namespace attribute.

    Attributes
    ----------
    asset : Asset
        The single downloadable file backing this sample.
    kind : str
        ``"raster"`` (open with :func:`eeo.load_raster`) or ``"vector"`` (read
        the path with GeoPandas).
    description : str
        One-line summary shown by :meth:`eeo.datasets.SamplePath.info`.
    attribution : str
        Data licence/attribution text.
    """

    asset: Asset
    kind: str
    description: str
    attribution: str


#: The curated samples, keyed by the attribute name they are exposed under on
#: :class:`eeo.datasets.SampleDataset`. Band files use the library's colour band
#: names. This is the single source of truth for both the download machinery and
#: the namespace attributes. Every raster published on the release has both a
#: plain and a Cloud-Optimized variant here, the latter suffixed ``_cog``.
SAMPLE_FILES: dict[str, SampleFile] = {
    "sentinel2_stacked": SampleFile(
        Asset(
            "sentinel2_small.tif",
            "f57186fc62574f123bd15359af55e5efa2e654e2146da662aff2320ff8617b92",
            5750737,
        ),
        "raster",
        "Sentinel-2 L2A 4-band stack (blue/green/red/nir), 1024x1024 @ 10 m, EPSG:32633.",
        _S2_ATTRIBUTION,
    ),
    "sentinel2_cog_stacked": SampleFile(
        Asset(
            "sentinel2_small_cog.tif",
            "192ba37bc715a4358e60b56407fed0e2a7f5bf7b57e8a721b6ce7c57b4569f8c",
            7870231,
        ),
        "raster",
        "Cloud-Optimized GeoTIFF variant of the 4-band Sentinel-2 stack (HTTP range-read demos).",
        _S2_ATTRIBUTION,
    ),
    "sentinel2_blue": SampleFile(
        Asset(
            "B02.tif",
            "dc60e793fde0b2d88a43a39c1e9ebf8a32fa6215e3c72e6e60057f178e0a328e",
            1500038,
        ),
        "raster",
        "Sentinel-2 L2A band B02 (blue), 1024x1024 @ 10 m, EPSG:32633.",
        _S2_ATTRIBUTION,
    ),
    "sentinel2_blue_cog": SampleFile(
        Asset(
            "B02_COG.tif",
            "555f6fb67522148c05d7bb9a4f49d157cde2d9da00d1304ed953bfd383c9ead1",
            2293197,
        ),
        "raster",
        "Cloud-Optimized GeoTIFF variant of the blue band (B02).",
        _S2_ATTRIBUTION,
    ),
    "sentinel2_green": SampleFile(
        Asset(
            "B03.tif",
            "4e8c9e0a4d79271607b1b9ae1c99649302c8b75dc22efb619811cc713c23bd38",
            1542821,
        ),
        "raster",
        "Sentinel-2 L2A band B03 (green), 1024x1024 @ 10 m, EPSG:32633.",
        _S2_ATTRIBUTION,
    ),
    "sentinel2_green_cog": SampleFile(
        Asset(
            "B03_COG.tif",
            "0f0e6cd16141725fe4dfd749a911f3846429b0cd7c222d543d90848ac1f95833",
            2404804,
        ),
        "raster",
        "Cloud-Optimized GeoTIFF variant of the green band (B03).",
        _S2_ATTRIBUTION,
    ),
    "sentinel2_red": SampleFile(
        Asset(
            "B04.tif",
            "56145db8d431d03718a0e082bf5e38dae2afe49eda54f6431ef208687c1be1b0",
            1553010,
        ),
        "raster",
        "Sentinel-2 L2A band B04 (red), 1024x1024 @ 10 m, EPSG:32633.",
        _S2_ATTRIBUTION,
    ),
    "sentinel2_red_cog": SampleFile(
        Asset(
            "B04_COG.tif",
            "5781624003c04ae978196df98dd54e742fe38d1bd5b56237ffc4162209eafc28",
            2436025,
        ),
        "raster",
        "Cloud-Optimized GeoTIFF variant of the red band (B04).",
        _S2_ATTRIBUTION,
    ),
    "sentinel2_nir": SampleFile(
        Asset(
            "B08.tif",
            "1f7ca601fae06ebec9a9ccaa7a255ce571f228bd6366f25c6ef90c08bcac6a2a",
            1544666,
        ),
        "raster",
        "Sentinel-2 L2A band B08 (nir), 1024x1024 @ 10 m, EPSG:32633.",
        _S2_ATTRIBUTION,
    ),
    "sentinel2_nir_cog": SampleFile(
        Asset(
            "B08_COG.tif",
            "1488f68cad9f37a90ffef9eb62716e2cab77bd7c22954a08e2f2a5d06ce4014d",
            2504710,
        ),
        "raster",
        "Cloud-Optimized GeoTIFF variant of the nir band (B08).",
        _S2_ATTRIBUTION,
    ),
    "copernicus_dem": SampleFile(
        Asset(
            "DEM.tif",
            "4aa03df4de877570d728ace4b0c07e71cb8be79a0c0640c6de602116ad1b3f88",
            3413595,
        ),
        "raster",
        "Copernicus GLO-30 DEM warped onto the Sentinel-2 grid, float32 metres.",
        _DEM_ATTRIBUTION,
    ),
    "copernicus_dem_cog": SampleFile(
        Asset(
            "DEM_COG.tif",
            "c1d763c4dcbec033695ab0d73c75fdbc58d5a2c0d91eb1fe443c13d48644ee27",
            5385324,
        ),
        "raster",
        "Cloud-Optimized GeoTIFF variant of the Copernicus GLO-30 DEM.",
        _DEM_ATTRIBUTION,
    ),
    "boundary": SampleFile(
        Asset(
            "roi.gpkg",
            "788a1800caed8d03ef7893de1a61ead281edd901ea049cdf969c5749c0af0744",
            98304,
        ),
        "vector",
        "Region-of-interest polygon (GeoPackage, EPSG:4326) inside the Sentinel-2 footprint.",
        _S2_ATTRIBUTION,
    ),
}
