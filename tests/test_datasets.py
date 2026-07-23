"""Tests for the sample-dataset fetch/cache helpers (:mod:`eeo.datasets`).

Every test in the default run is offline: downloads are redirected to
locally-built synthetic files via a patched ``_download``. One real-network
integration test is marked ``network`` and skipped unless ``--run-network``.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

import eeo
from eeo.datasets import _cache, _registry
from eeo.datasets._cache import DatasetError, cache_dir, ensure_asset
from eeo.datasets._registry import Asset, Dataset

UTM = CRS.from_epsg(32633)
TRANSFORM = Affine.translation(500_000.0, 4_200_000.0) * Affine.scale(10.0, -10.0)


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_band(path: Path, value: int, name: str) -> None:
    """Write a tiny single-band GeoTIFF with a band description."""
    arr = np.full((4, 4), value, dtype="uint16")
    profile = dict(
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="uint16",
        crs=UTM,
        transform=TRANSFORM,
        nodata=0,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.descriptions = (name,)


@pytest.fixture
def cache_env(tmp_path, monkeypatch):
    """Point the cache at a temp dir for the duration of a test."""
    monkeypatch.setenv("EEO_DATA_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    return tmp_path


@pytest.fixture
def local_asset(tmp_path):
    """A source file on disk plus the matching Asset (checksum computed here)."""
    src = tmp_path / "source.bin"
    payload = b"easy-eo sample payload \x00\x01\x02" * 100
    src.write_bytes(payload)
    asset = Asset("thing.bin", _sha256_bytes(payload), len(payload))
    return src, asset


def _patch_download_from(monkeypatch, source: Path):
    """Make ``_download`` copy ``source`` instead of hitting the network."""
    calls = {"n": 0}

    def fake_download(url, dest):
        calls["n"] += 1
        dest.write_bytes(source.read_bytes())

    monkeypatch.setattr(_cache, "_download", fake_download)
    return calls


# --------------------------------------------------------------------------- #
# cache_dir resolution
# --------------------------------------------------------------------------- #
def test_cache_dir_prefers_eeo_data_dir(tmp_path, monkeypatch):
    target = tmp_path / "explicit"
    monkeypatch.setenv("EEO_DATA_DIR", str(target))
    assert cache_dir() == target
    assert target.is_dir()


def test_cache_dir_uses_xdg_when_no_override(tmp_path, monkeypatch):
    monkeypatch.delenv("EEO_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_dir() == tmp_path / "xdg" / "easy-eo"


def test_cache_dir_defaults_to_home_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("EEO_DATA_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert cache_dir() == tmp_path / ".cache" / "easy-eo"


# --------------------------------------------------------------------------- #
# ensure_asset: download, cache, verify
# --------------------------------------------------------------------------- #
def test_ensure_asset_downloads_and_verifies(cache_env, local_asset, monkeypatch):
    src, asset = local_asset
    _patch_download_from(monkeypatch, src)
    path = ensure_asset(asset)
    assert path.is_file()
    assert path.read_bytes() == src.read_bytes()


def test_ensure_asset_uses_cache_second_time(cache_env, local_asset, monkeypatch):
    src, asset = local_asset
    calls = _patch_download_from(monkeypatch, src)
    ensure_asset(asset)
    ensure_asset(asset)
    assert calls["n"] == 1  # second call served from cache, no re-download


def test_ensure_asset_redownloads_corrupt_cache(cache_env, local_asset, monkeypatch):
    src, asset = local_asset
    calls = _patch_download_from(monkeypatch, src)
    # Seed a corrupt cached file with the right name but wrong bytes.
    (cache_dir() / asset.remote).write_bytes(b"corrupt")
    path = ensure_asset(asset)
    assert calls["n"] == 1  # corruption detected -> downloaded
    assert path.read_bytes() == src.read_bytes()


def test_ensure_asset_checksum_mismatch_raises(cache_env, local_asset, monkeypatch):
    src, asset = local_asset
    bad = asset.__class__(asset.remote, "0" * 64, asset.nbytes)  # wrong expected hash
    _patch_download_from(monkeypatch, src)
    with pytest.raises(DatasetError, match="checksum mismatch"):
        ensure_asset(bad)
    assert not (cache_dir() / asset.remote).exists()  # corrupt download removed


def test_download_writes_and_replaces_atomically(cache_env, monkeypatch):
    """The real ``_download`` streams the response to dest via a temp file.

    Exercises the success path (copy + atomic replace) without a network call
    by patching ``urlopen`` to return an in-memory response, and confirms no
    stray ``.part`` temp file is left behind.
    """
    import io

    payload = b"streamed sample bytes \x00\x01" * 200
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(payload))

    dest = cache_dir() / "downloaded.bin"
    _cache._download("https://example.invalid/downloaded.bin", dest)

    assert dest.read_bytes() == payload
    assert not list(dest.parent.glob("*.part"))  # temp file renamed


def test_download_network_error_raises_dataset_error(cache_env, local_asset, monkeypatch):
    import urllib.error

    _, asset = local_asset

    def boom(url, headers=None):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: boom(None))
    with pytest.raises(DatasetError, match="failed to download"):
        ensure_asset(asset)


# --------------------------------------------------------------------------- #
# fetch / load / info via a synthetic registry
# --------------------------------------------------------------------------- #
@pytest.fixture
def synthetic_registry(cache_env, tmp_path, monkeypatch):
    """Replace the registry with local synthetic rasters + a vector stand-in."""
    # Build two single-band rasters and a "vector" blob on disk.
    red = tmp_path / "red_src.tif"
    nir = tmp_path / "nir_src.tif"
    _write_band(red, value=1000, name="red")
    _write_band(nir, value=4000, name="nir")
    vec = tmp_path / "roi_src.gpkg"
    vec.write_bytes(b"fake-geopackage-bytes")

    sources = {"red.tif": red, "nir.tif": nir, "roi.gpkg": vec}

    def sha(p: Path) -> str:
        return _sha256_bytes(p.read_bytes())

    ts = datetime(2023, 9, 7, 10, 0, 31, tzinfo=timezone.utc)
    registry = {
        "s2": Dataset(
            name="s2",
            kind="raster",
            assets=[
                Asset("red.tif", sha(red), red.stat().st_size, "red"),
                Asset("nir.tif", sha(nir), nir.stat().st_size, "nir"),
            ],
            description="synthetic two-band",
            attribution="test attribution",
            timestamp=ts,
            nodata=0,
            band_names=["red", "nir"],
        ),
        "single": Dataset(
            name="single",
            kind="raster",
            assets=[Asset("red.tif", sha(red), red.stat().st_size, "red")],
            description="synthetic one-band",
            attribution="test attribution",
            timestamp=ts,
            band_names=["red"],
        ),
        "roi": Dataset(
            name="roi",
            kind="vector",
            assets=[Asset("roi.gpkg", sha(vec), vec.stat().st_size)],
            description="synthetic vector",
            attribution="test attribution",
        ),
    }
    monkeypatch.setattr(_registry, "_DATASETS", registry)

    def fake_download(url, dest):
        dest.write_bytes(sources[dest.name].read_bytes())

    monkeypatch.setattr(_cache, "_download", fake_download)
    return registry, ts


def test_fetch_single_returns_path(synthetic_registry):
    path = eeo.datasets.fetch("single")
    assert isinstance(path, Path)
    assert path.name == "red.tif"


def test_fetch_multi_returns_list_in_band_order(synthetic_registry):
    paths = eeo.datasets.fetch("s2")
    assert isinstance(paths, list)
    assert [p.name for p in paths] == ["red.tif", "nir.tif"]


def test_load_stack_sets_names_timestamp_and_stacks(synthetic_registry):
    _, ts = synthetic_registry
    ds = eeo.datasets.load("s2")
    assert ds.get_count() == 2
    assert ds.band_names == ["red", "nir"]
    assert ds.timestamp == ts
    arr = ds.to_array()
    assert arr.shape == (2, 4, 4)
    assert arr[0].min() == 1000 and arr[1].min() == 4000  # band order preserved


def test_load_single_raster_is_lazy_with_names(synthetic_registry):
    ds = eeo.datasets.load("single")
    assert ds.get_count() == 1
    assert ds.band_names == ["red"]


def test_load_vector_returns_path(synthetic_registry):
    result = eeo.datasets.load("roi")
    assert isinstance(result, Path)
    assert result.name == "roi.gpkg"


def test_load_stack_ndvi_by_name(synthetic_registry):
    ds = eeo.datasets.load("s2")
    ndvi = ds.ndvi(red="red", nir="nir")
    a = ndvi.to_array()
    # (4000 - 1000) / (4000 + 1000) = 0.6 everywhere
    assert np.allclose(a, 0.6, atol=1e-4)
    assert a.dtype == np.float32


def test_info_includes_attribution(synthetic_registry):
    text = eeo.datasets.info("s2")
    assert "test attribution" in text
    assert "red.tif" in text and "nir.tif" in text


# --------------------------------------------------------------------------- #
# Error handling and registry integrity (no network, real registry)
# --------------------------------------------------------------------------- #
def test_unknown_dataset_lists_available():
    with pytest.raises(KeyError, match="unknown dataset"):
        eeo.datasets.fetch("does_not_exist")


def test_available_matches_registry():
    names = eeo.datasets.available()
    assert names == sorted(names)
    assert "sentinel2_small" in names
    assert set(names) == set(_registry._DATASETS)


def test_real_registry_checksums_are_wellformed():
    for name in eeo.datasets.available():
        ds = _registry.get_dataset(name)
        assert ds.assets, name
        for asset in ds.assets:
            assert len(asset.sha256) == 64
            assert all(c in "0123456789abcdef" for c in asset.sha256)
            assert asset.nbytes > 0


def test_sentinel2_small_is_single_stacked_raster():
    ds = _registry.get_dataset("sentinel2_small")
    assert ds.kind == "raster"
    assert not ds.multi  # a single pre-stacked 4-band file
    assert len(ds.assets) == 1
    assert ds.band_names == ["blue", "green", "red", "nir"]


def test_sentinel2_small_bands_is_multi_in_band_order():
    ds = _registry.get_dataset("sentinel2_small_bands")
    assert ds.multi and ds.kind == "raster"
    assert [a.band_name for a in ds.assets] == ["blue", "green", "red", "nir"]


# --------------------------------------------------------------------------- #
# Opt-in real-download integration test
# --------------------------------------------------------------------------- #
@pytest.mark.network
def test_real_download_roundtrip(tmp_path, monkeypatch):
    """Download the smallest real asset and verify checksum + open it."""
    monkeypatch.setenv("EEO_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    path = eeo.datasets.fetch("sentinel2_small_boundary")
    assert path.is_file()
    ds = eeo.datasets.load("sentinel2_small")
    assert ds.band_names == ["blue", "green", "red", "nir"]
    assert ds.get_count() == 4
