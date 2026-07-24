"""Tests for the sample-dataset helpers (:mod:`eeo.datasets`).

Every test in the default run is offline: downloads are redirected to
locally-built synthetic files via a patched ``_download`` (or a stubbed
``ensure_asset``). One real-network integration test is marked ``network`` and
skipped unless ``--run-network``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

import eeo
from eeo.datasets import _cache, _registry, _samples
from eeo.datasets._cache import DatasetError, cache_dir, ensure_asset
from eeo.datasets._registry import Asset, SampleFile
from eeo.datasets._samples import SampleDataset, SamplePath, load_sample_dataset

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
# load_sample_dataset: attribute-addressable, lazy access
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_ensure(monkeypatch, tmp_path):
    """Stub ``ensure_asset`` with a real synthetic GeoTIFF per asset name.

    Returns the list of assets ``ensure_asset`` was called with, so tests can
    assert exactly which (and how many) files were fetched.
    """
    calls: list = []

    def _ensure(asset):
        calls.append(asset)
        dest = tmp_path / asset.remote
        if not dest.exists():
            _write_band(dest, value=1234, name="blue")
        return dest

    monkeypatch.setattr(_samples._cache, "ensure_asset", _ensure)
    return calls


def test_load_sample_dataset_exposes_all_names():
    sd = load_sample_dataset()
    assert isinstance(sd, SampleDataset)
    for name in _registry.SAMPLE_FILES:
        assert isinstance(getattr(sd, name), SamplePath)


def test_declared_attrs_match_registry():
    # The explicit class annotations (for autocomplete/type-checkers) must not
    # drift from the actual SAMPLE_FILES source of truth.
    assert set(SampleDataset.__annotations__) == set(_registry.SAMPLE_FILES)


def test_construction_does_not_fetch(fake_ensure):
    load_sample_dataset()
    assert fake_ensure == []  # no network on construction


def test_attribute_access_does_not_fetch(fake_ensure):
    sd = load_sample_dataset()
    _ = sd.copernicus_dem  # holding the handle must not download
    assert fake_ensure == []


def test_fspath_triggers_fetch_once(fake_ensure):
    sd = load_sample_dataset()
    p1 = os.fspath(sd.sentinel2_blue)
    p2 = os.fspath(sd.sentinel2_blue)  # memoized: no second fetch
    assert p1 == p2
    assert len(fake_ensure) == 1
    assert fake_ensure[0] is _registry.SAMPLE_FILES["sentinel2_blue"].asset


def test_prefetch_downloads_everything(fake_ensure):
    load_sample_dataset(prefetch=True)
    assert len(fake_ensure) == len(_registry.SAMPLE_FILES)


def test_load_raster_opens_sample_path(fake_ensure):
    sd = load_sample_dataset()
    ds = eeo.load_raster(sd.sentinel2_blue)
    assert ds.get_count() == 1
    assert ds.to_array().min() == 1234
    assert len(fake_ensure) == 1  # opened exactly one file


def test_path_property_fetches(fake_ensure):
    sd = load_sample_dataset()
    p = sd.copernicus_dem.path
    assert isinstance(p, Path)
    assert p.is_file()
    assert len(fake_ensure) == 1


def test_sample_path_str_is_side_effect_free(fake_ensure):
    sd = load_sample_dataset()
    # Before fetch: shows the target filename, no download.
    assert str(sd.copernicus_dem) == _registry.SAMPLE_FILES["copernicus_dem"].asset.remote
    assert fake_ensure == []
    # After fetch: shows the cached path.
    resolved = sd.copernicus_dem.fetch()
    assert str(sd.copernicus_dem) == str(resolved)


def test_sample_path_repr_and_name(fake_ensure):
    sd = load_sample_dataset()
    assert sd.copernicus_dem.name == "copernicus_dem"
    assert "not fetched" in repr(sd.copernicus_dem)
    sd.copernicus_dem.fetch()
    assert "cached" in repr(sd.copernicus_dem)
    assert "copernicus_dem" in repr(sd)


def test_info_and_attribution_travel_with_handle():
    sd = load_sample_dataset()
    # Attribution is queryable from code without touching the network.
    assert "Copernicus DEM" in sd.copernicus_dem.attribution
    assert "Sentinel-2" in sd.sentinel2_blue.attribution
    text = sd.sentinel2_blue.info()
    assert "B02.tif" in text
    assert "raster" in text
    assert sd.sentinel2_blue.attribution in text
    assert sd.sentinel2_blue.description in text
    assert "blue" in sd.sentinel2_blue.description.lower()


def test_boundary_is_vector():
    sd = load_sample_dataset()
    assert sd.boundary.kind == "vector"
    assert all(getattr(sd, n).kind == "raster" for n in _registry.SAMPLE_FILES if n != "boundary")


def test_namespace_iterates_all_handles():
    sd = load_sample_dataset()
    assert len(sd) == len(_registry.SAMPLE_FILES)
    assert {h.name for h in sd} == set(_registry.SAMPLE_FILES)
    assert all(isinstance(h, SamplePath) for h in sd)


def test_sample_files_are_single_pinned_assets():
    for name, sample in _registry.SAMPLE_FILES.items():
        assert isinstance(sample, SampleFile), name
        assert sample.kind in {"raster", "vector"}
        assert len(sample.asset.sha256) == 64
        assert all(c in "0123456789abcdef" for c in sample.asset.sha256)
        assert sample.asset.nbytes > 0


def test_string_key_api_is_not_public():
    # Data is reachable only through the namespace, never by hard-coded string.
    for removed in ("load", "fetch", "info", "available"):
        assert not hasattr(eeo.datasets, removed), removed


# --------------------------------------------------------------------------- #
# Opt-in real-download integration test
# --------------------------------------------------------------------------- #
@pytest.mark.network
def test_real_download_roundtrip(tmp_path, monkeypatch):
    """Download real assets through the namespace and verify checksum + open."""
    monkeypatch.setenv("EEO_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    sd = load_sample_dataset()
    assert sd.boundary.fetch().is_file()  # smallest asset, checksum-verified
    ds = eeo.load_raster(sd.sentinel2_stacked)
    assert ds.get_count() == 4
    assert ds.band_names == ["blue", "green", "red", "nir"]
