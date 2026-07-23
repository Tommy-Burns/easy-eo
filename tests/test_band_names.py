"""Band-name storage, seeding, and mutation on EEORasterDataset"""

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS
from rasterio.io import MemoryFile

from eeo import load_array
from eeo.core.core import EEORasterDataset
from eeo.core.exceptions import ValidationError

CRS_4326 = CRS.from_epsg(4326)
TRANSFORM = Affine.translation(0, 4) * Affine.scale(1, -1)


def _numpy_ds(count=3, **kwargs):
    """Build a small NumPy-backed dataset with ``count`` bands."""
    array = np.zeros((count, 4, 4), dtype=np.float32)
    return load_array(array, transform=TRANSFORM, crs=CRS_4326, **kwargs)


def _rasterio_ds_with_descriptions(descriptions):
    """Build an in-memory rasterio dataset with the given GDAL band descriptions."""
    count = len(descriptions)
    mf = MemoryFile()
    dst = mf.open(
        driver="GTiff",
        height=4,
        width=4,
        count=count,
        dtype="float32",
        crs=CRS_4326,
        transform=TRANSFORM,
    )
    dst.write(np.zeros((count, 4, 4), dtype=np.float32))
    for i, name in enumerate(descriptions, start=1):
        if name is not None:
            dst.set_band_description(i, name)
    return EEORasterDataset.from_rasterio(dst)


# ---------------------------------------------------------------------------
# Seeding / default state
# ---------------------------------------------------------------------------
def test_numpy_backend_defaults_to_all_none():
    ds = _numpy_ds(count=3)
    assert ds.band_names == [None, None, None]


def test_seeded_from_gdal_descriptions():
    ds = _rasterio_ds_with_descriptions(["blue", None, "swir"])
    assert ds.band_names == ["blue", None, "swir"]


def test_blank_gdal_description_is_none():
    # GDAL sometimes stores "" rather than None for an unnamed band.
    ds = _rasterio_ds_with_descriptions(["red", "", "nir"])
    assert ds.band_names == ["red", None, "nir"]


def test_explicit_band_names_at_construction():
    ds = EEORasterDataset.from_array(
        np.zeros((2, 4, 4), dtype=np.float32),
        transform=TRANSFORM,
        crs=CRS_4326,
        band_names=["  Red ", ""],
    )
    # whitespace stripped; blank normalized to None
    assert ds.band_names == ["Red", None]


# ---------------------------------------------------------------------------
# Bulk assignment via the property
# ---------------------------------------------------------------------------
def test_setter_replaces_all_names():
    ds = _numpy_ds(count=3)
    ds.band_names = ["red", "green", "blue"]
    assert ds.band_names == ["red", "green", "blue"]


def test_setter_none_clears_all_names():
    ds = _numpy_ds(count=3)
    ds.band_names = ["a", "b", "c"]
    ds.band_names = None
    assert ds.band_names == [None, None, None]


def test_setter_normalizes_blanks_and_whitespace():
    ds = _numpy_ds(count=2)
    ds.band_names = [" nir ", "   "]
    assert ds.band_names == ["nir", None]


def test_setter_wrong_length_raises():
    ds = _numpy_ds(count=3)
    with pytest.raises(ValidationError):
        ds.band_names = ["only-one"]


def test_setter_rejects_non_string_entry():
    ds = _numpy_ds(count=2)
    with pytest.raises(ValidationError):
        ds.band_names = ["red", 5]


# ---------------------------------------------------------------------------
# Single-band rename
# ---------------------------------------------------------------------------
def test_set_band_name_renames_one_band():
    ds = _numpy_ds(count=3)
    ds.set_band_name(2, "nir")
    assert ds.band_names == [None, "nir", None]


def test_set_band_name_clears_with_none_or_blank():
    ds = _numpy_ds(count=2)
    ds.band_names = ["red", "nir"]
    ds.set_band_name(1, None)
    ds.set_band_name(2, "   ")
    assert ds.band_names == [None, None]


@pytest.mark.parametrize("bad", [0, 4, -1])
def test_set_band_name_out_of_range_raises_index_error(bad):
    ds = _numpy_ds(count=3)
    with pytest.raises(IndexError):
        ds.set_band_name(bad, "red")


def test_set_band_name_rejects_non_string():
    ds = _numpy_ds(count=2)
    with pytest.raises(ValidationError):
        ds.set_band_name(1, 5)


# ---------------------------------------------------------------------------
# Getter is defensive (a copy, not the live list)
# ---------------------------------------------------------------------------
def test_getter_returns_copy_not_live_list():
    ds = _numpy_ds(count=3)
    ds.band_names = ["red", "green", "blue"]
    snapshot = ds.band_names
    snapshot[0] = "mutated"
    assert ds.band_names[0] == "red"


# ---------------------------------------------------------------------------
# Length invariant holds across the two backends
# ---------------------------------------------------------------------------
def test_band_names_length_matches_band_count():
    assert len(_numpy_ds(count=1).band_names) == 1
    assert len(_numpy_ds(count=4).band_names) == 4
    assert len(_rasterio_ds_with_descriptions([None] * 5).band_names) == 5
