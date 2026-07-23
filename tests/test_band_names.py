"""Band-name storage, seeding, mutation, resolution, and propagation."""

import numpy as np
import pytest
import rasterio as rio
from affine import Affine
from rasterio.crs import CRS
from rasterio.io import MemoryFile

from eeo import load_array, load_raster
from eeo.common import resolve_band_index
from eeo.core.core import EEORasterDataset
from eeo.core.exceptions import ValidationError
from eeo.viz.plot import _band_label, _normalize_bands

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


# ---------------------------------------------------------------------------
# Loader entry points
# ---------------------------------------------------------------------------
def test_load_array_accepts_band_names():
    ds = load_array(
        np.zeros((3, 4, 4), dtype=np.float32),
        transform=TRANSFORM,
        crs=CRS_4326,
        band_names=["red", "green", "blue"],
    )
    assert ds.band_names == ["red", "green", "blue"]


def test_load_array_wrong_length_band_names_raises():
    with pytest.raises(ValidationError):
        load_array(
            np.zeros((3, 4, 4), dtype=np.float32),
            transform=TRANSFORM,
            crs=CRS_4326,
            band_names=["red"],
        )


def test_load_raster_band_names_override_file_descriptions(tmp_path):
    path = tmp_path / "scene.tif"
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=2,
        dtype="float32",
        crs=CRS_4326,
        transform=TRANSFORM,
    ) as dst:
        dst.write(np.zeros((2, 4, 4), dtype=np.float32))
        dst.set_band_description(1, "from_file")
        dst.set_band_description(2, "also_from_file")

    # no override -> names come from the file's GDAL descriptions
    assert load_raster(str(path)).band_names == ["from_file", "also_from_file"]
    # explicit names win
    ds = load_raster(str(path), band_names=["red", "nir"])
    assert ds.band_names == ["red", "nir"]


# ---------------------------------------------------------------------------
# Name -> index resolution
# ---------------------------------------------------------------------------
def test_resolve_int_index_passes_through():
    ds = _numpy_ds(count=3)
    assert resolve_band_index(ds, 1) == 1
    assert resolve_band_index(ds, 3) == 3


@pytest.mark.parametrize("bad", [0, 4, -1])
def test_resolve_int_index_out_of_range_raises_index_error(bad):
    ds = _numpy_ds(count=3)
    with pytest.raises(IndexError):
        resolve_band_index(ds, bad)


def test_resolve_name_to_index():
    ds = _numpy_ds(count=3, band_names=["blue", "green", "red"])
    assert resolve_band_index(ds, "blue") == 1
    assert resolve_band_index(ds, "red") == 3


@pytest.mark.parametrize("spelling", ["RED", "red", "  Red  ", "rEd"])
def test_resolve_name_is_case_and_whitespace_insensitive(spelling):
    ds = _numpy_ds(count=2, band_names=["nir", "Red"])
    assert resolve_band_index(ds, spelling) == 2


def test_resolve_missing_name_raises_with_available_names():
    ds = _numpy_ds(count=2, band_names=["nir", "red"])
    with pytest.raises(ValidationError) as excinfo:
        resolve_band_index(ds, "swir")
    message = str(excinfo.value)
    assert "swir" in message
    assert "'nir'" in message and "'red'" in message


def test_resolve_name_on_unnamed_dataset_reports_none_available():
    ds = _numpy_ds(count=2)
    with pytest.raises(ValidationError) as excinfo:
        resolve_band_index(ds, "red")
    assert "none" in str(excinfo.value)


def test_resolve_ambiguous_name_raises():
    ds = _numpy_ds(count=3, band_names=["red", "red", "nir"])
    with pytest.raises(ValidationError) as excinfo:
        resolve_band_index(ds, "red")
    assert "ambiguous" in str(excinfo.value)


def test_numeric_string_is_a_name_never_an_index():
    # "4" must not resolve to band 4; it only matches a band named "4".
    ds = _numpy_ds(count=4)
    with pytest.raises(ValidationError):
        resolve_band_index(ds, "4")

    named = _numpy_ds(count=4, band_names=["4", "b", "c", "d"])
    assert resolve_band_index(named, "4") == 1


def test_bool_is_not_a_valid_band_index():
    ds = _numpy_ds(count=3)
    with pytest.raises(ValidationError):
        resolve_band_index(ds, True)


def test_resolve_rejects_other_types():
    ds = _numpy_ds(count=3)
    with pytest.raises(ValidationError):
        resolve_band_index(ds, 2.5)


# ---------------------------------------------------------------------------
# Names are accepted wherever a band index is (21.4)
# ---------------------------------------------------------------------------
def _named(ds, names):
    """Name ``ds``'s bands in place and return it, for terse test setup."""
    ds.band_names = names
    return ds


def test_get_band_accepts_a_name(multiband_uint16):
    ds = _named(multiband_uint16, ["blue", "green", "red", "nir"])
    np.testing.assert_array_equal(ds.get_band("red"), ds.get_band(3))


def test_get_band_rejects_an_unknown_name(multiband_uint16):
    with pytest.raises(ValidationError):
        multiband_uint16.get_band("red")


@pytest.mark.parametrize("op", ["get_maximum_pixel", "get_minimum_pixel", "get_mean_pixel"])
def test_stats_ops_accept_a_band_name(multiband_uint16, op):
    ds = _named(multiband_uint16, ["blue", "green", "red", "nir"])
    assert getattr(ds, op)(band_idx="red") == getattr(ds, op)(band_idx=3)


def test_get_percentile_pixel_accepts_a_band_name(multiband_uint16):
    ds = _named(multiband_uint16, ["blue", "green", "red", "nir"])
    assert ds.get_percentile_pixel(90, band_idx="nir") == ds.get_percentile_pixel(90, band_idx=4)


def test_extract_value_at_coordinate_accepts_a_band_name(raster_3x3):
    ds = _named(raster_3x3, ["elevation"])
    assert ds.extract_value_at_coordinate((1.5, 1.5), band_idx="elevation") == (
        ds.extract_value_at_coordinate((1.5, 1.5), band_idx=1)
    )


def test_index_band_specs_accept_names(multiband_uint16):
    ds = _named(multiband_uint16, ["blue", "green", "red", "nir"])
    by_name = ds.ndvi(red="red", nir="nir", return_as_ndarray=True)
    by_index = ds.ndvi(red=3, nir=4, return_as_ndarray=True)
    np.testing.assert_array_equal(by_name, by_index)


def test_index_band_spec_rejects_an_unknown_name(multiband_uint16):
    ds = _named(multiband_uint16, ["blue", "green", "red", "nir"])
    with pytest.raises(ValidationError):
        ds.ndvi(red="swir", nir="nir")


def test_index_band_spec_rejects_a_bad_type(multiband_uint16):
    with pytest.raises(ValidationError):
        multiband_uint16.ndvi(red=2.5)


def test_plot_band_selection_mixes_indices_and_names(multiband_uint16):
    ds = _named(multiband_uint16, ["blue", "green", "red", "nir"])
    assert _normalize_bands(ds, ["red", 2, "blue"]) == [3, 2, 1]
    assert _normalize_bands(ds, "nir") == [4]
    assert _normalize_bands(ds, None) == [1, 2, 3, 4]


def test_plot_composite_accepts_band_names(multiband_uint16):
    ds = _named(multiband_uint16, ["blue", "green", "red", "nir"])
    # Smoke test under the Agg backend: resolving the names must not raise.
    ds.plot_composite(bands=["red", "green", "blue"])


def test_plot_composite_rejects_a_selection_that_is_not_three_bands(multiband_uint16):
    ds = _named(multiband_uint16, ["blue", "green", "red", "nir"])
    with pytest.raises(ValidationError):
        ds.plot_composite(bands=["red", "green"])


def test_band_label_appends_the_name(multiband_uint16):
    ds = _named(multiband_uint16, ["blue", None, "red", "nir"])
    assert _band_label(ds, 1) == "Band 1 (blue)"
    assert _band_label(ds, 2) == "Band 2"


# ---------------------------------------------------------------------------
# Propagation through operations (21.5)
# ---------------------------------------------------------------------------
def test_to_rasterio_carries_band_names():
    ds = _numpy_ds(count=2, band_names=["red", "nir"])
    assert ds.to_rasterio().band_names == ["red", "nir"]


def test_scalar_algebra_preserves_band_names(multiband_uint16):
    ds = _named(multiband_uint16, ["blue", "green", "red", "nir"])
    assert ds.add(1).band_names == ["blue", "green", "red", "nir"]
    assert ds.multiply(2).sqrt().band_names == ["blue", "green", "red", "nir"]


def test_resample_preserves_band_names(multiband_uint16):
    ds = _named(multiband_uint16, ["blue", "green", "red", "nir"])
    assert ds.resample(scale_factor=0.5).band_names == ["blue", "green", "red", "nir"]


def test_normalize_preserves_band_names(multiband_uint16):
    ds = _named(multiband_uint16, ["blue", "green", "red", "nir"])
    assert ds.normalize_min_max().band_names == ["blue", "green", "red", "nir"]


def test_clip_preserves_band_names(multiband_uint16):
    ds = _named(multiband_uint16, ["blue", "green", "red", "nir"])
    left, bottom, right, top = ds.get_bounds()
    clipped = ds.clip_raster_with_bbox((left, bottom, left + 30, bottom + 30))
    assert clipped.band_names == ["blue", "green", "red", "nir"]


def test_unnamed_input_stays_unnamed_through_ops(multiband_uint16):
    assert multiband_uint16.add(1).band_names == [None] * 4


def test_index_output_is_unnamed_by_default(multiband_uint16):
    ds = _named(multiband_uint16, ["blue", "green", "red", "nir"])
    assert ds.ndvi(red="red", nir="nir").band_names == [None]


def test_index_output_honours_an_explicit_name(multiband_uint16):
    ds = _named(multiband_uint16, ["blue", "green", "red", "nir"])
    assert ds.ndvi(red="red", nir="nir", name="ndvi_2024").band_names == ["ndvi_2024"]


def test_single_band_index_input_is_not_inherited(single_band_float32):
    # A single-band NIR raster named "nir": the NDVI band is a new measurement,
    # so it must not silently inherit the receiver's name.
    nir = _named(single_band_float32, ["nir"])
    red = load_array(
        np.ones((6, 6), dtype=np.float32),
        transform=nir.get_transform(),
        crs=nir.get_crs(),
    ).to_rasterio()
    assert nir.ndvi(red).band_names == [None]


def test_normalized_difference_naming(single_band_float32):
    ds = _named(single_band_float32, ["nir"])
    other = load_array(
        np.ones((6, 6), dtype=np.float32),
        transform=ds.get_transform(),
        crs=ds.get_crs(),
    ).to_rasterio()
    assert ds.normalized_difference(other).band_names == [None]
    assert ds.normalized_difference(other, name="nd").band_names == ["nd"]


def test_normalized_difference_name_rejected_for_multiband_result(multiband_uint16):
    other = multiband_uint16.add(1)
    with pytest.raises(ValidationError):
        multiband_uint16.normalized_difference(other, name="nd")


def test_stack_concatenates_input_names(single_band_float32):
    red = _named(single_band_float32, ["red"])
    green = _named(
        load_array(
            np.ones((6, 6), dtype=np.float32),
            transform=red.get_transform(),
            crs=red.get_crs(),
        ).to_rasterio(),
        ["green"],
    )
    unnamed = load_array(
        np.zeros((6, 6), dtype=np.float32),
        transform=red.get_transform(),
        crs=red.get_crs(),
    ).to_rasterio()

    assert red.stack([green, unnamed]).band_names == ["red", "green", None]
    assert red.stack([green, unnamed], names=["a", "b", "c"]).band_names == ["a", "b", "c"]
    with pytest.raises(ValidationError):
        red.stack([green], names=["only-one"])


def test_mosaic_keeps_the_primary_names(single_band_float32):
    ds = _named(single_band_float32, ["red"])
    assert ds.mosaic(ds).band_names == ["red"]
    assert ds.mosaic(ds, names=["mosaicked"]).band_names == ["mosaicked"]
