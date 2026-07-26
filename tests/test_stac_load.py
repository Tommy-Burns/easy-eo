"""Tests for STACItem.load (eeo/io/stac.py).

The STAC metadata is faked, but the raster reads are real: assets point at
GeoTIFFs written to a temporary directory, so the cropping, grid alignment, and
warping code runs exactly as it would against a remote COG — without touching
the network.
"""

import datetime as dt

import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

import eeo

UTC = dt.timezone.utc

# A synthetic scene in UTM 33N: 200 x 200 pixels of 10 m, so 2 km on a side.
SCENE_CRS = "EPSG:32633"
SCENE_ORIGIN = (500000.0, 5000000.0)
SCENE_SIZE = 200
SCENE_RES = 10.0
TIMESTAMP = dt.datetime(2023, 6, 5, 10, 6, 21, tzinfo=UTC)


def write_asset(path, *, size=SCENE_SIZE, res=SCENE_RES, nodata=0, fill=None, count=1):
    """Write a synthetic single- or multi-band GeoTIFF and return its path."""
    transform = from_origin(SCENE_ORIGIN[0], SCENE_ORIGIN[1], res, res)
    if fill is None:
        band = np.arange(size * size, dtype="uint16").reshape(size, size)
        data = np.stack([band + offset for offset in range(count)])
    else:
        data = np.full((count, size, size), fill, dtype="uint16")

    profile = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": count,
        "dtype": "uint16",
        "crs": SCENE_CRS,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def utm_to_wgs84(minx, miny, maxx, maxy):
    """Convert scene-CRS bounds to the WGS 84 lon/lat bbox the API expects."""
    return transform_bounds(SCENE_CRS, "EPSG:4326", minx, miny, maxx, maxy, densify_pts=21)


class FakeAsset:
    def __init__(self, href):
        self.href = str(href)


class FakeItem:
    """Minimal stand-in for a pystac Item."""

    def __init__(self, assets, *, timestamp=TIMESTAMP):
        self.id = "S2A_TEST"
        self.datetime = timestamp
        self.collection_id = "sentinel-2-l2a"
        self.properties = {"eo:cloud_cover": 4.2}
        self.assets = {name: FakeAsset(href) for name, href in assets.items()}
        self.bbox = list(utm_to_wgs84(*scene_bounds()))


def scene_bounds():
    """Full extent of the synthetic scene in its own CRS."""
    left, top = SCENE_ORIGIN
    span = SCENE_SIZE * SCENE_RES
    return (left, top - span, left + span, top)


@pytest.fixture
def scene(tmp_path):
    """A one-band 10 m asset, plus a second 10 m band and a 20 m band."""
    return {
        "B04": write_asset(tmp_path / "B04.tif"),
        "B08": write_asset(tmp_path / "B08.tif", fill=1000),
        "B11": write_asset(tmp_path / "B11.tif", size=100, res=20.0, fill=500),
    }


@pytest.fixture
def aoi():
    """A 400 m box (40 pixels at 10 m) inset from the scene's top-left corner."""
    left, top = SCENE_ORIGIN
    return utm_to_wgs84(left + 200.0, top - 600.0, left + 600.0, top - 200.0)


def make_item(scene, *, search_bbox=None, assets=None, timestamp=TIMESTAMP):
    hrefs = scene if assets is None else {name: scene[name] for name in assets}
    return eeo.io.STACItem(FakeItem(hrefs, timestamp=timestamp), search_bbox=search_bbox)


# --------------------------------------------------------------------------
# Cropping
# --------------------------------------------------------------------------
def test_load_crops_to_the_search_aoi_by_default(scene, aoi):
    ds = make_item(scene, search_bbox=aoi).load("B04")

    # The AOI is 40x40 pixels of the 200x200 scene; projecting a lon/lat box
    # into UTM can add a pixel of margin, never a whole tile.
    height, width = ds.get_shape()
    assert 40 <= height <= 43
    assert 40 <= width <= 43
    assert ds.get_count() == 1


def test_cropped_window_covers_the_requested_area(scene, aoi):
    ds = make_item(scene, search_bbox=aoi).load("B04")

    left, bottom, right, top = ds.to_rasterio().ds.bounds
    want = transform_bounds("EPSG:4326", SCENE_CRS, *aoi, densify_pts=21)
    assert left <= want[0] and bottom <= want[1]
    assert right >= want[2] and top >= want[3]
    assert ds.get_crs().to_epsg() == 32633


def test_explicit_bbox_overrides_the_search_aoi(scene, aoi):
    left, top = SCENE_ORIGIN
    smaller = utm_to_wgs84(left + 200.0, top - 400.0, left + 400.0, top - 200.0)

    ds = make_item(scene, search_bbox=aoi).load("B04", bbox=smaller)

    height, width = ds.get_shape()
    assert 20 <= height <= 23
    assert 20 <= width <= 23


def test_crop_false_reads_the_whole_scene(scene, aoi):
    ds = make_item(scene, search_bbox=aoi).load("B04", crop=False)

    assert ds.get_shape() == (SCENE_SIZE, SCENE_SIZE)


def test_no_aoi_anywhere_reads_the_whole_scene(scene):
    ds = make_item(scene).load("B04")

    assert ds.get_shape() == (SCENE_SIZE, SCENE_SIZE)


def test_cropped_values_match_the_source_window(scene, aoi):
    ds = make_item(scene, search_bbox=aoi).load("B04")

    with rio.open(scene["B04"]) as src:
        window = rio.windows.from_bounds(
            *transform_bounds("EPSG:4326", SCENE_CRS, *aoi, densify_pts=21),
            transform=src.transform,
        )
        expected_corner = src.read(1)[int(window.row_off) + 1, int(window.col_off) + 1]

    assert ds.to_array()[0, 1, 1] == expected_corner


def test_aoi_outside_the_scene_raises_validation_error(scene):
    far_away = (0.0, 0.0, 0.1, 0.1)  # Gulf of Guinea; the scene is in the Alps

    with pytest.raises(eeo.ValidationError, match="does not overlap"):
        make_item(scene).load("B04", bbox=far_away)


def test_aoi_partly_outside_the_scene_is_clipped(scene):
    left, top = SCENE_ORIGIN
    overhanging = utm_to_wgs84(left - 1000.0, top - 400.0, left + 400.0, top + 1000.0)

    ds = make_item(scene).load("B04", bbox=overhanging)

    height, width = ds.get_shape()
    assert height <= SCENE_SIZE and width <= SCENE_SIZE
    assert height > 0 and width > 0


# --------------------------------------------------------------------------
# Stacking several assets
# --------------------------------------------------------------------------
def test_several_assets_stack_into_one_dataset(scene, aoi):
    ds = make_item(scene, search_bbox=aoi).load(["B04", "B08"])

    assert ds.get_count() == 2
    assert ds.band_names == ["B04", "B08"]
    values = ds.to_array()
    assert np.all(values[1] == 1000)  # B08 was filled with a constant


def test_coarser_asset_is_resampled_onto_the_first_grid(scene, aoi):
    ds = make_item(scene, search_bbox=aoi).load(["B04", "B11"])

    # B11 is 20 m; it must come back on B04's 10 m grid, not its own.
    assert ds.get_count() == 2
    assert ds.get_shape() == make_item(scene, search_bbox=aoi).load("B04").get_shape()
    assert np.all(ds.to_array()[1] == 500)


def test_first_asset_defines_the_grid(scene, aoi):
    coarse_first = make_item(scene, search_bbox=aoi).load(["B11", "B04"])
    fine_first = make_item(scene, search_bbox=aoi).load(["B04", "B11"])

    assert coarse_first.get_shape() != fine_first.get_shape()
    assert coarse_first.band_names == ["B11", "B04"]


def test_multiband_asset_names_its_bands_after_the_asset(tmp_path, aoi):
    href = write_asset(tmp_path / "visual.tif", count=3)
    item = eeo.io.STACItem(FakeItem({"visual": href}), search_bbox=aoi)

    ds = item.load("visual")

    assert ds.get_count() == 3
    assert ds.band_names == ["visual_1", "visual_2", "visual_3"]


def test_asset_in_another_crs_is_reprojected_onto_the_first_grid(tmp_path, scene, aoi):
    # Same ground, UTM zone 32 instead of 33 - a real possibility for assets
    # from different collections.
    left, bottom, right, top = transform_bounds(SCENE_CRS, "EPSG:32632", *scene_bounds())
    other = tmp_path / "zone32.tif"
    with rio.open(
        other,
        "w",
        driver="GTiff",
        height=200,
        width=200,
        count=1,
        dtype="uint16",
        crs="EPSG:32632",
        transform=from_origin(left, top, (right - left) / 200, (top - bottom) / 200),
        nodata=0,
    ) as dst:
        dst.write(np.full((1, 200, 200), 777, dtype="uint16"))

    item = eeo.io.STACItem(FakeItem({"B04": scene["B04"], "other": other}), search_bbox=aoi)
    ds = item.load(["B04", "other"])

    assert ds.get_crs().to_epsg() == 32633
    assert ds.get_count() == 2
    # The reprojected band lands on the first asset's grid, values intact.
    assert np.all(ds.to_array()[1] == 777)


def test_asset_covering_less_ground_is_filled_with_nodata(tmp_path, scene, aoi):
    # 300 m asset against a 2 km scene: the AOI only partly overlaps it.
    partial = write_asset(tmp_path / "partial.tif", size=30, fill=42)

    item = eeo.io.STACItem(FakeItem({"B04": scene["B04"], "partial": partial}), search_bbox=aoi)
    ds = item.load(["B04", "partial"])

    band = ds.to_array()[1]
    assert ds.get_shape() == make_item(scene, search_bbox=aoi).load("B04").get_shape()
    assert (band == 42).any()  # the overlapping part
    assert (band == 0).any()  # the rest, filled with the asset's nodata


def test_cropping_an_unreferenced_asset_raises_validation_error(tmp_path, aoi):
    plain = tmp_path / "no_crs.tif"
    with rio.open(
        plain,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="uint16",
        transform=from_origin(0, 10, 1, 1),  # placed, but in no known CRS
    ) as dst:
        dst.write(np.ones((1, 10, 10), dtype="uint16"))

    item = eeo.io.STACItem(FakeItem({"plain": plain}), search_bbox=aoi)

    with pytest.raises(eeo.ValidationError, match="no CRS"):
        item.load("plain")

    # ...but it still reads whole.
    assert item.load("plain", crop=False).get_shape() == (10, 10)


# --------------------------------------------------------------------------
# Metadata carried onto the dataset
# --------------------------------------------------------------------------
def test_load_carries_timestamp_nodata_and_provenance(scene, aoi):
    ds = make_item(scene, search_bbox=aoi).load(["B04", "B08"])

    assert ds.timestamp == TIMESTAMP
    assert ds.get_metadata()["nodata"] == 0
    assert ds.attrs["stac_item"] == "S2A_TEST"
    assert ds.attrs["stac_collection"] == "sentinel-2-l2a"
    assert ds.attrs["stac_assets"] == ["B04", "B08"]


def test_loaded_dataset_is_rasterio_backed_and_chainable(scene, aoi):
    ds = make_item(scene, search_bbox=aoi).load(["B04", "B08"])

    ndvi = ds.ndvi(red="B04", nir="B08")

    assert ndvi.get_count() == 1
    assert ndvi.to_array().dtype == np.float32
    # Provenance survives the chain (the timestamp is what a time series needs).
    assert ndvi.timestamp == TIMESTAMP


def test_dtype_is_the_assets_own(scene, aoi):
    ds = make_item(scene, search_bbox=aoi).load("B04")

    assert ds.to_array().dtype == np.dtype("uint16")


def test_undated_item_loads_without_a_timestamp(scene, aoi):
    ds = make_item(scene, search_bbox=aoi, timestamp=None).load("B04")

    assert ds.timestamp is None


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def test_unknown_asset_raises_validation_error_listing_the_available_ones(scene):
    with pytest.raises(eeo.ValidationError, match="no asset 'B99'"):
        make_item(scene).load("B99")

    with pytest.raises(eeo.ValidationError, match="B04"):
        make_item(scene).load("B99")


@pytest.mark.parametrize("bad", [[], (), 7])
def test_invalid_assets_argument_raises_validation_error(scene, bad):
    with pytest.raises(eeo.ValidationError, match="asset"):
        make_item(scene).load(bad)


def test_bbox_with_crop_false_raises_validation_error(scene, aoi):
    with pytest.raises(eeo.ValidationError, match="crop=False"):
        make_item(scene).load("B04", bbox=aoi, crop=False)


def test_invalid_bbox_raises_validation_error(scene):
    with pytest.raises(eeo.ValidationError, match="bbox"):
        make_item(scene).load("B04", bbox=(1.0, 2.0, 3.0))


def test_invalid_resampling_method_raises_validation_error(scene, aoi):
    with pytest.raises(eeo.ValidationError, match="resampling"):
        make_item(scene, search_bbox=aoi).load(["B04", "B11"], resampling="teleport")


# --------------------------------------------------------------------------
# Search integration
# --------------------------------------------------------------------------
def test_search_result_items_inherit_the_search_aoi(scene, aoi, monkeypatch):
    import sys
    import types

    class FakeSearch:
        def items(self):
            return iter([FakeItem(scene)])

    class FakeClient:
        @classmethod
        def open(cls, url, modifier=None):
            return cls()

        def search(self, **params):
            return FakeSearch()

    module = types.ModuleType("pystac_client")
    module.Client = FakeClient
    monkeypatch.setitem(sys.modules, "pystac_client", module)

    results = eeo.stac_search("sentinel-2-l2a", bbox=aoi, sign=False)

    assert results[0].search_bbox == pytest.approx(aoi)
    # ...and loading from the search result crops to it without repeating it.
    assert results[0].load("B04").get_shape() < (SCENE_SIZE, SCENE_SIZE)
