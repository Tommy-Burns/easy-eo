"""Tests for the ``intersects`` spatial filter on eeo.stac_search.

The catalog is faked (as in test_stac_search.py) and the assets are local
GeoTIFFs (as in test_stac_load.py), so geometry normalisation, the query sent
to the catalog, and geometry-aware cropping are all exercised offline.
"""

import sys
import types

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
import shapely
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

import eeo

SCENE_CRS = "EPSG:32633"
SCENE_ORIGIN = (500000.0, 5000000.0)
SCENE_SIZE = 200
SCENE_RES = 10.0


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class FakeAsset:
    def __init__(self, href):
        self.href = str(href)


class FakeItem:
    def __init__(self, assets=None):
        self.id = "S2A_TEST"
        self.datetime = None
        self.collection_id = "sentinel-2-l2a"
        self.properties = {}
        self.assets = {k: FakeAsset(v) for k, v in (assets or {}).items()}
        self.bbox = [11.0, 46.5, 11.2, 46.7]


class FakeSearch:
    def __init__(self, items):
        self._items = items

    def items(self):
        return iter(self._items)


class FakeClient:
    searched = []
    items_to_return = []

    @classmethod
    def open(cls, url, modifier=None):
        return cls()

    def search(self, **params):
        FakeClient.searched.append(params)
        return FakeSearch(list(FakeClient.items_to_return))


@pytest.fixture
def fake_stac(monkeypatch):
    FakeClient.searched = []
    FakeClient.items_to_return = []
    module = types.ModuleType("pystac_client")
    module.Client = FakeClient
    # The default catalog is Planetary Computer, so the signer is imported too;
    # fake it as well, or the real one would try to import the fake client.
    signer = types.ModuleType("planetary_computer")
    signer.sign_inplace = lambda item: item

    monkeypatch.setitem(sys.modules, "pystac_client", module)
    monkeypatch.setitem(sys.modules, "planetary_computer", signer)
    return FakeClient


# --------------------------------------------------------------------------
# Geometry fixtures
# --------------------------------------------------------------------------
def utm_box(minx, miny, maxx, maxy):
    """A rectangle in the scene's UTM CRS."""
    return shapely.geometry.box(minx, miny, maxx, maxy)


@pytest.fixture
def triangle_utm():
    """A non-rectangular AOI inside the scene, in UTM 33N."""
    left, top = SCENE_ORIGIN
    return shapely.geometry.Polygon(
        [(left + 200, top - 200), (left + 600, top - 200), (left + 200, top - 600)]
    )


@pytest.fixture
def triangle_wgs84(triangle_utm):
    """The same triangle in WGS 84 lon/lat."""
    gdf = gpd.GeoDataFrame(geometry=[triangle_utm], crs=SCENE_CRS).to_crs(4326)
    return gdf.geometry.iloc[0]


@pytest.fixture
def scene_asset(tmp_path):
    """A 200 x 200, 10 m GeoTIFF covering the scene, all pixels valued 500."""
    path = tmp_path / "B04.tif"
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=SCENE_SIZE,
        width=SCENE_SIZE,
        count=1,
        dtype="uint16",
        crs=SCENE_CRS,
        transform=from_origin(SCENE_ORIGIN[0], SCENE_ORIGIN[1], SCENE_RES, SCENE_RES),
        nodata=0,
    ) as dst:
        dst.write(np.full((1, SCENE_SIZE, SCENE_SIZE), 500, dtype="uint16"))
    return path


# --------------------------------------------------------------------------
# What reaches the catalog
# --------------------------------------------------------------------------
def test_shapely_geometry_is_sent_as_geojson(fake_stac, triangle_wgs84):
    eeo.stac_search("sentinel-2-l2a", intersects=triangle_wgs84)

    sent = fake_stac.searched[0]["intersects"]
    assert sent["type"] == "Polygon"
    assert "bbox" not in sent
    assert shapely.geometry.shape(sent).equals(triangle_wgs84)


def test_geodataframe_is_reprojected_to_lon_lat(fake_stac, triangle_utm, triangle_wgs84):
    projected = gpd.GeoDataFrame(geometry=[triangle_utm], crs=SCENE_CRS)

    eeo.stac_search("sentinel-2-l2a", intersects=projected)

    sent = shapely.geometry.shape(fake_stac.searched[0]["intersects"])
    # Sent in degrees, not metres - the CRS travelled with the data.
    assert sent.bounds == pytest.approx(triangle_wgs84.bounds, abs=1e-9)


def test_geoseries_is_accepted(fake_stac, triangle_utm):
    series = gpd.GeoSeries([triangle_utm], crs=SCENE_CRS)

    eeo.stac_search("sentinel-2-l2a", intersects=series)

    assert fake_stac.searched[0]["intersects"]["type"] == "Polygon"


def test_several_geometries_are_merged(fake_stac):
    left, right = utm_box(11.0, 46.5, 11.1, 46.6), utm_box(11.3, 46.5, 11.4, 46.6)
    gdf = gpd.GeoDataFrame(geometry=[left, right], crs="EPSG:4326")

    eeo.stac_search("sentinel-2-l2a", intersects=gdf)

    sent = shapely.geometry.shape(fake_stac.searched[0]["intersects"])
    assert sent.geom_type == "MultiPolygon"
    assert sent.bounds == pytest.approx((11.0, 46.5, 11.4, 46.6))


def test_geojson_mapping_feature_and_collection_are_accepted(fake_stac, triangle_wgs84):
    geometry = shapely.geometry.mapping(triangle_wgs84)
    feature = {"type": "Feature", "properties": {}, "geometry": geometry}
    collection = {"type": "FeatureCollection", "features": [feature]}

    for value in (geometry, feature, collection):
        eeo.stac_search("sentinel-2-l2a", intersects=value)

    for params in fake_stac.searched:
        assert shapely.geometry.shape(params["intersects"]).equals(triangle_wgs84)


def test_vector_file_path_is_read(fake_stac, tmp_path, triangle_utm, triangle_wgs84):
    path = tmp_path / "aoi.geojson"
    gpd.GeoDataFrame(geometry=[triangle_utm], crs=SCENE_CRS).to_file(path)

    eeo.stac_search("sentinel-2-l2a", intersects=str(path))

    sent = shapely.geometry.shape(fake_stac.searched[0]["intersects"])
    assert sent.bounds == pytest.approx(triangle_wgs84.bounds, abs=1e-6)


def test_bbox_is_still_sent_as_bbox(fake_stac):
    eeo.stac_search("sentinel-2-l2a", bbox=(11.0, 46.5, 11.2, 46.7))

    assert "intersects" not in fake_stac.searched[0]
    assert fake_stac.searched[0]["bbox"] == [11.0, 46.5, 11.2, 46.7]


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def test_bbox_and_intersects_together_raise_validation_error(fake_stac, triangle_wgs84):
    with pytest.raises(eeo.ValidationError, match="either bbox or intersects"):
        eeo.stac_search("sentinel-2-l2a", bbox=(11.0, 46.5, 11.2, 46.7), intersects=triangle_wgs84)

    assert fake_stac.searched == []


def test_projected_geometry_without_a_crs_is_rejected(fake_stac, triangle_utm):
    # Metres, no CRS to reproject from: the catalog would match nothing at all.
    with pytest.raises(eeo.ValidationError, match="WGS 84 lon/lat"):
        eeo.stac_search("sentinel-2-l2a", intersects=triangle_utm)


def test_out_of_range_latitudes_are_rejected(fake_stac):
    # Plausible longitudes, projected northings: caught on the latitude check.
    strip = shapely.geometry.box(10.0, 5_000_000.0, 11.0, 5_000_100.0)

    with pytest.raises(eeo.ValidationError, match="latitudes"):
        eeo.stac_search("sentinel-2-l2a", intersects=strip)


def test_geodataframe_without_a_crs_is_taken_as_lon_lat(fake_stac, triangle_wgs84):
    gdf = gpd.GeoDataFrame(geometry=[triangle_wgs84])  # no crs set
    assert gdf.crs is None

    eeo.stac_search("sentinel-2-l2a", intersects=gdf)

    assert fake_stac.searched[0]["intersects"]["type"] == "Polygon"


def test_empty_geometry_is_rejected(fake_stac):
    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    with pytest.raises(eeo.ValidationError, match="at least one geometry"):
        eeo.stac_search("sentinel-2-l2a", intersects=empty)


@pytest.mark.parametrize("bad", [42, object(), {"type": "Nonsense", "coordinates": []}])
def test_unusable_intersects_raises_validation_error(fake_stac, bad):
    with pytest.raises(eeo.ValidationError, match="intersects"):
        eeo.stac_search("sentinel-2-l2a", intersects=bad)


def test_unreadable_vector_path_raises_validation_error(fake_stac, tmp_path):
    with pytest.raises(eeo.ValidationError, match="vector file"):
        eeo.stac_search("sentinel-2-l2a", intersects=str(tmp_path / "missing.geojson"))


# --------------------------------------------------------------------------
# Retention on the result
# --------------------------------------------------------------------------
def test_result_and_items_retain_the_geometry(fake_stac, triangle_wgs84):
    fake_stac.items_to_return = [FakeItem()]

    results = eeo.stac_search("sentinel-2-l2a", intersects=triangle_wgs84)

    assert results.bbox is None
    assert shapely.geometry.shape(results.intersects).equals(triangle_wgs84)
    assert shapely.geometry.shape(results[0].search_intersects).equals(triangle_wgs84)
    assert results[0].search_bbox is None
    # ...and it survives slicing, like the bbox does.
    assert results[:1].intersects == results.intersects


def test_bbox_search_leaves_the_geometry_unset(fake_stac):
    fake_stac.items_to_return = [FakeItem()]

    results = eeo.stac_search("sentinel-2-l2a", bbox=(11.0, 46.5, 11.2, 46.7))

    assert results.intersects is None
    assert results[0].search_intersects is None


# --------------------------------------------------------------------------
# Loading against a geometry
# --------------------------------------------------------------------------
def test_load_crops_to_the_geometry_bounds(fake_stac, scene_asset, triangle_utm):
    triangle = gpd.GeoDataFrame(geometry=[triangle_utm], crs=SCENE_CRS).to_crs(4326)
    fake_stac.items_to_return = [FakeItem({"B04": scene_asset})]

    results = eeo.stac_search("sentinel-2-l2a", intersects=triangle)
    ds = results[0].load("B04")

    # The triangle spans 400 m of a 2 km scene: 40 pixels, not 200.
    height, width = ds.get_shape()
    assert 40 <= height <= 43
    assert 40 <= width <= 43


def test_load_without_mask_keeps_the_bounding_rectangle(fake_stac, scene_asset, triangle_utm):
    triangle = gpd.GeoDataFrame(geometry=[triangle_utm], crs=SCENE_CRS).to_crs(4326)
    fake_stac.items_to_return = [FakeItem({"B04": scene_asset})]

    ds = eeo.stac_search("sentinel-2-l2a", intersects=triangle)[0].load("B04")

    # Every pixel of the rectangle is data; nothing has been blanked.
    assert np.all(ds.to_array() == 500)


def test_load_with_mask_follows_the_geometry_outline(fake_stac, scene_asset, triangle_utm):
    triangle = gpd.GeoDataFrame(geometry=[triangle_utm], crs=SCENE_CRS).to_crs(4326)
    fake_stac.items_to_return = [FakeItem({"B04": scene_asset})]

    ds = eeo.stac_search("sentinel-2-l2a", intersects=triangle)[0].load("B04", mask=True)

    values = ds.to_array()
    inside = (values == 500).sum()
    outside = (values == 0).sum()
    assert inside > 0 and outside > 0
    # A right triangle keeps about half of its bounding rectangle.
    assert 0.35 < inside / values.size < 0.65


def test_mask_without_a_search_geometry_raises_validation_error(fake_stac, scene_asset):
    fake_stac.items_to_return = [FakeItem({"B04": scene_asset})]

    results = eeo.stac_search("sentinel-2-l2a", bbox=(11.0, 46.5, 11.2, 46.7))

    with pytest.raises(eeo.ValidationError, match="mask=True needs"):
        results[0].load("B04", mask=True)


def test_explicit_bbox_still_wins_over_the_search_geometry(fake_stac, scene_asset, triangle_utm):
    triangle = gpd.GeoDataFrame(geometry=[triangle_utm], crs=SCENE_CRS).to_crs(4326)
    fake_stac.items_to_return = [FakeItem({"B04": scene_asset})]
    left, top = SCENE_ORIGIN
    smaller = transform_bounds(
        SCENE_CRS, "EPSG:4326", left + 200, top - 400, left + 400, top - 200, densify_pts=21
    )

    ds = eeo.stac_search("sentinel-2-l2a", intersects=triangle)[0].load("B04", bbox=smaller)

    height, width = ds.get_shape()
    assert 20 <= height <= 23
    assert 20 <= width <= 23


def test_crop_false_reads_the_whole_scene_despite_a_geometry(fake_stac, scene_asset, triangle_utm):
    triangle = gpd.GeoDataFrame(geometry=[triangle_utm], crs=SCENE_CRS).to_crs(4326)
    fake_stac.items_to_return = [FakeItem({"B04": scene_asset})]

    ds = eeo.stac_search("sentinel-2-l2a", intersects=triangle)[0].load("B04", crop=False)

    assert ds.get_shape() == (SCENE_SIZE, SCENE_SIZE)
