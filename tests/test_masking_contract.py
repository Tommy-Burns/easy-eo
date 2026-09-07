"""Cross-cutting checks on masking (WP-26.5).

The decoders are pinned against the agencies' published tables in
``test_quality_masks.py`` and ``test_qa_pixel.py``, and the operation's own
behaviour in ``test_masking.py``. What is left, and what this module covers,
are the properties that span those pieces and so belong to none of them:

* the same quality values mask identically however the scene was loaded —
  from a downloaded product, from a catalog, or from a bare array;
* the nodata contract holds across every dtype a load can produce, not just
  the uint16 both missions happen to use;
* ``clear_fraction`` reports a proportion that was constructed, rather than
  one counted by hand off a six-pixel row.

Nothing here downloads anything.
"""

import datetime as dt

import numpy as np
import pytest
import rasterio as rio
from affine import Affine
from rasterio.crs import CRS

import eeo
from eeo.core.exceptions import ValidationError
from eeo.preprocessing.quality import SCLClass
from product_fixtures import build_safe

UTC = dt.timezone.utc
UTM = CRS.from_epsg(32633)
TRANSFORM = Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_200_000.0)
TIMESTAMP = dt.datetime(2024, 8, 30, 10, 6, 21, tzinfo=UTC)


# ---------------------------------------------------------------------------
# The three load routes, given identical pixels
# ---------------------------------------------------------------------------
class _Asset:
    def __init__(self, href):
        self.href = str(href)


class _Item:
    """The little a STACItem needs of a pystac Item."""

    def __init__(self, assets, *, properties=None):
        self.id = "PARITY_TEST"
        self.datetime = TIMESTAMP
        self.collection_id = "sentinel-2-l2a"
        self.properties = dict(properties or {})
        self.assets = {name: _Asset(href) for name, href in assets.items()}
        self.bbox = None


def _write_tif(path, array, *, nodata=0):
    """Write one band to a GeoTIFF on the shared grid."""
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=str(array.dtype),
        crs=UTM,
        transform=TRANSFORM,
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)
    return path


@pytest.fixture(scope="module")
def local_scene(tmp_path_factory):
    """A downloaded Sentinel-2 product, loaded as red plus SCL."""
    safe = build_safe(tmp_path_factory.mktemp("safe"))
    return eeo.load_sentinel2(safe, bands=["red", "scl"])


@pytest.fixture(scope="module")
def pixels(local_scene):
    """The exact arrays that product produced, to feed the other routes."""
    data = local_scene.read()
    return data[0].copy(), data[1].copy()


class TestEveryLoadRouteMasksAlike:
    """A workflow must not care which route the data took.

    The routes reach different code — a `.SAFE` reader, a STAC asset reader,
    and a bare array — but they converge on one dataset, and from there one
    mask. Where they diverged before, they diverged silently: 25.12 and 9.7
    were each a route that could load a quality band it could not then mask.
    """

    def test_the_fixture_actually_contains_both_kinds_of_pixel(self, pixels):
        # The premise. A scene that is all one class would let every
        # comparison below pass on a mask that does nothing.
        _, scl = pixels
        present = set(np.unique(scl).tolist())
        assert SCLClass.VEGETATION in present
        assert SCLClass.CLOUD_HIGH_PROBABILITY in present

    def test_a_bare_array_masks_like_the_downloaded_product(self, local_scene, pixels):
        red, scl = pixels
        array = eeo.load_array(
            np.stack([red, scl]), transform=TRANSFORM, crs=UTM, nodata=0
        ).to_rasterio()
        array.band_names = ["red", "scl"]
        assert np.array_equal(array.mask_clouds().read(), local_scene.mask_clouds().read())

    def test_a_stac_load_masks_like_the_downloaded_product(self, tmp_path, local_scene, pixels):
        red, scl = pixels
        item = eeo.io.STACItem(
            _Item(
                {
                    "red": _write_tif(tmp_path / "red.tif", red),
                    "scl": _write_tif(tmp_path / "scl.tif", scl),
                },
                properties={"platform": "sentinel-2a"},
            )
        )
        stac = item.load(["red", "scl"])
        assert stac.band_names == ["red", "scl"]
        assert np.array_equal(stac.mask_clouds().read(), local_scene.mask_clouds().read())

    def test_every_route_reports_the_same_clear_fraction(self, tmp_path, local_scene, pixels):
        red, scl = pixels
        array = eeo.load_array(
            np.stack([red, scl]), transform=TRANSFORM, crs=UTM, nodata=0
        ).to_rasterio()
        array.band_names = ["red", "scl"]
        item = eeo.io.STACItem(
            _Item(
                {
                    "red": _write_tif(tmp_path / "r.tif", red),
                    "scl": _write_tif(tmp_path / "s.tif", scl),
                }
            )
        )
        fractions = {
            "local": local_scene.mask_clouds().clear_fraction(),
            "array": array.mask_clouds().clear_fraction(),
            "stac": item.load(["red", "scl"]).mask_clouds().clear_fraction(),
        }
        assert len(set(fractions.values())) == 1, fractions
        # Half the fixture is high-probability cloud, so half must survive.
        assert fractions["local"] == pytest.approx(0.5)

    def test_a_landsat_quality_band_masks_alike_from_an_array_and_a_catalog(self, tmp_path):
        # 22280 is USGS's "High conf Cloud", 21824 its "Clear with lows set".
        qa = np.full((8, 8), 21824, dtype="uint16")
        qa[4:] = 22280
        red = np.full((8, 8), 1000, dtype="uint16")

        array = eeo.load_array(
            np.stack([red, qa]), transform=TRANSFORM, crs=UTM, nodata=0
        ).to_rasterio()
        array.band_names = ["red", "qa_pixel"]
        array.attrs = {"mission": "Landsat 9"}

        item = eeo.io.STACItem(
            _Item(
                {
                    "red": _write_tif(tmp_path / "lr.tif", red),
                    "qa_pixel": _write_tif(tmp_path / "lq.tif", qa),
                },
                properties={"platform": "landsat-9"},
            )
        )
        assert np.array_equal(
            array.mask_clouds().read(), item.load(["red", "qa_pixel"]).mask_clouds().read()
        )
        assert array.mask_clouds().clear_fraction() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The nodata contract, across dtypes
# ---------------------------------------------------------------------------
SCL_ROW = [SCLClass.VEGETATION, SCLClass.CLOUD_HIGH_PROBABILITY]

INTEGER_DTYPES = ["uint8", "uint16", "int16", "int32", "uint32", "int64"]
FLOAT_DTYPES = ["float32", "float64"]


def _two_band(dtype, *, nodata=None):
    """A 1x2 raster: one clear pixel, one high-probability cloud."""
    data = np.array([[[100, 200]], [[int(SCL_ROW[0]), int(SCL_ROW[1])]]], dtype=dtype)
    ds = eeo.load_array(data, transform=TRANSFORM, crs=UTM, nodata=nodata).to_rasterio()
    ds.band_names = ["red", "scl"]
    return ds


class TestNodataAcrossDtypes:
    """One sentence, held for every dtype: masked pixels take the raster's
    nodata, floats use NaN, and an integer raster that declares none is
    refused rather than assigned one."""

    @pytest.mark.parametrize("dtype", INTEGER_DTYPES)
    def test_an_integer_raster_keeps_its_sentinel_and_its_dtype(self, dtype):
        out = _two_band(dtype, nodata=0).mask_clouds()
        assert out.get_metadata()["dtype"] == dtype
        assert out.get_metadata()["nodata"] == 0
        assert out.read()[0].tolist() == [[100, 0]]

    @pytest.mark.parametrize("dtype", INTEGER_DTYPES)
    def test_an_integer_raster_without_nodata_is_refused(self, dtype):
        # An integer array cannot hold NaN, and guessing a sentinel could
        # delete real measurements.
        with pytest.raises(ValidationError, match="declares no nodata value"):
            _two_band(dtype).mask_clouds()

    @pytest.mark.parametrize("dtype", INTEGER_DTYPES)
    def test_an_explicit_sentinel_is_accepted_for_any_integer_dtype(self, dtype):
        out = _two_band(dtype).mask_clouds(nodata=1)
        assert out.read()[0].tolist() == [[100, 1]]
        assert out.get_metadata()["nodata"] == 1

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_a_float_raster_without_nodata_uses_nan(self, dtype):
        out = _two_band(dtype).mask_clouds()
        assert out.get_metadata()["dtype"] == dtype
        assert np.isnan(out.get_metadata()["nodata"])
        assert np.isnan(out.read()[0]).tolist() == [[False, True]]

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_a_float_raster_honours_a_declared_sentinel(self, dtype):
        out = _two_band(dtype, nodata=-9999).mask_clouds()
        assert out.get_metadata()["nodata"] == -9999
        assert out.read()[0].tolist() == [[100.0, -9999.0]]

    @pytest.mark.parametrize("dtype", ["uint8", "int16"])
    def test_a_sentinel_too_large_for_the_dtype_is_refused(self, dtype):
        with pytest.raises(ValidationError, match="does not fit"):
            _two_band(dtype).mask_clouds(nodata=10**9)

    @pytest.mark.parametrize("dtype", INTEGER_DTYPES + FLOAT_DTYPES)
    def test_the_mask_itself_does_not_depend_on_dtype(self, dtype):
        # Whatever the container, the same pixel is the cloudy one.
        out = _two_band(dtype, nodata=0).mask_clouds()
        assert out.clear_fraction() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# clear_fraction against a constructed proportion
# ---------------------------------------------------------------------------
class TestClearFractionAgainstAKnownProportion:
    """Counted off a built proportion rather than a six-pixel row."""

    @pytest.mark.parametrize("cloudy", [0, 1, 37, 500, 9_999, 10_000])
    def test_the_fraction_is_the_proportion_that_was_built(self, cloudy):
        total = 10_000
        scl = np.full(total, int(SCLClass.VEGETATION), dtype="uint16")
        scl[:cloudy] = int(SCLClass.CLOUD_HIGH_PROBABILITY)
        rng = np.random.default_rng(0)
        rng.shuffle(scl)  # scattered, so no test depends on a contiguous block

        data = np.stack([np.full(total, 1000, dtype="uint16"), scl]).reshape(2, 100, 100)
        ds = eeo.load_array(data, transform=TRANSFORM, crs=UTM, nodata=0).to_rasterio()
        ds.band_names = ["red", "scl"]

        assert ds.mask_clouds().clear_fraction() == pytest.approx((total - cloudy) / total)

    @pytest.mark.parametrize("dtype", INTEGER_DTYPES + FLOAT_DTYPES)
    def test_a_pixel_absent_from_any_band_is_not_clear(self, dtype):
        # Nodata is contagious, so the clear set is the intersection of the
        # bands and not their union. The two only differ on a raster whose
        # bands were masked separately -- which mask_clouds never produces,
        # so it has to be built deliberately.
        data = np.array([[[0, 1, 1, 1]], [[1, 0, 1, 1]]], dtype=dtype)
        ds = eeo.load_array(data, transform=TRANSFORM, crs=UTM, nodata=0).to_rasterio()
        assert ds.clear_fraction() == pytest.approx(0.5)
        assert ds.clear_fraction(band=1) == pytest.approx(0.75)
        assert ds.clear_fraction(band=2) == pytest.approx(0.75)

    def test_an_unmasked_scene_of_known_fill_reports_that_fill(self):
        red = np.full(10_000, 1000, dtype="uint16")
        red[:2_500] = 0  # a quarter of the scene is fill
        ds = eeo.load_array(
            red.reshape(1, 100, 100), transform=TRANSFORM, crs=UTM, nodata=0
        ).to_rasterio()
        assert ds.clear_fraction() == pytest.approx(0.75)

    def test_fill_and_cloud_together_are_counted_once(self):
        # A pixel that is both fill and cloudy must not be subtracted twice.
        red = np.full(100, 1000, dtype="uint16")
        red[:20] = 0  # fill
        scl = np.full(100, int(SCLClass.VEGETATION), dtype="uint16")
        scl[:30] = int(SCLClass.CLOUD_HIGH_PROBABILITY)  # overlaps the fill
        ds = eeo.load_array(
            np.stack([red, scl]).reshape(2, 10, 10), transform=TRANSFORM, crs=UTM, nodata=0
        ).to_rasterio()
        ds.band_names = ["red", "scl"]
        # 30 cloudy pixels cover all 20 fill pixels, so 70 survive.
        assert ds.mask_clouds().clear_fraction() == pytest.approx(0.70)
