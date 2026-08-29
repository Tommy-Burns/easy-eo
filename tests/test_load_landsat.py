"""Tests for load_landsat (eeo/io/products.py).

The fixture, from :mod:`product_fixtures`, is a real Collection 2 Level-2
product: genuine GeoTIFFs written through GDAL on one 30 m grid, beside a
metadata file of the real shape, with the fill values USGS actually declares —
0 in the imagery and 1 in the quality rasters, which is what makes the choice
of leading band observable. Two products are built, Landsat 9 and Landsat 7,
because the whole point of naming bands is that ``"red"`` reaches band 4 on one
and band 3 on the other.

Nothing is downloaded.
"""

import datetime as dt

import numpy as np
import pytest
from rasterio.warp import transform_bounds

import eeo
from eeo.core.exceptions import ValidationError
from product_fixtures import (
    L7_ID,
    L9_ID,
    OLI_BANDS,
    TM_BANDS,
)
from product_fixtures import (
    LANDSAT_CRS as CRS,
)
from product_fixtures import (
    LANDSAT_RES as RES,
)
from product_fixtures import (
    LANDSAT_SIZE as SIZE,
)
from product_fixtures import (
    LANDSAT_ULX as ULX,
)
from product_fixtures import (
    LANDSAT_ULY as ULY,
)
from product_fixtures import (
    build_landsat as build_product,
)


@pytest.fixture
def product(tmp_path):
    """A Landsat 9 Level-2 science product."""
    return build_product(tmp_path)


class TestLoading:
    """A product on disk becomes a dataset on the scene's own grid."""

    def test_returns_a_rasterio_backed_dataset(self, product):
        scene = eeo.load_landsat(product, ["red", "nir08"])
        assert scene.to_array().shape == (2, SIZE, SIZE)
        assert scene.get_crs().to_string() == CRS
        assert scene.get_transform()[0] == RES
        assert scene.get_transform()[2] == ULX

    def test_values_are_the_stored_digital_numbers(self, product):
        got = eeo.load_landsat(product, ["red"]).to_array()[0]
        assert np.all(got == OLI_BANDS["SR_B4"])

    def test_band_names_are_the_common_names(self, product):
        assert eeo.load_landsat(product, ["red", "nir08"]).band_names == ["red", "nir08"]

    def test_native_tokens_resolve_to_common_names(self, product):
        assert eeo.load_landsat(product, ["SR_B4", "B5"]).band_names == ["red", "nir08"]

    def test_band_order_follows_the_request(self, product):
        scene = eeo.load_landsat(product, ["nir08", "red"])
        assert scene.band_names == ["nir08", "red"]
        assert scene.to_array()[0].max() == OLI_BANDS["SR_B5"]
        assert scene.to_array()[1].max() == OLI_BANDS["SR_B4"]

    def test_metadata_file_can_be_given_directly(self, product):
        metadata = product / f"{L9_ID}_MTL.json"
        assert eeo.load_landsat(metadata, ["red"]).band_names == ["red"]

    def test_timestamp_is_the_acquisition_time(self, product):
        scene = eeo.load_landsat(product, ["red"])
        assert scene.timestamp == dt.datetime(
            2026, 8, 22, 10, 4, 22, 118358, tzinfo=dt.timezone.utc
        )

    def test_a_thermal_band_loads_beside_a_reflectance_one(self, product):
        scene = eeo.load_landsat(product, ["red", "lwir11"])
        assert scene.band_names == ["red", "lwir11"]
        assert scene.to_array()[1].max() == OLI_BANDS["ST_B10"]


class TestOneLoaderAcrossMissions:
    """The same band names reach different band numbers on different sensors."""

    def test_red_is_band_4_on_landsat_9(self, product):
        assert eeo.load_landsat(product, ["red"]).to_array()[0].max() == OLI_BANDS["SR_B4"]

    def test_red_is_band_3_on_landsat_7(self, tmp_path):
        seven = build_product(tmp_path, product_id=L7_ID)
        assert eeo.load_landsat(seven, ["red"]).to_array()[0].max() == TM_BANDS["SR_B3"]

    def test_the_same_request_works_on_both(self, tmp_path):
        nine = build_product(tmp_path, product_id=L9_ID)
        seven = build_product(tmp_path, product_id=L7_ID)
        request = ["red", "nir08", "swir16"]
        assert eeo.load_landsat(nine, request).band_names == request
        assert eeo.load_landsat(seven, request).band_names == request

    def test_the_mission_is_reported_from_the_metadata(self, tmp_path):
        seven = build_product(tmp_path, product_id=L7_ID)
        attrs = eeo.load_landsat(seven, ["red"]).attrs
        assert attrs["mission"] == "Landsat 7"
        assert attrs["instrument"] == "ETM+"
        assert attrs["spacecraft"] == "LANDSAT_7"

    def test_landsat_7_thermal_is_band_6(self, tmp_path):
        seven = build_product(tmp_path, product_id=L7_ID)
        scene = eeo.load_landsat(seven, ["lwir"])
        assert scene.to_array()[0].max() == TM_BANDS["ST_B6"]

    def test_a_landsat_9_band_name_is_refused_on_landsat_7(self, tmp_path):
        # Landsat 7 has no band 9 at all, so this is a band table question
        # rather than a missing-file one.
        seven = build_product(tmp_path, product_id=L7_ID)
        with pytest.raises(ValidationError, match="no band"):
            eeo.load_landsat(seven, ["coastal"])


class TestNodata:
    """The result's nodata describes the imagery, not a quality raster."""

    def test_imagery_fill_is_zero(self, product):
        assert eeo.load_landsat(product, ["red"]).get_metadata()["nodata"] == 0

    def test_a_leading_quality_band_does_not_supply_nodata(self, product):
        # QA_PIXEL declares 1; asking for it first must not make 1 the fill
        # value of the reflectance band travelling with it.
        scene = eeo.load_landsat(product, ["qa_pixel", "red"])
        assert scene.band_names == ["qa_pixel", "red"]
        assert scene.get_metadata()["nodata"] == 0

    def test_quality_values_survive_being_read_second(self, product):
        scene = eeo.load_landsat(product, ["qa_pixel", "red"])
        assert scene.to_array()[0].max() == OLI_BANDS["QA_PIXEL"]
        assert scene.to_array()[1].max() == OLI_BANDS["SR_B4"]

    def test_a_quality_only_request_keeps_its_own_fill(self, product):
        assert eeo.load_landsat(product, ["qa_pixel"]).get_metadata()["nodata"] == 1


class TestProvenance:
    """The result carries enough to say which scene it came from."""

    def test_identity_is_recorded(self, product):
        attrs = eeo.load_landsat(product, ["red"]).attrs
        assert attrs["product"] == L9_ID
        assert attrs["mission"] == "Landsat 9"
        assert attrs["level"] == "L2SP"
        assert (attrs["wrs_path"], attrs["wrs_row"]) == ("193", "028")
        assert attrs["collection_number"] == "02"
        assert attrs["collection_category"] == "T1"
        assert attrs["resolution"] == 30

    def test_requested_bands_are_recorded_both_ways(self, product):
        attrs = eeo.load_landsat(product, ["red", "nir08"]).attrs
        assert attrs["bands"] == ["red", "nir08"]
        assert attrs["band_ids"] == ["SR_B4", "SR_B5"]

    def test_scaling_coefficients_travel_with_the_data(self, product):
        # Carried, never applied: the values above are digital numbers.
        attrs = eeo.load_landsat(product, ["red"]).attrs
        assert attrs["reflectance_scaling"]["SR_B4"] == (2.75e-05, -0.2)
        assert attrs["temperature_scaling"]["ST_B10"] == (0.00341802, 149.0)


class TestCropping:
    """A bbox in lon/lat degrees selects a window of the scene."""

    def test_bbox_reads_a_smaller_window(self, product):
        quarter = (ULX, ULY - SIZE / 2 * RES, ULX + SIZE / 2 * RES, ULY)
        bounds = transform_bounds(CRS, "EPSG:4326", *quarter, densify_pts=21)
        scene = eeo.load_landsat(product, ["red"], bbox=bounds)
        height, width = scene.to_array().shape[-2:]
        assert height < SIZE and width < SIZE
        assert np.all(scene.to_array()[0] == OLI_BANDS["SR_B4"])

    def test_bands_stay_aligned_under_a_bbox(self, product):
        quarter = (ULX, ULY - SIZE / 2 * RES, ULX + SIZE / 2 * RES, ULY)
        bounds = transform_bounds(CRS, "EPSG:4326", *quarter, densify_pts=21)
        scene = eeo.load_landsat(product, ["red", "nir08"], bbox=bounds)
        array = scene.to_array()
        assert array.shape[0] == 2 and array.shape[1] < SIZE
        assert array[1].max() == OLI_BANDS["SR_B5"]

    def test_a_bbox_off_the_scene_is_refused(self, product):
        with pytest.raises(ValidationError, match="does not overlap"):
            eeo.load_landsat(product, ["red"], bbox=(-50.0, -30.0, -49.9, -29.9))


class TestRefusals:
    """What the loader will not do, and what it says instead."""

    def test_a_missing_product_is_named(self, tmp_path):
        with pytest.raises(ValidationError, match="no such Landsat product"):
            eeo.load_landsat(tmp_path / "absent", ["red"])

    def test_level_1_is_refused_by_name(self, tmp_path):
        one = build_product(
            tmp_path, product_id="LC09_L1TP_193028_20260822_20260823_02_T1", level="L1TP"
        )
        with pytest.raises(ValidationError, match="Level-1 is not supported"):
            eeo.load_landsat(one, ["red"])

    def test_an_unknown_band_name_is_refused(self, product):
        with pytest.raises(ValidationError, match="no band"):
            eeo.load_landsat(product, ["chlorophyll"])

    def test_a_band_the_product_lacks_is_refused(self, tmp_path):
        # L2SR: reflectance without surface temperature, which is a normal
        # product rather than a damaged one.
        tokens = [t for t in OLI_BANDS if not t.startswith("ST_")]
        reflectance_only = build_product(tmp_path, tokens=tokens, level="L2SR")
        with pytest.raises(ValidationError, match="holds no 'lwir11'"):
            eeo.load_landsat(reflectance_only, ["lwir11"])

    def test_a_reflectance_only_product_still_loads_its_bands(self, tmp_path):
        tokens = [t for t in OLI_BANDS if not t.startswith("ST_")]
        reflectance_only = build_product(tmp_path, tokens=tokens, level="L2SR")
        scene = eeo.load_landsat(reflectance_only, ["red", "nir08"])
        assert scene.attrs["level"] == "L2SR"
        assert scene.band_names == ["red", "nir08"]

    def test_a_listed_but_absent_file_is_named(self, tmp_path):
        truncated = build_product(tmp_path, omit_file="SR_B4")
        with pytest.raises(ValidationError, match="does not hold it"):
            eeo.load_landsat(truncated, ["red"])

    def test_level_override_contradicting_the_metadata_is_refused(self, product):
        with pytest.raises(ValidationError, match="contradicts the product"):
            eeo.load_landsat(product, ["red"], level="L2SR")

    def test_level_override_agreeing_with_the_metadata_is_accepted(self, product):
        assert eeo.load_landsat(product, ["red"], level="L2SP").band_names == ["red"]
