"""Tests for load_sentinel2 (eeo/io/products.py).

The fixture, from :mod:`product_fixtures`, is a real ``.SAFE`` tree: genuine
JPEG 2000 images written through GDAL, laid out in the resolution
subdirectories a product uses, under manifests of the real shape — the same
manifests the metadata tests parse. So the loader is exercised over actual
image reads, actual warping between resolutions, and actual georeferencing;
nothing about the read path is stubbed. Nothing is downloaded.
"""

import numpy as np
import pytest
import rasterio as rio
from rasterio.warp import transform_bounds

import eeo
from eeo.core.exceptions import ValidationError
from product_fixtures import (
    S2_FILL as FILL,
)
from product_fixtures import (
    S2_GRANULE as GRANULE,
)
from product_fixtures import (
    S2_LAYOUT as LAYOUT,
)
from product_fixtures import (
    S2_SENSING_TIME,
    build_safe,
)
from product_fixtures import (
    S2_SIZE_10M as SIZE_10M,
)
from product_fixtures import (
    S2_STEM as STEM,
)
from product_fixtures import (
    S2_ULX as ULX,
)
from product_fixtures import (
    S2_ULY as ULY,
)


@pytest.fixture
def safe(tmp_path):
    """A whole Level-2A product, images included."""
    return build_safe(tmp_path)


class TestValuesAndNaming:
    """A load returns the product's own numbers under common band names."""

    def test_bands_come_back_as_stored_digital_numbers(self, safe):
        scene = eeo.load_sentinel2(safe, ["red", "nir"])
        array = scene.to_array()
        assert array.dtype == np.uint16
        assert array[0][0, 0] == FILL["B04"]
        assert array[1][0, 0] == FILL["B08"]

    def test_values_match_reading_the_image_directly(self, safe):
        # The contract: what a GIS shows is what comes back.
        direct = safe / "GRANULE" / GRANULE / "IMG_DATA" / "R10m" / f"{STEM}_B04_10m.jp2"
        with rio.open(direct) as src:
            expected = src.read(1)
        got = eeo.load_sentinel2(safe, ["red"]).to_array()[0]
        assert np.array_equal(got, expected)

    def test_band_names_are_common_names(self, safe):
        assert eeo.load_sentinel2(safe, ["red", "nir"]).band_names == ["red", "nir"]

    def test_native_ids_resolve_to_common_names(self, safe):
        assert eeo.load_sentinel2(safe, ["B04", "B08"]).band_names == ["red", "nir"]

    def test_order_follows_the_request(self, safe):
        scene = eeo.load_sentinel2(safe, ["nir", "red"])
        assert scene.band_names == ["nir", "red"]
        assert scene.to_array()[0][0, 0] == FILL["B08"]

    def test_georeferencing_and_timestamp(self, safe):
        scene = eeo.load_sentinel2(safe, ["red"])
        assert scene.get_crs().to_string() == "EPSG:32632"
        assert scene.get_transform()[0] == 10.0
        assert scene.timestamp.isoformat() == S2_SENSING_TIME

    def test_provenance_is_recorded(self, safe):
        attrs = eeo.load_sentinel2(safe, ["red"]).attrs
        assert attrs["mission"] == "Sentinel-2"
        assert attrs["level"] == "L2A"
        assert attrs["tile"] == "T32TPS"
        assert attrs["processing_baseline"] == "05.11"
        assert attrs["bands"] == ["red"] and attrs["band_ids"] == ["B04"]
        # The coefficients a reader needs to convert to reflectance by hand.
        assert attrs["quantification_value"] == 10000.0
        assert attrs["band_offsets"]["B04"] == -1000.0


class TestResolution:
    """The output grid keeps the finest detail requested, and no more."""

    def test_default_is_the_finest_native_resolution(self, safe):
        scene = eeo.load_sentinel2(safe, ["red", "swir16"])
        assert scene.get_transform()[0] == 10.0
        assert scene.to_array().shape == (2, SIZE_10M, SIZE_10M)

    def test_all_coarse_bands_stay_coarse(self, safe):
        # Nothing requested was sensed at 10 m, so upsampling would invent detail.
        scene = eeo.load_sentinel2(safe, ["swir16", "swir22"])
        assert scene.get_transform()[0] == 20.0

    def test_explicit_resolution(self, safe):
        scene = eeo.load_sentinel2(safe, ["red", "swir16"], resolution=20)
        assert scene.get_transform()[0] == 20.0
        assert scene.to_array().shape == (2, SIZE_10M // 2, SIZE_10M // 2)

    def test_coarse_band_is_warped_onto_the_fine_grid(self, safe):
        scene = eeo.load_sentinel2(safe, ["red", "swir16"], resolution=10)
        array = scene.to_array()
        assert array.shape == (2, SIZE_10M, SIZE_10M)
        assert array[1][0, 0] == FILL["B11"]

    def test_order_is_restored_when_a_coarse_band_leads(self, safe):
        # swir16 cannot define a 10 m grid, so red leads internally; the
        # caller's order must survive that.
        scene = eeo.load_sentinel2(safe, ["swir16", "red"], resolution=10)
        assert scene.band_names == ["swir16", "red"]
        array = scene.to_array()
        assert array[0][0, 0] == FILL["B11"]
        assert array[1][0, 0] == FILL["B04"]

    def test_invalid_resolution_rejected(self, safe):
        with pytest.raises(ValidationError, match="resolution must be one of"):
            eeo.load_sentinel2(safe, ["red"], resolution=15)

    def test_no_band_at_the_requested_resolution(self, safe):
        # B08 exists only at 10 m, so a 20 m request has no grid to read onto.
        with pytest.raises(ValidationError) as excinfo:
            eeo.load_sentinel2(safe, ["nir"], resolution=20)
        message = str(excinfo.value)
        assert "none of the requested bands is written at 20 m" in message
        assert "nir at 10 m" in message


class TestQualityBands:
    """Class numbers are never blended, whatever resampling is asked for."""

    def test_scl_keeps_its_classes_when_warped(self, safe):
        # SCL is written at 20 m and must land on the 10 m grid by nearest
        # neighbour even though bilinear was requested for the rest.
        scene = eeo.load_sentinel2(safe, ["red", "scl"], resolution=10, resampling="bilinear")
        classes = set(np.unique(scene.to_array()[1]))
        assert classes <= {4, 9}, f"blending invented classes: {sorted(classes)}"

    def test_scl_can_be_loaded_alone(self, safe):
        scene = eeo.load_sentinel2(safe, ["scl"])
        assert scene.band_names == ["scl"]
        assert scene.get_transform()[0] == 20.0


class TestCropping:
    """A bbox bounds what is read."""

    def test_bbox_reads_a_window(self, safe):
        bounds = transform_bounds(
            "EPSG:32632", "EPSG:4326", ULX, ULY - 200, ULX + 200, ULY, densify_pts=21
        )
        scene = eeo.load_sentinel2(safe, ["red"], bbox=bounds)
        height, width = scene.to_array().shape[-2:]
        assert height < SIZE_10M and width < SIZE_10M

    def test_bbox_outside_the_tile_rejected(self, safe):
        with pytest.raises(ValidationError, match="does not overlap"):
            eeo.load_sentinel2(safe, ["red"], bbox=(-50.0, -30.0, -49.9, -29.9))


class TestRefusals:
    """Unsupported or malformed requests fail with an actionable message."""

    def test_level1c_is_refused(self, tmp_path):
        safe = build_safe(tmp_path, level="1C", product_type="S2MSI1C")
        with pytest.raises(ValidationError, match="Level-1C is not supported"):
            eeo.load_sentinel2(safe, ["red"])

    def test_unknown_band_rejected(self, safe):
        with pytest.raises(ValidationError, match="no band 'purple'"):
            eeo.load_sentinel2(safe, ["purple"])

    def test_duplicate_band_rejected(self, safe):
        with pytest.raises(ValidationError, match="named twice"):
            eeo.load_sentinel2(safe, ["red", "B04"])

    def test_empty_band_list_rejected(self, safe):
        with pytest.raises(ValidationError, match="at least one band"):
            eeo.load_sentinel2(safe, [])

    def test_a_bare_string_is_not_a_band_list(self, safe):
        with pytest.raises(ValidationError, match="expected a sequence of band names"):
            eeo.load_sentinel2(safe, "red")

    def test_band_absent_from_the_product(self, tmp_path):
        layout = {k: v for k, v in LAYOUT.items() if k != "B12"}
        safe = build_safe(tmp_path, layout=layout)
        with pytest.raises(ValidationError, match="no 'swir22' \\(B12\\) band at all"):
            eeo.load_sentinel2(safe, ["swir22"])

    def test_manifest_listing_a_file_that_is_missing(self, tmp_path):
        safe = build_safe(tmp_path, omit_file=("B04", 10))
        with pytest.raises(ValidationError, match="lists an image the product does not hold"):
            eeo.load_sentinel2(safe, ["red"], resolution=10)


class TestChaining:
    """The result is an ordinary dataset and behaves like one."""

    def test_ndvi_by_band_name(self, safe):
        scene = eeo.load_sentinel2(safe, ["red", "nir"])
        ndvi = scene.ndvi(red="red", nir="nir")
        expected = (FILL["B08"] - FILL["B04"]) / (FILL["B08"] + FILL["B04"])
        assert ndvi.to_array()[0][0, 0] == pytest.approx(expected, abs=1e-6)

    def test_names_survive_a_save_and_reload(self, safe, tmp_path):
        out = tmp_path / "stack.tif"
        eeo.load_sentinel2(safe, ["red", "nir"]).save_raster(out)
        assert eeo.load_raster(out).band_names == ["red", "nir"]
