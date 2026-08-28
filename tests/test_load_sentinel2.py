"""Tests for load_sentinel2 (eeo/io/products.py).

The fixture is a real ``.SAFE`` tree: genuine JPEG 2000 images written through
GDAL, laid out in the resolution subdirectories a product uses, under manifests
of the real shape. So the loader is exercised over actual image reads, actual
warping between resolutions, and actual georeferencing — nothing about the read
path is stubbed. Nothing is downloaded.
"""

import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import from_origin

import eeo
from eeo.core.exceptions import ValidationError

GRANULE = "L2A_T32TPS_A039516_20240929T101318"
PRODUCT = "S2B_MSIL2A_20240929T100719_N0511_R022_T32TPS_20240929T133706.SAFE"
STEM = "T32TPS_20240929T100719"
ULX, ULY = 300000.0, 5100000.0
SIZE_10M = 64

# Which bands the fixture writes, and at which resolutions, mirroring a real
# product: B08 only at 10 m, B01 at 20 m and 60 m though it is sensed at 60,
# B09 only at 60 m, SCL at 20 m and 60 m.
LAYOUT = {
    "B02": (10, 20),
    "B03": (10, 20),
    "B04": (10, 20, 60),
    "B08": (10,),
    "B05": (20, 60),
    "B11": (20, 60),
    "B12": (20, 60),
    "B8A": (20, 60),
    "B01": (20, 60),
    "B09": (60,),
    "SCL": (20, 60),
}
FILL = {
    "B02": 700,
    "B03": 900,
    "B04": 1234,
    "B08": 4321,
    "B05": 3000,
    "B11": 2222,
    "B12": 1800,
    "B8A": 4000,
    "B01": 1100,
    "B09": 1500,
}


def _write_jp2(path, res, size, band):
    """Write one georeferenced JPEG 2000 image."""
    if band == "SCL":
        # Distinct class numbers in blocks, so any blending is detectable.
        data = np.zeros((size, size), dtype="uint16")
        data[: size // 2] = 4  # vegetation
        data[size // 2 :] = 9  # cloud high probability
        data = data[None]
    else:
        data = np.full((1, size, size), FILL[band], dtype="uint16")
    # Only the options JP2OpenJPEG accepts: copying a GeoTIFF profile would
    # carry INTERLEAVE and tiling keys the driver rejects.
    #
    # REVERSIBLE and QUALITY are both required, and only together: the writer
    # is lossy by default and stays lossy with either one alone, which quietly
    # rewrites a class number 4 as 3 and would make a resampling test read as a
    # loader bug. Real products are losslessly encoded, so the fixture must be
    # too or it is testing the compressor rather than the reader.
    profile = {
        "driver": "JP2OpenJPEG",
        "height": size,
        "width": size,
        "count": 1,
        "dtype": "uint16",
        "crs": "EPSG:32632",
        "transform": from_origin(ULX, ULY, res, res),
        "nodata": 0,
        "REVERSIBLE": "YES",
        "QUALITY": "100",
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(data)


def build_safe(tmp_path, *, level="2A", product_type="S2MSI2A", layout=None, omit_file=None):
    """Build a .SAFE tree with real JPEG 2000 images and return its path."""
    layout = LAYOUT if layout is None else layout
    safe = tmp_path / PRODUCT
    img = safe / "GRANULE" / GRANULE / "IMG_DATA"
    entries = []
    for band, resolutions in layout.items():
        for res in resolutions:
            (img / f"R{res}m").mkdir(parents=True, exist_ok=True)
            relative = f"GRANULE/{GRANULE}/IMG_DATA/R{res}m/{STEM}_{band}_{res}m"
            entries.append(relative)
            if omit_file == (band, res):
                continue
            _write_jp2(safe / f"{relative}.jp2", res, SIZE_10M * 10 // res, band)

    files = "".join(f"<IMAGE_FILE>{e}</IMAGE_FILE>" for e in entries)
    (safe / f"MTD_MSIL{level}.xml").write_text(
        f'<?xml version="1.0"?>'
        f'<n1:Level-{level}_User_Product xmlns:n1="https://psd-15.sentinel2.eo.esa.int/'
        f'PSD/User_Product_Level-{level}.xsd"><n1:General_Info><Product_Info>'
        f"<PRODUCT_START_TIME>2024-09-29T10:07:19.024Z</PRODUCT_START_TIME>"
        f"<PRODUCT_URI>{PRODUCT}</PRODUCT_URI>"
        f"<PRODUCT_TYPE>{product_type}</PRODUCT_TYPE>"
        f"<PROCESSING_BASELINE>05.11</PROCESSING_BASELINE>"
        f'<Product_Organisation><Granule_List><Granule granuleIdentifier="{GRANULE}">'
        f"{files}</Granule></Granule_List></Product_Organisation></Product_Info>"
        f"<Product_Image_Characteristics><QUANTIFICATION_VALUES_LIST>"
        f'<BOA_QUANTIFICATION_VALUE unit="none">10000</BOA_QUANTIFICATION_VALUE>'
        f"</QUANTIFICATION_VALUES_LIST><BOA_ADD_OFFSET_VALUES_LIST>"
        f'<BOA_ADD_OFFSET band_id="3">-1000</BOA_ADD_OFFSET>'
        f"</BOA_ADD_OFFSET_VALUES_LIST><Spectral_Information_List>"
        f'<Spectral_Information bandId="3" physicalBand="B4"/>'
        f"</Spectral_Information_List></Product_Image_Characteristics>"
        f"</n1:General_Info></n1:Level-{level}_User_Product>"
    )
    (safe / "GRANULE" / GRANULE / "MTD_TL.xml").write_text(
        '<?xml version="1.0"?>'
        '<n1:Level-2A_Tile_ID xmlns:n1="https://psd-15.sentinel2.eo.esa.int/'
        'PSD/S2_PDI_Level-2A_Tile_Metadata.xsd"><n1:General_Info>'
        "<SENSING_TIME>2024-09-29T10:17:59.919283Z</SENSING_TIME></n1:General_Info>"
        "<n1:Geometric_Info><Tile_Geocoding>"
        "<HORIZONTAL_CS_CODE>EPSG:32632</HORIZONTAL_CS_CODE>"
        "</Tile_Geocoding></n1:Geometric_Info></n1:Level-2A_Tile_ID>"
    )
    return safe


@pytest.fixture
def safe(tmp_path):
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
        assert scene.timestamp.isoformat() == "2024-09-29T10:17:59.919283+00:00"

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
        from rasterio.warp import transform_bounds

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
