"""Tests for the per-sensor band tables (eeo/io/_bands.py).

The tables are the place a cross-sensor mistake becomes invisible, so the
checks here are mostly invariants rather than spot values: that a common name
means the same wavelength on every sensor that has it, that no name is used
twice within one sensor, and that the Landsat 7 / Landsat 8-9 numbering
divergence resolves in both directions.
"""

import pytest

from eeo.core.exceptions import ValidationError
from eeo.io._bands import (
    LANDSAT_OLI,
    LANDSAT_TM,
    SENTINEL2,
    finest_resolution,
    landsat_band_file,
    landsat_bands,
    resolve_band,
    resolve_bands,
    sentinel2_available_resolutions,
    sentinel2_band_file,
)

ALL_TABLES = {"sentinel-2": SENTINEL2, "landsat-oli": LANDSAT_OLI, "landsat-tm": LANDSAT_TM}

# Wavelength ranges for each common name, from the STAC Electro-Optical
# extension. A band whose centre falls outside its name's range is mislabelled.
STAC_RANGES = {
    "coastal": (0.40, 0.45),
    "blue": (0.45, 0.53),
    "green": (0.51, 0.60),
    "red": (0.62, 0.69),
    "rededge1": (0.69, 0.79),
    "rededge2": (0.69, 0.79),
    "rededge3": (0.69, 0.79),
    "nir": (0.76, 1.00),
    "nir08": (0.80, 0.90),
    "nir09": (0.90, 1.00),
    "swir16": (1.55, 1.75),
    "swir22": (2.08, 2.35),
    "lwir": (10.4, 12.5),
    "lwir11": (10.5, 11.5),
}

# Real Sentinel-2 image paths, one per resolution the product writes the band
# at. Availability is irregular: B01 is sensed at 60 m but written at 20 m too,
# B08 only at 10 m, B09 only at 60 m.
S2_FILES = [
    "GRANULE/L2A_T32TPS_A039/IMG_DATA/R10m/T32TPS_20240929T100719_B04_10m",
    "GRANULE/L2A_T32TPS_A039/IMG_DATA/R20m/T32TPS_20240929T100719_B04_20m",
    "GRANULE/L2A_T32TPS_A039/IMG_DATA/R60m/T32TPS_20240929T100719_B04_60m",
    "GRANULE/L2A_T32TPS_A039/IMG_DATA/R10m/T32TPS_20240929T100719_B08_10m",
    "GRANULE/L2A_T32TPS_A039/IMG_DATA/R20m/T32TPS_20240929T100719_B11_20m",
    "GRANULE/L2A_T32TPS_A039/IMG_DATA/R60m/T32TPS_20240929T100719_B11_60m",
    "GRANULE/L2A_T32TPS_A039/IMG_DATA/R20m/T32TPS_20240929T100719_B01_20m",
    "GRANULE/L2A_T32TPS_A039/IMG_DATA/R60m/T32TPS_20240929T100719_B01_60m",
    "GRANULE/L2A_T32TPS_A039/IMG_DATA/R20m/T32TPS_20240929T100719_SCL_20m",
]

L9_FILES = {
    t: f"LC09_L2SP_193028_20260822_20260823_02_T1_{t}.TIF"
    for t in (
        "SR_B1",
        "SR_B2",
        "SR_B3",
        "SR_B4",
        "SR_B5",
        "SR_B6",
        "SR_B7",
        "ST_B10",
        "QA_PIXEL",
        "QA_RADSAT",
        "SR_QA_AEROSOL",
    )
}


class TestTheBandNumberingTrap:
    """Landsat 7 and Landsat 8/9 number the same wavelengths differently."""

    def test_red_is_a_different_band_number_per_sensor(self):
        assert landsat_bands(7)["red"].band_id == "SR_B3"
        assert landsat_bands(8)["red"].band_id == "SR_B4"
        assert landsat_bands(9)["red"].band_id == "SR_B4"

    def test_the_same_native_id_means_different_things(self):
        # This is the whole reason common names are the primary spelling.
        assert resolve_band("SR_B4", landsat_bands(7)).common_name == "nir08"
        assert resolve_band("SR_B4", landsat_bands(9)).common_name == "red"

    def test_landsat_8_and_9_share_one_table(self):
        assert landsat_bands(8) is landsat_bands(9)

    def test_landsat_4_5_and_7_share_one_table(self):
        assert landsat_bands(4) is landsat_bands(5) is landsat_bands(7)

    def test_older_sensors_have_no_coastal_band(self):
        assert "coastal" not in landsat_bands(7)
        assert "coastal" in landsat_bands(9)

    def test_unknown_mission_rejected(self):
        with pytest.raises(ValidationError, match="no band table for Landsat 3"):
            landsat_bands(3)


class TestTableInvariants:
    """Properties that must hold for every table, whatever the sensor."""

    @pytest.mark.parametrize("label", sorted(ALL_TABLES))
    def test_common_names_are_unique(self, label):
        # The EO extension requires it, and a duplicate would make a name
        # ambiguous exactly where it is meant to disambiguate.
        table = ALL_TABLES[label]
        assert len(table) == len({band.common_name for band in table.values()})

    @pytest.mark.parametrize("label", sorted(ALL_TABLES))
    def test_native_ids_are_unique(self, label):
        table = ALL_TABLES[label]
        assert len(table) == len({band.band_id for band in table.values()})

    @pytest.mark.parametrize("label", sorted(ALL_TABLES))
    def test_wavelengths_match_their_common_name(self, label):
        for band in ALL_TABLES[label].values():
            if band.wavelength is None:
                continue
            low, high = STAC_RANGES[band.common_name]
            assert low <= band.wavelength <= high, (
                f"{label} {band.common_name} at {band.wavelength} um is outside "
                f"the {low}-{high} um range that name denotes"
            )

    @pytest.mark.parametrize("label", sorted(ALL_TABLES))
    def test_the_table_key_matches_the_band(self, label):
        for name, band in ALL_TABLES[label].items():
            assert name == band.common_name

    @pytest.mark.parametrize("label", sorted(ALL_TABLES))
    def test_quality_bands_carry_no_wavelength(self, label):
        for band in ALL_TABLES[label].values():
            if band.kind in ("quality", "ancillary"):
                assert band.wavelength is None

    def test_red_means_the_same_wavelength_on_every_sensor(self):
        centres = [table["red"].wavelength for table in ALL_TABLES.values()]
        assert max(centres) - min(centres) < 0.03

    def test_sentinel2_level2a_has_no_cirrus_band(self):
        # B10 is consumed by the atmospheric correction and never written.
        assert all(band.band_id != "B10" for band in SENTINEL2.values())


class TestResolveBand:
    """Names resolve exactly, in several spellings, and fail loudly otherwise."""

    @pytest.mark.parametrize("spec", ["red", "RED", "  Red  ", "B04", "b04", "B4"])
    def test_sentinel2_spellings(self, spec):
        assert resolve_band(spec, SENTINEL2).band_id == "B04"

    @pytest.mark.parametrize("spec", ["red", "SR_B4", "sr_b4", "B4", " b4 "])
    def test_landsat_spellings(self, spec):
        assert resolve_band(spec, LANDSAT_OLI).band_id == "SR_B4"

    def test_unknown_band_lists_what_is_available(self):
        with pytest.raises(ValidationError) as excinfo:
            resolve_band("purple", SENTINEL2)
        message = str(excinfo.value)
        assert "no band 'purple'" in message
        assert "red" in message and "B04" in message

    def test_a_band_from_another_sensor_is_not_found(self):
        # "coastal" exists on OLI but not on ETM+.
        with pytest.raises(ValidationError, match="no band 'coastal'"):
            resolve_band("coastal", LANDSAT_TM)

    @pytest.mark.parametrize("spec", [4, None, 4.0, ["red"]])
    def test_non_string_rejected(self, spec):
        with pytest.raises(ValidationError, match="must be named by a string"):
            resolve_band(spec, SENTINEL2)


class TestResolveBands:
    """Several bands at once: order kept, duplicates refused."""

    def test_order_is_preserved(self):
        got = resolve_bands(["nir", "red", "swir16"], SENTINEL2)
        assert [b.band_id for b in got] == ["B08", "B04", "B11"]

    def test_empty_rejected(self):
        with pytest.raises(ValidationError, match="at least one band"):
            resolve_bands([], SENTINEL2)

    def test_a_bare_string_is_not_a_sequence_of_bands(self):
        # "red" would otherwise iterate as ['r', 'e', 'd'].
        with pytest.raises(ValidationError, match="expected a sequence of band names"):
            resolve_bands("red", SENTINEL2)

    def test_duplicate_under_the_same_spelling_rejected(self):
        with pytest.raises(ValidationError, match="named twice"):
            resolve_bands(["red", "red"], SENTINEL2)

    def test_duplicate_under_different_spellings_rejected(self):
        # "red" and "B04" are the same band; loading it twice is a mistake.
        with pytest.raises(ValidationError, match="named twice"):
            resolve_bands(["red", "B04"], SENTINEL2)


class TestFinestResolution:
    """The default output grid keeps the finest detail actually requested."""

    def test_ten_metre_bands(self):
        assert finest_resolution(resolve_bands(["red", "nir"], SENTINEL2)) == 10

    def test_mixed_resolutions_take_the_finest(self):
        assert finest_resolution(resolve_bands(["red", "swir16"], SENTINEL2)) == 10

    def test_all_coarse_bands_stay_coarse(self):
        # Nothing here was sensed at 10 m, so upsampling would invent detail.
        assert finest_resolution(resolve_bands(["swir16", "swir22"], SENTINEL2)) == 20
        assert finest_resolution(resolve_bands(["coastal", "nir09"], SENTINEL2)) == 60

    def test_landsat_is_uniform(self):
        assert finest_resolution(resolve_bands(["red", "lwir11"], LANDSAT_OLI)) == 30

    def test_empty_rejected(self):
        with pytest.raises(ValidationError, match="at least one band"):
            finest_resolution([])


class TestSentinel2Files:
    """Band-to-file resolution reads availability from the product."""

    def test_band_written_at_every_resolution(self):
        assert sentinel2_available_resolutions(S2_FILES, SENTINEL2["red"]) == [10, 20, 60]

    def test_band_written_only_at_ten_metres(self):
        assert sentinel2_available_resolutions(S2_FILES, SENTINEL2["nir"]) == [10]

    def test_coarse_band_written_finer_than_it_was_sensed(self):
        # B01 is a 60 m band, but baseline 04.00 onward also writes it at 20 m.
        assert SENTINEL2["coastal"].resolution == 60
        assert sentinel2_available_resolutions(S2_FILES, SENTINEL2["coastal"]) == [20, 60]

    def test_file_is_found_at_a_requested_resolution(self):
        path = sentinel2_band_file(S2_FILES, SENTINEL2["swir16"], 20)
        assert path.endswith("R20m/T32TPS_20240929T100719_B11_20m")

    def test_wrong_resolution_names_the_ones_that_exist(self):
        with pytest.raises(ValidationError) as excinfo:
            sentinel2_band_file(S2_FILES, SENTINEL2["swir16"], 10)
        message = str(excinfo.value)
        assert "not available at 10 m" in message
        assert "20 m, 60 m" in message

    def test_band_absent_from_the_product(self):
        with pytest.raises(ValidationError, match="no 'nir09' \\(B09\\) band at all"):
            sentinel2_band_file(S2_FILES, SENTINEL2["nir09"], 60)


class TestLandsatFiles:
    """Band-to-file resolution reads the metadata's own file list."""

    def test_file_is_found(self):
        assert landsat_band_file(L9_FILES, LANDSAT_OLI["red"]).endswith("_SR_B4.TIF")
        assert landsat_band_file(L9_FILES, LANDSAT_OLI["lwir11"]).endswith("_ST_B10.TIF")

    def test_absent_band_lists_what_the_product_holds(self):
        # An L2SR product carries no thermal band; that is normal, not damage.
        without_thermal = {k: v for k, v in L9_FILES.items() if k != "ST_B10"}
        with pytest.raises(ValidationError) as excinfo:
            landsat_band_file(without_thermal, LANDSAT_OLI["lwir11"])
        assert "no 'lwir11'" in str(excinfo.value)
        assert "SR_B4" in str(excinfo.value)
