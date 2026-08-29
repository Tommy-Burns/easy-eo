"""Tests for reading a product out of the archive it was downloaded as.

The fixtures are the real ones: this module packs the same ``.SAFE`` tree and
the same Landsat product directory the loader tests use, from
:mod:`product_fixtures`, into a zip and a tar. An archive fixture that drifted
from the directory fixture would be asserting that two different products
agree.

The central claim is that the container makes no difference: the same request
against a directory and against the archive of that directory must return the
same values on the same grid. Anything else means discovery and reading
disagree about where the product is.

Nothing is downloaded.
"""

import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pytest
from rasterio.warp import transform_bounds

import eeo
from eeo.core.exceptions import ValidationError
from eeo.io._archive import ArchiveSource, DirectorySource, open_product
from product_fixtures import (
    L9_ID,
    OLI_BANDS,
    build_landsat,
    build_safe,
    pack_tar,
    pack_zip,
)
from product_fixtures import (
    S2_FILL as FILL,
)
from product_fixtures import (
    S2_PRODUCT as PRODUCT,
)


@pytest.fixture
def safe(tmp_path):
    """An unpacked Sentinel-2 product."""
    return build_safe(tmp_path)


@pytest.fixture
def safe_zip(tmp_path, safe):
    """The same product as a Copernicus-style zip, the .SAFE at its root."""
    return pack_zip(safe, tmp_path / "S2B_MSIL2A_20240929T100719.zip", base=safe.parent)


@pytest.fixture
def scene(tmp_path):
    """An unpacked Landsat 9 product."""
    return build_landsat(tmp_path)


@pytest.fixture
def scene_tar(tmp_path, scene):
    """The same product as a USGS-style tar, its files at the tar's root."""
    return pack_tar(scene, tmp_path / f"{L9_ID}.tar", base=scene)


class TestOpenProduct:
    """What a path is taken to be."""

    def test_a_missing_path_names_the_mission(self, tmp_path):
        with pytest.raises(ValidationError, match="no such Sentinel-2 product"):
            open_product(tmp_path / "absent", "Sentinel-2")

    def test_a_directory_becomes_a_directory_source(self, safe):
        source = open_product(safe, "Sentinel-2")
        assert isinstance(source, DirectorySource)
        assert source.named_entry is None
        assert source.name == PRODUCT

    def test_a_named_file_is_remembered(self, safe):
        source = open_product(safe / "MTD_MSIL2A.xml", "Sentinel-2")
        assert isinstance(source, DirectorySource)
        assert source.named_entry == PurePosixPath("MTD_MSIL2A.xml")
        assert source.root == safe

    def test_a_zip_becomes_an_archive_source(self, safe_zip):
        source = open_product(safe_zip, "Sentinel-2")
        assert isinstance(source, ArchiveSource)
        assert source.exists(f"{PRODUCT}/MTD_MSIL2A.xml")

    def test_a_tar_becomes_an_archive_source(self, scene_tar):
        source = open_product(scene_tar, "Landsat")
        assert isinstance(source, ArchiveSource)
        assert source.exists(f"{L9_ID}_MTL.json")

    def test_a_corrupt_archive_is_named(self, tmp_path):
        broken = tmp_path / "broken.zip"
        broken.write_bytes(b"not a zip file at all")
        with pytest.raises(ValidationError, match="not a readable archive"):
            open_product(broken, "Sentinel-2")


class TestCompressedArchivesAreRefused:
    """A gzipped tar is refused with instructions, not read slowly."""

    def test_tar_gz_is_refused(self, tmp_path, scene):
        archive = pack_tar(scene, tmp_path / f"{L9_ID}.tar.gz", base=scene, compress=True)
        with pytest.raises(ValidationError, match="compressed archive"):
            eeo.load_landsat(archive, ["red"])

    def test_the_refusal_says_what_to_do(self, tmp_path, scene):
        archive = pack_tar(scene, tmp_path / f"{L9_ID}.tar.gz", base=scene, compress=True)
        with pytest.raises(ValidationError) as raised:
            eeo.load_landsat(archive, ["red"])
        message = str(raised.value)
        assert "tar -xf" in message
        # The directory the extraction will produce, so the next step is named.
        assert f"'{L9_ID}.tar'" in message

    def test_the_one_piece_spellings_are_refused_too(self, tmp_path, scene):
        archive = pack_tar(scene, tmp_path / f"{L9_ID}.tgz", base=scene, compress=True)
        with pytest.raises(ValidationError, match="compressed archive"):
            open_product(archive, "Landsat")


class TestGlobParity:
    """A directory and its archive answer the same questions the same way."""

    def test_the_same_pattern_finds_the_same_files(self, safe, safe_zip):
        loose = open_product(safe.parent, "Sentinel-2")
        packed = open_product(safe_zip, "Sentinel-2")
        assert loose.glob("*.SAFE/MTD_MSIL2A.xml") == packed.glob("*.SAFE/MTD_MSIL2A.xml")

    def test_a_star_does_not_cross_a_separator(self, safe, safe_zip):
        # The manifest sits one level down, so a root-level pattern must not
        # reach it — in either container.
        loose = open_product(safe.parent, "Sentinel-2")
        packed = open_product(safe_zip, "Sentinel-2")
        assert loose.glob("*.xml") == []
        assert packed.glob("*.xml") == []

    def test_existence_agrees(self, safe, safe_zip):
        loose = open_product(safe, "Sentinel-2")
        packed = open_product(safe_zip, "Sentinel-2")
        assert loose.exists("MTD_MSIL2A.xml")
        assert packed.exists(f"{PRODUCT}/MTD_MSIL2A.xml")
        assert not loose.exists("MTD_MSIL1C.xml")
        assert not packed.exists(f"{PRODUCT}/MTD_MSIL1C.xml")


class TestHref:
    """What each source hands to rasterio."""

    def test_a_directory_gives_a_plain_path(self, safe):
        source = open_product(safe, "Sentinel-2")
        assert source.href("MTD_MSIL2A.xml") == str(safe / "MTD_MSIL2A.xml")

    def test_a_zip_gives_a_vsizip_path(self, safe_zip):
        source = open_product(safe_zip, "Sentinel-2")
        href = source.href(f"{PRODUCT}/MTD_MSIL2A.xml")
        assert href == f"/vsizip/{safe_zip.as_posix()}/{PRODUCT}/MTD_MSIL2A.xml"

    def test_a_tar_gives_a_vsitar_path(self, scene_tar):
        source = open_product(scene_tar, "Landsat")
        href = source.href(f"{L9_ID}_SR_B4.TIF")
        assert href == f"/vsitar/{scene_tar.as_posix()}/{L9_ID}_SR_B4.TIF"


class TestSentinel2FromAZip:
    """A Copernicus zip loads without being unpacked."""

    def test_bands_load(self, safe_zip):
        scene = eeo.load_sentinel2(safe_zip, ["red", "nir"])
        assert scene.band_names == ["red", "nir"]
        assert scene.to_array()[0][0, 0] == FILL["B04"]
        assert scene.to_array()[1][0, 0] == FILL["B08"]

    def test_the_zip_matches_the_directory_exactly(self, safe, safe_zip):
        loose = eeo.load_sentinel2(safe, ["red", "swir16"])
        packed = eeo.load_sentinel2(safe_zip, ["red", "swir16"])
        np.testing.assert_array_equal(loose.to_array(), packed.to_array())
        assert loose.get_transform() == packed.get_transform()
        assert loose.get_crs() == packed.get_crs()

    def test_provenance_names_the_product_not_the_zip(self, safe_zip):
        # The .SAFE inside is what identifies the scene; the zip is packaging.
        assert eeo.load_sentinel2(safe_zip, ["red"]).attrs["product"] == PRODUCT

    def test_a_zip_holding_no_product_is_refused(self, tmp_path):
        empty = tmp_path / "empty.zip"
        with zipfile.ZipFile(empty, "w") as zipped:
            zipped.writestr("readme.txt", "nothing here")
        with pytest.raises(ValidationError, match="holds no Sentinel-2 product manifest"):
            eeo.load_sentinel2(empty, ["red"])


class TestLandsatFromAnArchive:
    """A USGS tar loads without being unpacked, and so does a zip of one."""

    def test_bands_load_from_a_tar(self, scene_tar):
        loaded = eeo.load_landsat(scene_tar, ["red", "nir08"])
        assert loaded.band_names == ["red", "nir08"]
        assert loaded.to_array()[0].max() == OLI_BANDS["SR_B4"]
        assert loaded.to_array()[1].max() == OLI_BANDS["SR_B5"]

    def test_the_tar_matches_the_directory_exactly(self, scene, scene_tar):
        loose = eeo.load_landsat(scene, ["red", "lwir11"])
        packed = eeo.load_landsat(scene_tar, ["red", "lwir11"])
        np.testing.assert_array_equal(loose.to_array(), packed.to_array())
        assert loose.get_transform() == packed.get_transform()
        assert loose.attrs["product"] == packed.attrs["product"]

    def test_nodata_still_comes_from_a_data_band(self, scene_tar):
        loaded = eeo.load_landsat(scene_tar, ["qa_pixel", "red"])
        assert loaded.get_metadata()["nodata"] == 0

    def test_a_zipped_landsat_product_also_loads(self, tmp_path, scene):
        # The container is orthogonal to the mission: nothing about the zip
        # path is Sentinel-2 specific, and nothing about tar is Landsat's.
        archive = pack_zip(scene, tmp_path / f"{L9_ID}.zip", base=scene.parent)
        assert eeo.load_landsat(archive, ["red"]).band_names == ["red"]

    def test_a_bbox_still_crops_inside_an_archive(self, scene, scene_tar):
        bounds = eeo.load_landsat(scene, ["red"]).get_bounds()
        lonlat = transform_bounds(
            "EPSG:32633",
            "EPSG:4326",
            bounds.left,
            (bounds.bottom + bounds.top) / 2,
            (bounds.left + bounds.right) / 2,
            bounds.top,
            densify_pts=21,
        )
        cropped = eeo.load_landsat(scene_tar, ["red"], bbox=lonlat)
        height, width = cropped.to_array().shape[-2:]
        assert height < 64 and width < 64


class TestNamedFilesInsideAProduct:
    """Pointing at one file still selects that file."""

    def test_a_named_manifest_is_used(self, safe):
        loaded = eeo.load_sentinel2(safe / "MTD_MSIL2A.xml", ["red"])
        assert loaded.band_names == ["red"]

    def test_a_named_metadata_file_is_used(self, scene):
        metadata = Path(scene) / f"{L9_ID}_MTL.json"
        assert eeo.load_landsat(metadata, ["red"]).band_names == ["red"]


class TestSourceReadFailures:
    """What a source does when asked for something it cannot give."""

    def test_a_directory_read_of_a_directory_is_reported(self, safe):
        source = open_product(safe, "Sentinel-2")
        with pytest.raises(ValidationError, match="could not read GRANULE"):
            source.read_bytes("GRANULE")

    def test_an_absent_zip_member_is_reported(self, safe_zip):
        source = open_product(safe_zip, "Sentinel-2")
        with pytest.raises(ValidationError, match="could not read absent.xml"):
            source.read_bytes("absent.xml")

    def test_a_tar_member_that_is_not_a_file_is_reported(self, tmp_path, scene):
        # Directories are tar members too, and are not listed as files, so
        # asking for one by name has to fail rather than return empty bytes.
        archive = tmp_path / "with_dirs.tar"
        with tarfile.open(archive, "w") as tarred:
            tarred.add(scene, arcname=L9_ID, recursive=False)
        source = open_product(archive, "Landsat")
        with pytest.raises(ValidationError, match="not a file in"):
            source.read_bytes(L9_ID)


class TestGlobParityForSingleCharacters:
    """`?` behaves the same in an archive as on disk, like the rest of glob."""

    def test_a_question_mark_matches_one_character(self, safe, safe_zip):
        loose = open_product(safe.parent, "Sentinel-2")
        packed = open_product(safe_zip, "Sentinel-2")
        pattern = f"{PRODUCT[:-1]}?/MTD_MSIL2A.xml"
        assert loose.glob(pattern) == packed.glob(pattern)
        assert loose.glob(pattern) == [PurePosixPath(PRODUCT) / "MTD_MSIL2A.xml"]

    def test_a_question_mark_does_not_match_a_separator(self, safe, safe_zip):
        loose = open_product(safe.parent, "Sentinel-2")
        packed = open_product(safe_zip, "Sentinel-2")
        assert loose.glob(f"{PRODUCT}?MTD_MSIL2A.xml") == []
        assert packed.glob(f"{PRODUCT}?MTD_MSIL2A.xml") == []
