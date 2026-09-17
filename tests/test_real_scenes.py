"""Easy-EO checked against real downloaded products.

Two things are verified here, both of which synthetic rasters cannot reach.

**Quality-layer decoding.** The synthetic tests in ``test_quality_masks.py``
and ``test_qa_pixel.py`` pin the tables to what ESA and USGS publish. These
check the same code against what the agencies actually ship, which is a
different question: a table can be transcribed correctly and still be read
against the wrong band, the wrong dtype, or a grid the loader resampled on the
way in.

**Block-wise execution.** The engine's claim is that splitting a raster into
blocks changes nothing about the answer. Synthetic tests make that claim over
six-pixel rasters with tidy nodata; a real scene tests it over millions of
pixels whose off-swath fill runs diagonally across every block seam, read
through the formats and internal tiling the agencies actually ship — JP2 in
1024-pixel tiles for Sentinel-2, a 256-pixel-tiled GeoTIFF for Landsat.

The checks are invariants rather than fixed numbers, because they must hold
for *any* scene the maintainer happens to have, not just for one. The
strongest of the quality ones is that each single-bit Landsat flag must equal
its own confidence field reading High: that one statement exercises all eight
bit positions and all four two-bit field offsets against data neither we nor
the test author wrote. The strongest of the block-wise ones needs no expected
values at all — it recomputes the same NDVI eagerly and demands the two agree
bit for bit.

**The lazy backend.** The xarray adapter must report the same metadata and
return the same pixels as rasterio for the same file. Synthetic GeoTIFFs cannot
show that for a JP2 decoded through ``/vsizip/``, or for a real product's nodata
and internal tiling, and those are what a user will point it at.

Every check that applies to both missions runs on both products.

Opt in with ``--run-realdata``, pointing the environment variables at
downloaded products::

    EEO_TEST_SENTINEL2_SCENE=~/Downloads/sen/S2A_....SAFE.zip \\
    EEO_TEST_LANDSAT_SCENE=~/Downloads/ls/LC09_....tar \\
    pytest tests/test_real_scenes.py --run-realdata

Nothing here downloads anything; the products must already be on disk.
"""

import numpy as np
import pytest

import eeo
from eeo.core.blockwise import (
    BlockSource,
    apply_blockwise,
    block_windows,
    resolve_block_shape,
)
from eeo.core.exceptions import ValidationError
from eeo.preprocessing.quality import (
    QAConfidence,
    QAConfidenceField,
    QAPixelFlag,
    SCLClass,
    qa_pixel_confidence,
    qa_pixel_flag,
    qa_pixel_mask,
    scl_mask,
)

pytestmark = pytest.mark.realdata

# Flag / confidence-field pairs that describe the same phenomenon. USGS
# defines each flag as "its confidence is High", so the two must agree
# everywhere, on every mission that has them.
FLAG_AND_FIELD = (
    (QAPixelFlag.CLOUD, QAConfidenceField.CLOUD),
    (QAPixelFlag.CLOUD_SHADOW, QAConfidenceField.CLOUD_SHADOW),
    (QAPixelFlag.SNOW, QAConfidenceField.SNOW_ICE),
    (QAPixelFlag.CIRRUS, QAConfidenceField.CIRRUS),
)

# The three fields for which USGS documents the value 2 as Reserved. A real
# product must therefore never contain it there.
FIELDS_RESERVING_TWO = (
    QAConfidenceField.CLOUD_SHADOW,
    QAConfidenceField.SNOW_ICE,
    QAConfidenceField.CIRRUS,
)


@pytest.fixture(scope="module")
def scl(sentinel2_scene):
    """The ``SCL`` band of the real Sentinel-2 product, read once."""
    return eeo.load_sentinel2(str(sentinel2_scene), bands=["scl"]).read()[0]


@pytest.fixture(scope="module")
def landsat(landsat_scene):
    """The ``QA_PIXEL`` band of the real Landsat product, and its mission.

    The mission is taken from the loaded dataset's own ``attrs`` rather than
    re-parsed from the product, because that is the route a chainable masking
    op has to it. It is recorded there as prose — ``"Landsat 9"`` — so the
    number a decoder needs has to be dug out of it.
    """
    ds = eeo.load_landsat(str(landsat_scene), bands=["qa_pixel"])
    mission = int(ds.attrs["mission"].rsplit(maxsplit=1)[-1])
    return ds.read()[0], mission


class TestRealSentinel2SCL:
    def test_every_value_present_is_a_documented_class(self, scl):
        # An undocumented class number would mean the enumeration is
        # incomplete, which no synthetic test can discover.
        present = set(np.unique(scl).tolist())
        assert present <= {member.value for member in SCLClass}

    def test_the_band_is_an_integer_class_map(self, scl):
        # Class numbers must arrive unmodified; a resampled SCL would show
        # fractional values and mean nothing.
        assert np.issubdtype(scl.dtype, np.integer)

    def test_the_scene_is_classified_rather_than_empty(self, scl):
        # Guards against silently reading the wrong asset: a scene of one
        # single class is a band we misidentified, not a classification.
        assert len(np.unique(scl)) > 1

    def test_the_mask_is_exactly_the_union_of_its_classes(self, scl):
        # np.isin over many classes must agree with the classes counted one
        # at a time, on a real class distribution rather than a crafted one.
        classes = [SCLClass.CLOUD_SHADOWS, SCLClass.CLOUD_HIGH_PROBABILITY]
        mask = scl_mask(scl, classes=classes)
        assert int(mask.sum()) == sum(int((scl == c).sum()) for c in classes)

    def test_masking_more_classes_never_masks_fewer_pixels(self, scl):
        few = scl_mask(scl, classes=[SCLClass.CLOUD_HIGH_PROBABILITY])
        many = scl_mask(scl, classes=[SCLClass.CLOUD_HIGH_PROBABILITY, SCLClass.THIN_CIRRUS])
        assert int(many.sum()) >= int(few.sum())
        assert (many | few == many).all()

    @pytest.mark.parametrize(("older", "current"), [("bare_soil", 5), ("dark_features", 2)])
    def test_a_renamed_class_resolves_to_the_same_real_pixels(self, scl, older, current):
        assert int(scl_mask(scl, classes=[older]).sum()) == int((scl == current).sum())

    def test_a_float_copy_decodes_identically(self, scl):
        # Stacking SCL beside float reflectance widens it; the decode must
        # survive that on real values, not just on hand-built ones.
        assert np.array_equal(scl_mask(scl.astype("float32")), scl_mask(scl))


class TestRealLandsatQAPixel:
    def test_the_band_is_an_integer_bit_field(self, landsat):
        qa, _ = landsat
        assert np.issubdtype(qa.dtype, np.integer)

    @pytest.mark.parametrize(("flag", "field"), FLAG_AND_FIELD)
    def test_each_flag_equals_its_confidence_field_reading_high(self, landsat, flag, field):
        # The strongest available check: it pins every bit position and every
        # two-bit field offset simultaneously, against agency-produced data.
        qa, mission = landsat
        if flag not in eeo.qa_pixel_flags(mission):
            pytest.skip(f"Landsat {mission} leaves {flag.name.lower()} Unused")
        by_flag = qa_pixel_flag(qa, flag, mission=mission)
        by_confidence = qa_pixel_confidence(qa, field, mission=mission) == QAConfidence.HIGH
        assert np.array_equal(by_flag, by_confidence)

    def test_clear_is_the_absence_of_cloud_and_dilated_cloud(self, landsat):
        # USGS derives bit 6 from bits 1 and 3. Fill pixels are exempt: every
        # bit is zero there, so 'clear' reads 0 where the derivation says 1.
        qa, mission = landsat
        clear = qa_pixel_flag(qa, QAPixelFlag.CLEAR, mission=mission)
        cloud = qa_pixel_flag(qa, QAPixelFlag.CLOUD, mission=mission)
        dilated = qa_pixel_flag(qa, QAPixelFlag.DILATED_CLOUD, mission=mission)
        fill = qa_pixel_flag(qa, QAPixelFlag.FILL, mission=mission)
        assert ((clear == ~(cloud | dilated)) | fill).all()

    @pytest.mark.parametrize("field", FIELDS_RESERVING_TWO)
    def test_a_reserved_confidence_value_never_appears(self, landsat, field):
        # If 2 turned up here, treating these fields as a three-level scale
        # would be defensible after all. It does not.
        qa, mission = landsat
        if field not in eeo.qa_pixel_confidence_fields(mission):
            pytest.skip(f"Landsat {mission} leaves {field.name.lower()} confidence Unused")
        assert 2 not in np.unique(qa_pixel_confidence(qa, field, mission=mission)).tolist()

    def test_cloud_confidence_does_use_the_medium_value(self, landsat):
        # The other half of the asymmetry: 2 means Medium here, and real
        # scenes contain it. Without this, the parametrized test above would
        # pass just as well if our field offsets were all wrong.
        qa, mission = landsat
        present = np.unique(qa_pixel_confidence(qa, QAConfidenceField.CLOUD, mission=mission))
        assert QAConfidence.MEDIUM in present.tolist()

    def test_medium_confidence_cloud_carries_no_flag_of_its_own(self, landsat):
        # The premise of the default threshold, on real data: these pixels
        # exist and the single-bit flags do not catch them.
        qa, mission = landsat
        medium = qa_pixel_confidence(qa, QAConfidenceField.CLOUD, mission=mission) == (
            QAConfidence.MEDIUM
        )
        assert medium.any(), "scene has no medium-confidence cloud to test with"
        assert not qa_pixel_flag(qa, QAPixelFlag.CLOUD, mission=mission)[medium].any()

    def test_the_default_mask_catches_what_the_flags_miss(self, landsat):
        qa, mission = landsat
        default = qa_pixel_mask(qa, mission=mission)
        flags_only = qa_pixel_mask(qa, mission=mission, min_cloud_confidence=None)
        assert (default | flags_only == default).all(), "the default must never mask less"
        assert int(default.sum()) > int(flags_only.sum()), "no mid-confidence cloud was caught"

    def test_a_high_threshold_reproduces_the_flags(self, landsat):
        qa, mission = landsat
        by_threshold = qa_pixel_mask(qa, mission=mission, min_cloud_confidence=QAConfidence.HIGH)
        by_flag = qa_pixel_mask(qa, mission=mission, min_cloud_confidence=None)
        assert np.array_equal(by_threshold, by_flag)

    def test_fill_marks_the_scene_corners_and_nothing_masks_more(self, landsat):
        # A north-up grid over a rotated WRS-2 footprint always has fill, and
        # the default mask must be a superset of it.
        qa, mission = landsat
        fill = qa_pixel_flag(qa, QAPixelFlag.FILL, mission=mission)
        assert fill.any(), "a Landsat scene on a north-up grid must have fill pixels"
        assert (qa_pixel_mask(qa, mission=mission) | fill).sum() == int(
            qa_pixel_mask(qa, mission=mission).sum()
        )

    def test_a_float_copy_decodes_identically(self, landsat):
        qa, mission = landsat
        assert np.array_equal(
            qa_pixel_mask(qa.astype("float32"), mission=mission),
            qa_pixel_mask(qa, mission=mission),
        )

    def test_reading_a_cirrus_bit_the_sensor_lacks_still_raises(self, landsat):
        qa, mission = landsat
        if mission in (8, 9):
            pytest.skip("this mission has a cirrus band")
        with pytest.raises(ValidationError, match="Unused"):
            qa_pixel_flag(qa, QAPixelFlag.CIRRUS, mission=mission)


class TestMaskCloudsOnRealScenes:
    """The whole point of WP-26, end to end on agency data."""

    def test_landsat_masks_and_records_its_nodata(self, landsat_scene):
        ds = eeo.load_landsat(str(landsat_scene), bands=["red", "nir08", "qa_pixel"])
        declared = ds.get_metadata()["nodata"]
        assert declared is not None, "Landsat C2 declares 0 as fill"
        out = ds.mask_clouds()
        assert out.get_metadata()["nodata"] == declared
        assert out.get_metadata()["dtype"] == ds.get_metadata()["dtype"]
        assert out.band_names == ds.band_names
        assert out.get_transform() == ds.get_transform()

    def test_landsat_masking_reaches_every_band(self, landsat_scene):
        ds = eeo.load_landsat(str(landsat_scene), bands=["red", "nir08", "qa_pixel"])
        out = ds.mask_clouds().read()
        fill = out[0] == 0
        assert ((out[1] == 0) == fill).all()
        assert fill.any() and not fill.all()

    def test_landsat_masking_removes_more_than_the_scene_already_lacked(self, landsat_scene):
        ds = eeo.load_landsat(str(landsat_scene), bands=["red", "qa_pixel"])
        before = ds.read()[0]
        after = ds.mask_clouds().read()[0]
        assert int((after == 0).sum()) > int((before == 0).sum()), "nothing was masked"

    def test_an_index_over_a_masked_scene_excludes_the_masked_pixels(self, landsat_scene):
        # WP-26's "Done when": ndvi over the masked result excludes the
        # flagged pixels. Asserted on a real cloudy scene.
        ds = eeo.load_landsat(str(landsat_scene), bands=["red", "nir08", "qa_pixel"])
        masked = ds.mask_clouds()
        nodata_pixels = masked.read()[0] == 0
        ndvi = masked.ndvi("red", nir="nir08").read()[0]
        assert np.isnan(ndvi)[nodata_pixels].all(), "a masked pixel reached the index"
        assert np.isfinite(ndvi).any(), "the whole scene was masked"

    def test_sentinel2_carries_the_fill_value_esa_declares(self, sentinel2_scene):
        # The JP2s declare no nodata; the manifest does. A real product is the
        # only place that distinction shows up — the fixture cannot prove that
        # ESA really states it this way.
        ds = eeo.load_sentinel2(str(sentinel2_scene), bands=["red", "scl"])
        assert ds.get_metadata()["nodata"] == 0

    def test_sentinel2_masks_without_being_told_the_fill_value(self, sentinel2_scene):
        ds = eeo.load_sentinel2(str(sentinel2_scene), bands=["red", "scl"])
        out = ds.mask_clouds()
        assert out.get_metadata()["nodata"] == 0
        assert out.band_names == ds.band_names

    def test_both_missions_agree_on_the_fill_value(self, sentinel2_scene, landsat_scene):
        # The contract WP-25 states: a workflow does not care which route or
        # which mission the data came from.
        s2 = eeo.load_sentinel2(str(sentinel2_scene), bands=["red"])
        ls = eeo.load_landsat(str(landsat_scene), bands=["red"])
        assert s2.get_metadata()["nodata"] == ls.get_metadata()["nodata"] == 0

    def test_sentinel2_masks_exactly_the_flagged_classes(self, sentinel2_scene):
        ds = eeo.load_sentinel2(str(sentinel2_scene), bands=["red", "scl"])
        scl = ds.read()[1]
        expected = scl_mask(scl)
        after = ds.mask_clouds(nodata=0).read()[0]
        # Every flagged pixel is nodata. The converse need not hold: a real
        # reflectance band already contains zeros of its own.
        assert (after[expected] == 0).all()

    def test_the_quality_band_is_upsampled_by_nearest_neighbour(self, sentinel2_scene):
        # SCL is a 20 m band; asking for it beside 10 m bands resamples it.
        # An interpolating method would invent class numbers, which is the
        # one thing that silently destroys a mask.
        ds = eeo.load_sentinel2(str(sentinel2_scene), bands=["red", "scl"])
        upsampled = ds.read()[1]
        assert ds.attrs["resolution"] == 10, "expected the 10 m grid"
        assert set(np.unique(upsampled).tolist()) <= {m.value for m in SCLClass}


class TestClearFractionOnRealScenes:
    """The number 20.2's composite will select scenes on."""

    def test_landsat_clear_fraction_falls_after_masking(self, landsat_scene):
        ds = eeo.load_landsat(str(landsat_scene), bands=["red", "qa_pixel"])
        before = ds.clear_fraction()
        after = ds.mask_clouds().clear_fraction()
        assert 0.0 < after < before < 1.0, "a real scene has both fill and cloud"

    def test_landsat_clear_fraction_matches_the_mask_it_was_told_about(self, landsat_scene):
        # The fraction must be the complement of what mask_clouds actually
        # wrote, not an independently drifting number.
        ds = eeo.load_landsat(str(landsat_scene), bands=["red", "qa_pixel"])
        masked = ds.mask_clouds()
        nodata_pixels = (masked.read() == 0).any(axis=0)
        assert masked.clear_fraction() == pytest.approx(1.0 - nodata_pixels.mean())

    def test_scene_edge_fill_dominates_the_whole_scene_figure(self, landsat_scene):
        # Why the docstring warns against reading a whole-scene clear fraction
        # as cloudiness: a north-up grid over a rotated WRS-2 footprint is
        # heavily fill before any cloud is masked.
        ds = eeo.load_landsat(str(landsat_scene), bands=["red"])
        assert ds.clear_fraction() < 0.9

    def test_sentinel2_clear_fraction_is_high_on_a_clear_tile(self, sentinel2_scene):
        ds = eeo.load_sentinel2(str(sentinel2_scene), bands=["red", "scl"])
        assert ds.mask_clouds().clear_fraction() > 0.9


# ---------------------------------------------------------------------------
# Block-wise execution
# ---------------------------------------------------------------------------


def _ndvi(nir, red):
    """NDVI over a pair of blocks, as a pixel-wise function of them alone.

    Written once and handed to both the eager and the block-wise run, so the
    only thing that differs between the two is whether it saw the scene whole
    or a strip at a time.
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    total = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total != 0, (nir - red) / total, np.float32(0))


def _eager_ndvi(ds):
    """NDVI over the whole scene at once, with the nodata contract applied."""
    red, nir = ds.read(1), ds.read(2)
    result = _ndvi(nir, red).astype(np.float32)
    nodata = ds.get_metadata()["nodata"]
    if nodata is None:
        return result
    invalid = (red == nodata) | (nir == nodata)
    return np.where(invalid, np.float32(np.nan), result)


def _blockwise_ndvi(ds, **kwargs):
    """NDVI over the same scene a block at a time, through the engine.

    The caller owns the result and must close it: on a full Landsat scene it
    is a quarter of a gigabyte of in-memory raster, and this module's tests
    would otherwise accumulate one per test.
    """
    return apply_blockwise(
        ds,
        _ndvi,
        sources=[BlockSource.from_dataset(ds, band=2), BlockSource.from_dataset(ds, band=1)],
        fractional=True,
        **kwargs,
    )


@pytest.fixture(scope="module")
def sentinel2_red_nir(sentinel2_scene):
    """Real Sentinel-2 red and NIR at 60 m, the cheapest genuine full tile.

    60 m is the coarsest resolution the product carries, which keeps the tile
    to 1830x1830 while still decoding real JP2 imagery over the real footprint.
    """
    return eeo.load_sentinel2(str(sentinel2_scene), bands=["red", "nir"], resolution=60)


@pytest.fixture(scope="module")
def landsat_red_nir(landsat_scene):
    """Real Landsat red and NIR over the whole scene.

    Loaded whole rather than through a ``bbox``, because the point is the
    off-swath fill: a Landsat scene is rotated inside its bounding box, so
    roughly a third of the grid is nodata arranged as four diagonal corners
    that no block seam can avoid crossing.
    """
    return eeo.load_landsat(str(landsat_scene), bands=["red", "nir08"])


@pytest.fixture(scope="module")
def sentinel2_ndvi(sentinel2_red_nir):
    """Block-wise NDVI over the Sentinel-2 tile, computed once for the module."""
    result = _blockwise_ndvi(sentinel2_red_nir)
    yield result
    result.close()


@pytest.fixture(scope="module")
def landsat_ndvi(landsat_red_nir):
    """Block-wise NDVI over the Landsat scene, computed once for the module."""
    result = _blockwise_ndvi(landsat_red_nir)
    yield result
    result.close()


#: ``(scene fixture, block-wise NDVI fixture)`` for each real product, so the
#: checks that hold for both missions are stated once.
BOTH_SCENES = [
    pytest.param("sentinel2_red_nir", "sentinel2_ndvi", id="sentinel2"),
    pytest.param("landsat_red_nir", "landsat_ndvi", id="landsat"),
]


#: The scene fixture of each real product, for checks that need no NDVI.
BOTH_SCENE_FIXTURES = [
    pytest.param("sentinel2_red_nir", id="sentinel2"),
    pytest.param("landsat_red_nir", id="landsat"),
]


class TestRealSceneBlockwise:
    @pytest.mark.parametrize(("scene", "ndvi"), BOTH_SCENES)
    def test_the_scene_needs_more_than_one_block(self, scene, ndvi, request):
        # Without this the comparisons below could pass on a scene that fits
        # in a single block, which would test nothing about seams.
        ds = request.getfixturevalue(scene)
        shape = ds.get_shape()
        assert len(list(block_windows(shape, resolve_block_shape(shape)))) > 1

    @pytest.mark.parametrize(("scene", "ndvi"), BOTH_SCENES)
    def test_blocking_changes_nothing(self, scene, ndvi, request):
        # The central invariant, and it needs no expected values: the same
        # arithmetic over the same pixels must not care how they were grouped.
        ds = request.getfixturevalue(scene)
        blocked = request.getfixturevalue(ndvi)
        assert np.array_equal(blocked.read()[0], _eager_ndvi(ds), equal_nan=True)

    @pytest.mark.parametrize(("scene", "ndvi"), BOTH_SCENES)
    def test_the_result_is_float32_on_the_scene_s_own_grid(self, scene, ndvi, request):
        ds = request.getfixturevalue(scene)
        blocked = request.getfixturevalue(ndvi)
        assert blocked.read().dtype == np.float32
        assert blocked.get_count() == 1
        assert blocked.get_shape() == ds.get_shape()
        assert blocked.get_crs() == ds.get_crs()
        assert blocked.get_transform() == ds.get_transform()

    @pytest.mark.parametrize(("scene", "ndvi"), BOTH_SCENES)
    def test_the_output_driver_is_chosen_not_inherited(self, scene, ndvi, request):
        # The source driver records how the scene was read; the output is a
        # GTiff whichever format it came from.
        assert request.getfixturevalue(ndvi).get_metadata()["driver"] == "GTiff"

    @pytest.mark.parametrize(("scene", "ndvi"), BOTH_SCENES)
    def test_the_nodata_mask_is_exactly_the_bands_fill(self, scene, ndvi, request):
        # Nodata is contagious, so the invalid pixels of the result must be
        # precisely the union of the two bands' fill — no seam may round the
        # boundary, and no interior pixel may be dropped. This compares two
        # masks of tens of millions of real pixels against each other, and it
        # holds whatever the scene's fill happens to look like.
        ds = request.getfixturevalue(scene)
        blocked = request.getfixturevalue(ndvi)
        fill = ds.get_metadata()["nodata"]
        assert fill is not None

        expected = (ds.read(1) == fill) | (ds.read(2) == fill)
        assert np.isnan(blocked.get_metadata()["nodata"])
        assert np.array_equal(np.isnan(blocked.read()[0]), expected)

    def test_the_landsat_scene_really_does_straddle_its_swath_edge(self, landsat_red_nir):
        # Keeps the check above from passing vacuously on the mission whose
        # fill is the interesting case: a Landsat scene is rotated inside its
        # bounding box, so a large minority of the grid is off-swath fill
        # arranged as four diagonal corners no block seam can avoid crossing.
        ds = landsat_red_nir
        fill = (ds.read(1) == ds.get_metadata()["nodata"]).mean()
        assert 0.05 < float(fill) < 0.95

    @pytest.mark.parametrize(("scene", "ndvi"), BOTH_SCENES)
    def test_streaming_to_disk_gives_the_same_raster(self, scene, ndvi, request, tmp_path):
        # The route that keeps peak memory bounded by the block rather than by
        # the output, which is the only one a larger-than-memory result can use.
        path = tmp_path / "ndvi.tif"
        streamed = _blockwise_ndvi(request.getfixturevalue(scene), save_path=path)
        try:
            assert path.exists()
            assert np.array_equal(
                streamed.read(), request.getfixturevalue(ndvi).read(), equal_nan=True
            )
        finally:
            streamed.close()

    @pytest.mark.parametrize(("scene", "ndvi"), BOTH_SCENES)
    def test_the_ndvi_op_matches_the_engine_called_by_hand(self, scene, ndvi, request):
        # Task 16.2 routed the spectral indices through the engine, so the
        # chainable op and a hand-built apply_blockwise over the same bands are
        # now the same computation and must agree on a real scene.
        ds = request.getfixturevalue(scene)
        nir_name = ds.band_names[1]
        op_result = ds.ndvi("red", nir=nir_name)
        try:
            assert np.array_equal(
                op_result.read(), request.getfixturevalue(ndvi).read(), equal_nan=True
            )
        finally:
            op_result.close()

    @pytest.mark.parametrize(("scene", "ndvi"), BOTH_SCENES)
    def test_the_ndvi_op_matches_the_eager_reference(self, scene, ndvi, request):
        # And against arithmetic written out longhand, which shares no code
        # with the engine at all.
        ds = request.getfixturevalue(scene)
        op_result = ds.ndvi("red", nir=ds.band_names[1])
        try:
            assert np.array_equal(op_result.read()[0], _eager_ndvi(ds), equal_nan=True)
            assert op_result.read().dtype == np.float32
            assert np.isnan(op_result.get_metadata()["nodata"])
        finally:
            op_result.close()

    @pytest.mark.parametrize("scene", BOTH_SCENE_FIXTURES)
    def test_algebra_on_a_real_scene_keeps_its_fill(self, scene, request):
        # The plain algebra path, on real fill (a third of the Landsat grid).
        # Stated as "fill stays fill" rather than "the fill mask is unchanged",
        # because the two are not the same claim on a uint16 scene: doubling
        # wraps, so a valid pixel of exactly 32768 lands on 0 — the fill value
        # — of its own accord. The Landsat scene contains two of them. That is
        # the documented integer behaviour, not a masking failure.
        ds = request.getfixturevalue(scene)
        fill = ds.get_metadata()["nodata"]
        source = ds.read()
        was_fill = source == fill
        doubled = ds.multiply(2)
        try:
            assert doubled.read().dtype == source.dtype
            assert (doubled.read()[was_fill] == fill).all()
            assert np.array_equal(doubled.read()[~was_fill], (source * 2)[~was_fill])
        finally:
            doubled.close()

    @pytest.mark.parametrize("scene", BOTH_SCENE_FIXTURES)
    def test_a_fractional_op_on_a_real_scene_cannot_wrap(self, scene, request):
        # The float32 route past that wrap: divide is a fractional-result op,
        # so the fill mask of the result is exactly the input's fill.
        ds = request.getfixturevalue(scene)
        was_fill = ds.read() == ds.get_metadata()["nodata"]
        halved = ds.divide(2)
        try:
            assert halved.read().dtype == np.float32
            assert np.array_equal(np.isnan(halved.read()), was_fill)
        finally:
            halved.close()

    @pytest.mark.parametrize(("scene", "ndvi"), BOTH_SCENES)
    @pytest.mark.parametrize("block_shape", ["strip", (997, 503), (371, 371)])
    def test_the_block_shape_does_not_change_the_answer(self, scene, ndvi, block_shape, request):
        # Shapes that divide both scenes (1830x1830 and 8081x7991) unevenly, so
        # the final row and column of blocks are truncated, plus full-width
        # strips of a single row.
        ds = request.getfixturevalue(scene)
        if block_shape == "strip":
            block_shape = (1, ds.get_width())
        other = _blockwise_ndvi(ds, block_shape=block_shape)
        try:
            assert np.array_equal(
                other.read(), request.getfixturevalue(ndvi).read(), equal_nan=True
            )
        finally:
            other.close()


class TestRealSceneStreamingStatistics:
    """Global statistics over a real scene, against the whole-array answer.

    These are the two-pass ops from 16.3. On the Landsat scene the reference
    arrays are a few hundred megabytes, which is the point: the streamed form
    never holds one, and the comparison only exists to prove it did not need to.
    """

    @pytest.mark.parametrize("scene", BOTH_SCENE_FIXTURES)
    def test_the_streamed_range_is_exact(self, scene, request):
        from eeo.core.streaming import valid_min_max

        ds = request.getfixturevalue(scene)

        band = ds.read(1).astype(np.float64)
        band[band == ds.get_metadata()["nodata"]] = np.nan
        assert valid_min_max(ds, 1) == (
            float(np.nanmin(band)),
            float(np.nanmax(band)),
        )

    @pytest.mark.parametrize("scene", BOTH_SCENE_FIXTURES)
    def test_the_streamed_mean_and_deviation_match(self, scene, request):
        from eeo.core.streaming import valid_mean_std

        ds = request.getfixturevalue(scene)

        band = ds.read(1).astype(np.float64)
        band[band == ds.get_metadata()["nodata"]] = np.nan
        mean, std = valid_mean_std(ds, 1)
        # Up to 40 million valid pixels of four-digit reflectance: the sum of
        # squares would be past float64's significant digits, which is what
        # Chan's parallel update avoids.
        assert mean == pytest.approx(float(np.nanmean(band)), rel=1e-12)
        assert std == pytest.approx(float(np.nanstd(band)), rel=1e-12)

    @pytest.mark.parametrize("scene", BOTH_SCENE_FIXTURES)
    def test_the_streamed_percentiles_are_exact_on_an_integer_scene(self, scene, request):
        from eeo.core.streaming import valid_percentiles

        # Both missions ship integer imagery, so both take the histogram path
        # and the answer must be exact, not close.
        ds = request.getfixturevalue(scene)
        assert np.issubdtype(np.dtype(ds.get_metadata()["dtype"]), np.integer)
        band = ds.read(1).astype(np.float64)
        band[band == ds.get_metadata()["nodata"]] = np.nan

        wanted = [2, 50, 98]
        assert valid_percentiles(ds, wanted, 1) == pytest.approx(
            list(np.nanpercentile(band, wanted)), rel=0, abs=0
        )

    @pytest.mark.parametrize("scene", BOTH_SCENE_FIXTURES)
    def test_the_brightest_pixel_is_the_one_numpy_finds(self, scene, request):
        ds = request.getfixturevalue(scene)
        band = ds.read(1).astype(np.float64)
        band[band == ds.get_metadata()["nodata"]] = np.nan
        peak = ds.get_maximum_pixel(return_position_as_pixel_coordinate=True)
        row, col = np.unravel_index(int(np.nanargmax(band)), band.shape)
        assert peak["value"] == float(np.nanmax(band))
        assert peak["position"] == (int(row), int(col))

    @pytest.mark.parametrize("scene", BOTH_SCENE_FIXTURES)
    def test_the_darkest_pixel_is_found_despite_fill_sharing_its_value(self, scene, request):
        ds = request.getfixturevalue(scene)
        # Both missions' fill is 0, the smallest a uint16 can be, so the
        # minimum search has to exclude fill rather than merely order values.
        floor = ds.get_minimum_pixel()
        assert floor["value"] > 0

    @pytest.mark.parametrize("scene", BOTH_SCENE_FIXTURES)
    def test_percentile_normalization_streams_and_stays_in_range(self, scene, request):
        ds = request.getfixturevalue(scene)
        stretched = ds.normalize_percentile()
        try:
            values = stretched.read()
            assert stretched.read().dtype == np.float32
            assert np.nanmin(values) == pytest.approx(0.0)
            assert np.nanmax(values) == pytest.approx(1.0)
            # Fill is excluded from the thresholds and stays nodata after.
            assert np.array_equal(np.isnan(values), ds.read() == ds.get_metadata()["nodata"])
        finally:
            stretched.close()

    @pytest.mark.parametrize("scene", BOTH_SCENE_FIXTURES)
    def test_sampling_a_point_does_not_read_the_scene(self, scene, request, monkeypatch):
        ds = request.getfixturevalue(scene)
        # The op used to read the whole band to index one pixel out of it —
        # 129 MB on the Landsat scene, for one number.
        reads = []
        # The class is not exported at package level; users reach a dataset
        # through the loaders.
        dataset_class = eeo.core.core.EEORasterDataset
        original = dataset_class.read

        def spy(self, *args, **kwargs):
            array = original(self, *args, **kwargs)
            reads.append(np.shape(array))
            return array

        monkeypatch.setattr(dataset_class, "read", spy)
        left, _bottom, _right, top = ds.get_bounds()
        ds.extract_value_at_coordinate((left + 1000.0, top - 1000.0))
        assert reads == [(1, 1)], f"expected one 1x1 read, got {reads}"


# ---------------------------------------------------------------------------
# Lazy backend
# ---------------------------------------------------------------------------

#: Chunk size for the lazy tests: small enough that both products' bands split
#: into several chunks along each axis, so windows and saves cross chunk seams.
LAZY_CHUNKS = 1024

#: Red and NIR members of each product, as shipped. Sentinel-2 is read at 20 m,
#: where both bands exist as 5490x5490 JP2s; Landsat as its 8081x7991 GeoTIFFs.
LAZY_BAND_PATTERNS = {
    "sentinel2": (
        "Sentinel-2",
        "*/GRANULE/*/IMG_DATA/R20m/*_B04_20m.jp2",
        "*/GRANULE/*/IMG_DATA/R20m/*_B8A_20m.jp2",
    ),
    "landsat": ("Landsat", "*_SR_B4.TIF", "*_SR_B5.TIF"),
}


@pytest.fixture(scope="module", params=["sentinel2", "landsat"])
def real_band_hrefs(request):
    """GDAL paths to one product's red and NIR bands, inside its archive."""
    pytest.importorskip("dask.array")
    pytest.importorskip("rioxarray")
    from eeo.io._archive import open_product

    product = request.param
    scene = request.getfixturevalue(f"{product}_scene")
    mission, *patterns = LAZY_BAND_PATTERNS[product]
    source = open_product(scene, mission)
    hrefs = []
    for pattern in patterns:
        (member,) = source.glob(pattern)
        hrefs.append(source.href(member))
    return tuple(hrefs)


def _open_both(href):
    """The same band opened lazily and with rasterio."""
    from eeo.core.core import EEORasterDataset

    return (
        EEORasterDataset.from_path(href, chunks=LAZY_CHUNKS),
        EEORasterDataset.from_path(href),
    )


class TestRealSceneLazyBackend:
    def test_the_bands_split_into_several_chunks_on_each_axis(self, real_band_hrefs):
        # Guards the seam checks below against a band that fits in one chunk.
        for href in real_band_hrefs:
            lazy, _ = _open_both(href)
            _, rows, cols = lazy.ds.data.chunks
            assert len(rows) > 1
            assert len(cols) > 1

    def test_opening_and_metadata_compute_nothing(self, real_band_hrefs):
        from dask.callbacks import Callback

        computed = []
        with Callback(start=lambda dsk: computed.append(dsk)):
            for href in real_band_hrefs:
                lazy, _ = _open_both(href)
                lazy.get_metadata()
                lazy.describe()
        assert computed == []

    def test_metadata_matches_rasterio(self, real_band_hrefs):
        for href in real_band_hrefs:
            lazy, reference = _open_both(href)
            assert lazy.get_metadata() == reference.get_metadata()
            assert lazy.get_bounds() == reference.get_bounds()
            assert lazy.band_names == reference.band_names

    def test_windows_across_chunk_seams_and_edges_match_rasterio(self, real_band_hrefs):
        from rasterio.windows import Window

        for href in real_band_hrefs:
            lazy, reference = _open_both(href)
            height, width = lazy.get_shape()
            windows = [
                # Straddles the first seam in both directions.
                Window(LAZY_CHUNKS - 7, LAZY_CHUNKS - 5, 300, 200),
                # One row across the whole width, crossing every column seam.
                Window(0, LAZY_CHUNKS, width, 1),
                # The bottom-right corner, inside the truncated last chunks.
                Window(width - 13, height - 11, 13, 11),
            ]
            for window in windows:
                assert np.array_equal(lazy.read(1, window=window), reference.read(1, window=window))

    def test_the_whole_band_matches_rasterio(self, real_band_hrefs):
        for href in real_band_hrefs:
            lazy, reference = _open_both(href)
            full = lazy.read()
            assert full.dtype == reference.read().dtype
            assert np.array_equal(full, reference.read())

    def test_saving_chunk_by_chunk_reproduces_the_band(self, real_band_hrefs, tmp_path):
        import rasterio as rio

        for i, href in enumerate(real_band_hrefs):
            lazy, reference = _open_both(href)
            out = tmp_path / f"band{i}.tif"
            lazy.save_raster(out)
            with rio.open(out) as saved:
                assert np.array_equal(saved.read(), reference.read())
                assert saved.dtypes[0] == reference.get_metadata()["dtype"]
                np.testing.assert_equal(saved.nodata, reference.get_metadata()["nodata"])
                assert saved.crs == reference.get_crs()
                assert saved.transform == reference.get_transform()

    def test_an_index_gives_the_same_answer_on_both_backends(self, real_band_hrefs):
        # Operations promote a lazy dataset today; whatever route they take,
        # the answer must not depend on which backend the bands were opened on.
        red_href, nir_href = real_band_hrefs
        lazy_red, red = _open_both(red_href)
        lazy_nir, nir = _open_both(nir_href)
        lazy_ndvi = lazy_nir.normalized_difference(lazy_red)
        ndvi = nir.normalized_difference(red)
        try:
            assert np.array_equal(lazy_ndvi.read(), ndvi.read(), equal_nan=True)
        finally:
            lazy_ndvi.close()
            ndvi.close()
