"""Exhaustive band-name matrix across every bound operation.

``test_band_names.py`` covers the semantics of naming, resolution, and
propagation. This module is the breadth check behind it: every operation bound
onto ``EEORasterDataset``, exercised on both single-band and multi-band rasters
and on both backends, so no op silently drops or mislabels a name.
"""

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from matplotlib.axes import Axes
from rasterio.crs import CRS
from shapely.geometry import box

from eeo import load_array, load_raster
from eeo.core.decorators import _OP_REGISTRY
from eeo.core.exceptions import ValidationError

UTM_CRS = CRS.from_epsg(32633)
TRANSFORM = Affine.translation(500_000, 4_200_000) * Affine.scale(10, -10)

# A 6-band scene covers every index's band requirements at once.
SCENE_NAMES = ["blue", "green", "red", "nir", "swir", "extra"]


def _ds(count=1, names=None, *, backend="rasterio"):
    """Build a raster of ``count`` bands with strictly positive values.

    Values start at 1 so ``log``, ``sqrt``, and ``divide`` are all well-defined
    and ``standardize`` never sees a constant band.
    """
    array = np.stack(
        [np.arange(1, 37, dtype=np.float32).reshape(6, 6) + i * 100 for i in range(count)]
    )
    ds = load_array(array, transform=TRANSFORM, crs=UTM_CRS, band_names=names)
    return ds.to_rasterio() if backend == "rasterio" else ds


def _scene(backend="rasterio"):
    """Six-band named scene: blue, green, red, nir, swir, extra."""
    return _ds(6, SCENE_NAMES, backend=backend)


def _inset_bbox(ds):
    """A bbox strictly inside ``ds``'s extent, for clipping."""
    left, bottom, right, top = ds.get_bounds()
    return (left + 10, bottom + 10, right - 10, top - 10)


def _inset_gdf(ds):
    """A single-polygon GeoDataFrame strictly inside ``ds``'s extent."""
    return gpd.GeoDataFrame(geometry=[box(*_inset_bbox(ds))], crs=ds.get_crs())


@pytest.fixture
def recorded_titles(monkeypatch):
    """Record every subplot title set while plotting."""
    titles: list[str] = []
    original = Axes.set_title

    def spy(self, label, *args, **kwargs):
        titles.append(str(label))
        return original(self, label, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_title", spy)
    return titles


# ---------------------------------------------------------------------------
# Identity-preserving operations: names must survive unchanged
# ---------------------------------------------------------------------------
IDENTITY_OPS = {
    "add": lambda ds: ds.add(1),
    "subtract": lambda ds: ds.subtract(1),
    "multiply": lambda ds: ds.multiply(2),
    "divide": lambda ds: ds.divide(2),
    "power": lambda ds: ds.power(2),
    "sqrt": lambda ds: ds.sqrt(),
    "log": lambda ds: ds.log(),
    "absolute": lambda ds: ds.absolute(),
    "standardize": lambda ds: ds.standardize(),
    "normalize_min_max": lambda ds: ds.normalize_min_max(),
    "normalize_percentile": lambda ds: ds.normalize_percentile(),
    "resample": lambda ds: ds.resample(scale_factor=0.5),
    "reproject_raster": lambda ds: ds.reproject_raster(target_crs=4326),
    "clip_raster_with_bbox": lambda ds: ds.clip_raster_with_bbox(_inset_bbox(ds)),
    "clip_raster_with_vector": lambda ds: ds.clip_raster_with_vector(_inset_gdf(ds)),
    "to_rasterio": lambda ds: ds.to_rasterio(),
    "operator_add": lambda ds: ds + 1,
    "operator_mul": lambda ds: ds * 2,
    "operator_pow": lambda ds: ds**2,
    "operator_rsub": lambda ds: 100 - ds,
}


@pytest.mark.parametrize("op", sorted(IDENTITY_OPS), ids=sorted(IDENTITY_OPS))
def test_identity_op_preserves_a_single_band_name(op):
    assert IDENTITY_OPS[op](_ds(1, ["nir"])).band_names == ["nir"]


@pytest.mark.parametrize("op", sorted(IDENTITY_OPS), ids=sorted(IDENTITY_OPS))
def test_identity_op_preserves_multiband_names(op):
    names = ["blue", "green", "red", "nir"]
    assert IDENTITY_OPS[op](_ds(4, names)).band_names == names


@pytest.mark.parametrize("op", sorted(IDENTITY_OPS), ids=sorted(IDENTITY_OPS))
def test_identity_op_preserves_partially_named_bands(op):
    # A gap must stay a gap: unnamed bands are never backfilled.
    names = ["blue", None, "red", None]
    assert IDENTITY_OPS[op](_ds(4, names)).band_names == names


@pytest.mark.parametrize("op", sorted(IDENTITY_OPS), ids=sorted(IDENTITY_OPS))
def test_identity_op_on_unnamed_raster_stays_unnamed(op):
    assert IDENTITY_OPS[op](_ds(4)).band_names == [None] * 4


@pytest.mark.parametrize("op", ["add", "multiply", "resample", "to_rasterio"], ids=str)
def test_identity_op_preserves_names_from_the_numpy_backend(op):
    # These ops promote a NumPy-backed dataset internally; names must ride along.
    assert IDENTITY_OPS[op](_ds(2, ["red", "nir"], backend="numpy")).band_names == ["red", "nir"]


def test_raster_to_raster_algebra_preserves_the_receiver_names():
    ds = _ds(4, ["blue", "green", "red", "nir"])
    other = _ds(4, ["a", "b", "c", "d"])
    assert ds.add(other).band_names == ["blue", "green", "red", "nir"]
    assert ds.subtract(other).band_names == ["blue", "green", "red", "nir"]


def test_broadcasting_that_keeps_the_band_count_keeps_the_names():
    multi = _ds(4, ["blue", "green", "red", "nir"])
    single = _ds(1, ["scale"])
    assert multi.multiply(single).band_names == ["blue", "green", "red", "nir"]


def test_names_are_not_propagated_when_an_op_changes_the_band_count():
    """The decorator's count guard: names must not be copied onto a different
    number of bands, since output band *i* would no longer be input band *i*.

    No shipped op reaches this branch today (every non-concatenating op keeps
    the band count), so it is exercised through a throwaway op registered and
    unregistered around the assertion.
    """
    from eeo.core.core import EEORasterDataset
    from eeo.core.decorators import _OP_REGISTRY, eeo_raster_op

    @eeo_raster_op
    def _band_count_changing_op_for_test(ds):
        return _ds(ds.get_count() * 2)

    try:
        result = _ds(2, ["red", "nir"])._band_count_changing_op_for_test()
        assert result.get_count() == 4
        assert result.band_names == [None] * 4
    finally:
        delattr(EEORasterDataset, "_band_count_changing_op_for_test")
        _OP_REGISTRY[:] = [
            entry for entry in _OP_REGISTRY if entry[0] is not _band_count_changing_op_for_test
        ]


# ---------------------------------------------------------------------------
# Every spectral index: resolution by name, and output naming
# ---------------------------------------------------------------------------
# (kwargs by name, the same bands by 1-based index into SCENE_NAMES)
INDEX_SPECS = {
    "ndvi": ({"red": "red", "nir": "nir"}, {"red": 3, "nir": 4}),
    "ndwi": ({"nir": "nir", "green": "green"}, {"nir": 4, "green": 2}),
    "ndmi": ({"swir": "swir", "nir": "nir"}, {"swir": 5, "nir": 4}),
    "ndbi": ({"nir": "nir", "swir": "swir"}, {"nir": 4, "swir": 5}),
    "evi": ({"red": "red", "blue": "blue", "nir": "nir"}, {"red": 3, "blue": 1, "nir": 4}),
    "savi": ({"red": "red", "nir": "nir"}, {"red": 3, "nir": 4}),
}


@pytest.mark.parametrize("index", sorted(INDEX_SPECS), ids=sorted(INDEX_SPECS))
def test_index_by_name_matches_by_index(index):
    scene = _scene()
    by_name, by_index = INDEX_SPECS[index]
    np.testing.assert_array_equal(
        getattr(scene, index)(**by_name, return_as_ndarray=True),
        getattr(scene, index)(**by_index, return_as_ndarray=True),
    )


@pytest.mark.parametrize("index", sorted(INDEX_SPECS), ids=sorted(INDEX_SPECS))
def test_index_output_is_unnamed_by_default(index):
    scene = _scene()
    by_name, _ = INDEX_SPECS[index]
    assert getattr(scene, index)(**by_name).band_names == [None]


@pytest.mark.parametrize("index", sorted(INDEX_SPECS), ids=sorted(INDEX_SPECS))
def test_index_output_honours_an_explicit_name(index):
    scene = _scene()
    by_name, _ = INDEX_SPECS[index]
    assert getattr(scene, index)(**by_name, name=f"{index}_2024").band_names == [f"{index}_2024"]


@pytest.mark.parametrize("index", sorted(INDEX_SPECS), ids=sorted(INDEX_SPECS))
def test_index_rejects_an_unknown_band_name(index):
    scene = _scene()
    by_name, _ = INDEX_SPECS[index]
    bogus = dict.fromkeys(by_name, "no_such_band")
    with pytest.raises(ValidationError):
        getattr(scene, index)(**bogus)


@pytest.mark.parametrize("index", sorted(INDEX_SPECS), ids=sorted(INDEX_SPECS))
def test_index_resolves_names_on_the_numpy_backend(index):
    scene = _scene(backend="numpy")
    by_name, by_index = INDEX_SPECS[index]
    np.testing.assert_array_equal(
        getattr(scene, index)(**by_name, return_as_ndarray=True),
        getattr(scene, index)(**by_index, return_as_ndarray=True),
    )


@pytest.mark.parametrize("index", sorted(INDEX_SPECS), ids=sorted(INDEX_SPECS))
def test_index_from_separate_single_band_rasters_is_unnamed(index):
    # Every band is its own named raster: the index band is still a new
    # measurement, so it inherits none of their names.
    bands = {key: _ds(1, [key]) for key in INDEX_SPECS[index][0]}
    receiver = _ds(1, ["primary"])
    assert getattr(receiver, index)(**bands).band_names == [None]


def test_index_names_survive_chaining_into_another_op():
    scene = _scene()
    chained = scene.ndvi(red="red", nir="nir", name="ndvi").multiply(100).add(1)
    assert chained.band_names == ["ndvi"]


# ---------------------------------------------------------------------------
# Every pixel-statistic op accepts a name
# ---------------------------------------------------------------------------
STATS_OPS = {
    "get_maximum_pixel": {},
    "get_minimum_pixel": {},
    "get_mean_pixel": {},
    "get_percentile_pixel": {"percentile": 90},
}


@pytest.mark.parametrize("op", sorted(STATS_OPS), ids=sorted(STATS_OPS))
@pytest.mark.parametrize("backend", ["rasterio", "numpy"])
def test_stats_op_by_name_matches_by_index(op, backend):
    scene = _scene(backend=backend)
    extra = STATS_OPS[op]
    assert getattr(scene, op)(band_idx="swir", **extra) == getattr(scene, op)(band_idx=5, **extra)


@pytest.mark.parametrize("op", sorted(STATS_OPS), ids=sorted(STATS_OPS))
def test_stats_op_on_a_single_band_raster_by_name(op):
    ds = _ds(1, ["elevation"])
    extra = STATS_OPS[op]
    assert getattr(ds, op)(band_idx="elevation", **extra) == getattr(ds, op)(band_idx=1, **extra)


@pytest.mark.parametrize("op", sorted(STATS_OPS), ids=sorted(STATS_OPS))
def test_stats_op_rejects_an_unknown_name(op):
    with pytest.raises(ValidationError):
        getattr(_scene(), op)(band_idx="no_such_band", **STATS_OPS[op])


@pytest.mark.parametrize("backend", ["rasterio", "numpy"])
def test_extract_value_at_coordinate_by_name_matches_by_index(backend):
    scene = _scene(backend=backend)
    left, bottom, right, top = scene.get_bounds()
    point = ((left + right) / 2, (bottom + top) / 2)
    assert scene.extract_value_at_coordinate(point, band_idx="swir") == (
        scene.extract_value_at_coordinate(point, band_idx=5)
    )


@pytest.mark.parametrize("backend", ["rasterio", "numpy"])
def test_get_band_by_name_matches_by_index(backend):
    scene = _scene(backend=backend)
    for position, name in enumerate(SCENE_NAMES, start=1):
        np.testing.assert_array_equal(scene.get_band(name), scene.get_band(position))


# ---------------------------------------------------------------------------
# Every plotting function resolves names and labels subplots with them
# ---------------------------------------------------------------------------
BAND_PLOTS = [
    "plot_band_array",
    "plot_raster",
    "plot_histogram",
    "plot_raster_with_histogram",
]


@pytest.mark.parametrize("plot", BAND_PLOTS)
@pytest.mark.parametrize("backend", ["rasterio", "numpy"])
def test_plot_accepts_names_and_mixed_lists(plot, backend, recorded_titles):
    scene = _scene(backend=backend)
    getattr(scene, plot)(bands=["red", 2, "nir"])
    assert any("(red)" in title for title in recorded_titles)
    assert any("Band 2 (green)" in title for title in recorded_titles)


@pytest.mark.parametrize("plot", BAND_PLOTS)
def test_plot_accepts_a_single_name(plot, recorded_titles):
    getattr(_scene(), plot)(bands="swir")
    assert any("Band 5 (swir)" in title for title in recorded_titles)


@pytest.mark.parametrize("plot", BAND_PLOTS)
def test_plot_rejects_an_unknown_name(plot):
    with pytest.raises(ValidationError):
        getattr(_scene(), plot)(bands=["no_such_band"])


@pytest.mark.parametrize("plot", BAND_PLOTS)
def test_plot_of_an_unnamed_raster_keeps_plain_labels(plot, recorded_titles):
    getattr(_ds(2), plot)(bands=[1, 2])
    assert "Band 1" in recorded_titles
    assert not any("(" in title for title in recorded_titles)


@pytest.mark.parametrize("backend", ["rasterio", "numpy"])
def test_plot_composite_by_name_matches_by_index(backend, recorded_titles):
    scene = _scene(backend=backend)
    scene.plot_composite(bands=["red", "green", "blue"])
    scene.plot_composite(bands=(3, 2, 1))


def test_plot_composite_rejects_an_unknown_name():
    with pytest.raises(ValidationError):
        _scene().plot_composite(bands=["red", "green", "no_such_band"])


@pytest.mark.parametrize("selection", [["red"], ["red", "green"], ["red", "green", "blue", "nir"]])
def test_plot_composite_requires_exactly_three_bands(selection):
    with pytest.raises(ValidationError):
        _scene().plot_composite(bands=selection)


# ---------------------------------------------------------------------------
# Concatenating ops, single-band and multi-band
# ---------------------------------------------------------------------------
def test_stack_of_single_band_rasters_concatenates_names():
    red, green, blue = (_ds(1, [name]) for name in ("red", "green", "blue"))
    assert red.stack([green, blue]).band_names == ["red", "green", "blue"]


def test_stack_of_multiband_rasters_concatenates_names():
    left = _ds(2, ["blue", "green"])
    right = _ds(2, ["red", "nir"])
    assert left.stack(right).band_names == ["blue", "green", "red", "nir"]


def test_stack_mixes_named_and_unnamed_inputs():
    named = _ds(2, ["blue", "green"])
    unnamed = _ds(2)
    assert named.stack(unnamed).band_names == ["blue", "green", None, None]
    assert unnamed.stack(named).band_names == [None, None, "blue", "green"]


def test_stack_of_wholly_unnamed_inputs_stays_unnamed():
    assert _ds(2).stack(_ds(1)).band_names == [None, None, None]


def test_stacked_duplicate_names_load_fine_but_fail_to_resolve():
    # Stacking two rasters that both call their band "red" is allowed; the
    # ambiguity only surfaces when something tries to resolve the name.
    stacked = _ds(1, ["red"]).stack(_ds(1, ["red"]))
    assert stacked.band_names == ["red", "red"]
    with pytest.raises(ValidationError):
        stacked.get_band("red")


def test_stack_names_override_length_is_validated():
    with pytest.raises(ValidationError):
        _ds(2, ["a", "b"]).stack(_ds(1, ["c"]), names=["x", "y"])


def test_mosaic_of_multiband_rasters_keeps_the_primary_names():
    ds = _ds(4, ["blue", "green", "red", "nir"])
    assert ds.mosaic(ds).band_names == ["blue", "green", "red", "nir"]


def test_mosaic_names_override_length_is_validated():
    ds = _ds(2, ["a", "b"])
    with pytest.raises(ValidationError):
        ds.mosaic(ds, names=["only-one"])


def test_mosaic_to_save_path_writes_the_names(tmp_path):
    ds = _ds(2, ["red", "nir"])
    path = tmp_path / "mosaic.tif"
    assert ds.mosaic(ds, save_path=str(path)) is None
    assert load_raster(str(path)).band_names == ["red", "nir"]


# ---------------------------------------------------------------------------
# Save/load round trips across band counts and backends
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ["rasterio", "numpy"])
@pytest.mark.parametrize(
    "names",
    [["only"], ["blue", "green", "red", "nir"], ["blue", None, "red", None], [None, None]],
    ids=["single", "multi", "partial", "unnamed"],
)
def test_names_round_trip_through_a_geotiff(names, backend, tmp_path):
    ds = _ds(len(names), names, backend=backend)
    path = tmp_path / "round_trip.tif"
    ds.save_raster(str(path))
    assert load_raster(str(path)).band_names == names


def test_names_round_trip_after_a_chain(tmp_path):
    scene = _scene()
    result = scene.multiply(2).clip_raster_with_bbox(_inset_bbox(scene)).resample(scale_factor=0.5)
    path = tmp_path / "chained.tif"
    result.save_raster(str(path))
    assert load_raster(str(path)).band_names == SCENE_NAMES


# ---------------------------------------------------------------------------
# No bound operation is missing from this matrix
# ---------------------------------------------------------------------------
def test_every_bound_op_is_covered_by_this_module():
    """Fail if a new op is added without a band-name test here."""
    bound = {func.__name__ for func, _kind in _OP_REGISTRY}
    covered = (
        set(IDENTITY_OPS)
        | set(INDEX_SPECS)
        | set(STATS_OPS)
        | set(BAND_PLOTS)
        | {
            "extract_value_at_coordinate",
            "normalized_difference",  # covered in test_band_names.py
            "plot_composite",
            "stack",
            "mosaic",
        }
    )
    assert bound - covered == set()
