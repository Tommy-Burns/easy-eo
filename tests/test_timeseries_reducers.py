"""Temporal reducers (eeo/timeseries/reducers.py).

The season fixtures in conftest.py are built so every reduction over them is a
round number, and so that one pixel is missing at exactly two timesteps — which
is where a nodata-aware reducer and a naive one part company.
"""

import datetime as dt

import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import from_origin

import eeo
from eeo.core.exceptions import ValidationError
from eeo.timeseries import reducers

UTC = dt.timezone.utc
UTM = "EPSG:32633"
TRANSFORM = from_origin(500_000.0, 4_200_000.0, 10.0, 10.0)


def scene(month, array, *, nodata=0, names=None):
    """A rasterio-backed scene dated the first of ``month`` in 2023."""
    return eeo.load_array(
        np.asarray(array),
        transform=TRANSFORM,
        crs=UTM,
        nodata=nodata,
        timestamp=dt.datetime(2023, month, 1, tzinfo=UTC),
        band_names=names,
    ).to_rasterio()


def uniform(month, value, *, dtype="uint16", nodata=0, size=4):
    """A single-band scene of one repeated value."""
    return scene(month, np.full((size, size), value, dtype=dtype), nodata=nodata)


# --------------------------
# What each reducer computes
# --------------------------
def test_median_over_the_season(season_series):
    result = season_series.median()

    values = result.to_array()
    assert values[0, 1, 1] == pytest.approx(900.0)
    assert values[1, 1, 1] == pytest.approx(3000.0)


def test_mean_over_the_season(season_series):
    values = season_series.mean().to_array()

    assert values[0, 1, 1] == pytest.approx(920.0)
    assert values[1, 1, 1] == pytest.approx(2800.0)


def test_min_and_max_over_the_season(season_series):
    smallest = season_series.min().to_array()
    largest = season_series.max().to_array()

    assert (int(smallest[0, 1, 1]), int(smallest[1, 1, 1])) == (800, 2000)
    assert (int(largest[0, 1, 1]), int(largest[1, 1, 1])) == (1000, 4000)


def test_a_reduction_keeps_the_grid_bands_and_names(season_series):
    result = season_series.median()

    assert result.get_shape() == season_series.shape
    assert result.get_transform() == season_series.transform
    assert result.get_crs() == season_series.crs
    assert result.get_count() == season_series.band_count
    assert result.band_names == ["red", "nir"]


def test_a_reduction_records_what_it_reduced(season_series):
    attrs = season_series.mean().attrs

    assert attrs["temporal_reduction"] == "mean"
    assert attrs["timesteps"] == 5
    assert attrs["time_start"] == season_series.timestamps[0]
    assert attrs["time_end"] == season_series.timestamps[-1]


def test_a_reduction_carries_no_acquisition_time(season_series):
    # A composite was not acquired at any one moment; the span is in attrs.
    assert season_series.median().timestamp is None


def test_shared_provenance_survives_but_per_scene_provenance_does_not():
    first = uniform(4, 100)
    second = uniform(5, 200)
    for index, ds in enumerate((first, second)):
        ds.attrs.update(mission="Sentinel-2", stac_item=f"item_{index}")

    attrs = eeo.time_series([first, second]).median().attrs

    assert attrs["mission"] == "Sentinel-2"
    assert "stac_item" not in attrs


# ----------------
# Dtype and nodata
# ----------------
def test_median_and_mean_are_float32_with_nan_nodata(season_series):
    for result in (season_series.median(), season_series.mean()):
        assert result.get_metadata()["dtype"] == "float32"
        assert np.isnan(result.get_metadata()["nodata"])


def test_min_and_max_keep_the_timesteps_dtype(season_series):
    # A minimum selects a measured value; it must not be promoted to float.
    for result in (season_series.min(), season_series.max()):
        assert result.get_metadata()["dtype"] == "uint16"
        assert result.get_metadata()["nodata"] == 0


def test_mixed_dtypes_promote_for_min_and_max():
    ts = eeo.time_series([uniform(4, 100), uniform(5, 200, dtype="int32")])

    assert ts.min().get_metadata()["dtype"] == "int32"


def test_a_series_declaring_no_nodata_produces_none():
    ts = eeo.time_series([uniform(4, 100, nodata=None), uniform(5, 200, nodata=None)])

    result = ts.min()

    assert result.get_metadata()["nodata"] is None
    assert int(result.to_array()[0, 0, 0]) == 100


# --------------------------------
# Nodata is absent, not contagious
# --------------------------------
def test_a_pixel_missing_at_some_timesteps_still_reduces(season_series):
    # Pixel (0, 0) is nodata at the third and fourth timesteps, leaving red
    # 1000, 900, 1000 — so the median is 1000 and the mean 966.67, not nodata.
    assert season_series.median().to_array()[0, 0, 0] == pytest.approx(1000.0)
    assert season_series.mean().to_array()[0, 0, 0] == pytest.approx(2900 / 3, abs=1e-3)
    assert int(season_series.min().to_array()[0, 0, 0]) == 900
    assert int(season_series.max().to_array()[0, 0, 0]) == 1000


def test_a_fill_value_never_wins_a_minimum():
    # The naive answer would be 0, the fill value, at the masked pixel.
    first = np.full((1, 4, 4), 500, dtype="uint16")
    second = np.full((1, 4, 4), 700, dtype="uint16")
    second[0, 2, 2] = 0
    ts = eeo.time_series([scene(4, first), scene(5, second)])

    assert int(ts.min().to_array()[0, 2, 2]) == 500


def test_a_pixel_missing_everywhere_is_nodata():
    first = np.full((1, 4, 4), 500, dtype="uint16")
    second = np.full((1, 4, 4), 700, dtype="uint16")
    first[0, 3, 3] = 0
    second[0, 3, 3] = 0
    ts = eeo.time_series([scene(4, first), scene(5, second)])

    assert np.isnan(ts.median().to_array()[0, 3, 3])
    assert np.isnan(ts.mean().to_array()[0, 3, 3])
    # Integer output cannot hold NaN, so it takes the declared sentinel.
    assert int(ts.min().to_array()[0, 3, 3]) == 0
    assert int(ts.max().to_array()[0, 3, 3]) == 0


def test_nan_in_floating_data_counts_as_missing():
    first = np.full((1, 4, 4), 1.0, dtype="float32")
    second = np.full((1, 4, 4), 3.0, dtype="float32")
    second[0, 1, 1] = np.nan
    ts = eeo.time_series([scene(4, first, nodata=None), scene(5, second, nodata=None)])

    assert ts.mean().to_array()[0, 1, 1] == pytest.approx(1.0)
    assert ts.mean().to_array()[0, 2, 2] == pytest.approx(2.0)


# ---------------------------
# Streaming, saving, chaining
# ---------------------------
def test_a_reduction_streams_across_block_seams():
    # Larger than one block (1 << 20 pixels), so the reduction is written in
    # several strips; a seam bug shows as a band of wrong rows.
    side = 1200
    rows = np.arange(side, dtype="uint16").reshape(side, 1)
    stack = [
        np.broadcast_to(rows + offset, (side, side)).astype("uint16") for offset in (0, 100, 200)
    ]
    ts = eeo.time_series(
        [
            scene(month, array[np.newaxis], nodata=None)
            for month, array in zip((4, 5, 6), stack, strict=True)
        ]
    )

    reduced = ts.median().to_array()[0]

    expected = np.median(np.stack(stack), axis=0).astype("float32")
    np.testing.assert_allclose(reduced, expected)


def test_save_path_writes_the_reduction_to_disk(season_series, tmp_path):
    out = tmp_path / "composite.tif"

    result = season_series.median(save_path=out)

    assert out.exists()
    assert result.path == out
    assert result.to_array()[0, 1, 1] == pytest.approx(900.0)
    assert result.band_names == ["red", "nir"]
    result.close()


def test_a_reduction_is_an_ordinary_dataset_and_chains(season_series):
    peak = season_series.map(eeo.ndvi, red="red", nir="nir").max()

    # Peak greenness over the season is midsummer's NDVI.
    assert peak.to_array()[0, 1, 1] == pytest.approx(2 / 3, abs=1e-6)
    assert peak.multiply(100).to_array()[0, 1, 1] == pytest.approx(200 / 3, abs=1e-4)


def test_a_single_timestep_series_reduces_to_itself():
    ts = eeo.time_series([uniform(4, 42)])

    assert int(ts.min().to_array()[0, 0, 0]) == 42
    assert ts.mean().to_array()[0, 0, 0] == pytest.approx(42.0)


def test_an_unknown_reduction_is_refused(season_series):
    with pytest.raises(ValidationError, match="how must be"):
        reducers.reduce_series(season_series, "mode")


def test_an_empty_band_list_is_refused(season_series):
    with pytest.raises(ValidationError, match="bands is empty"):
        reducers.median(season_series, bands=[])


def outputs_opened(monkeypatch):
    """Record every raster a reduction opens to write into, and its MemoryFile."""
    opened = []
    real_open, real_memfile = rio.open, rio.io.MemoryFile

    def spy_open(path, mode="r", **kwargs):
        dataset = real_open(path, mode, **kwargs)
        if mode == "w":
            opened.append(dataset)
        return dataset

    def spy_memfile(*args, **kwargs):
        opened.append(real_memfile(*args, **kwargs))
        return opened[-1]

    monkeypatch.setattr(rio, "open", spy_open)
    monkeypatch.setattr(rio.io, "MemoryFile", spy_memfile)
    return opened


def failing_block(*args, **kwargs):
    raise RuntimeError("out of memory")


@pytest.mark.parametrize("to_disk", [False, True], ids=["in-memory", "save_path"])
def test_a_failed_reduction_closes_its_half_written_output(
    season_series, tmp_path, monkeypatch, to_disk
):
    opened = outputs_opened(monkeypatch)
    monkeypatch.setattr(reducers, "_reduce_block", failing_block)

    with pytest.raises(RuntimeError, match="out of memory"):
        season_series.median(save_path=tmp_path / "composite.tif" if to_disk else None)

    # Left open, its dirty blocks would sit in GDAL's cache.
    assert opened
    assert all(handle.closed for handle in opened)


def test_a_save_path_that_cannot_be_opened_raises_before_anything_is_read(season_series, tmp_path):
    # The output fails to open, so there is nothing yet to close on the way out.
    with pytest.raises(rio.errors.RasterioIOError):
        season_series.median(save_path=tmp_path / "no-such-folder" / "composite.tif")


def test_the_reducers_are_also_plain_functions(season_series):
    # The methods are thin: the work lives in functions taking a series, which
    # is what keeps core.py from growing a statistics library.
    assert reducers.median(season_series).to_array()[0, 1, 1] == pytest.approx(900.0)
    assert reducers.maximum(season_series).to_array()[0, 1, 1] == 1000


# --------------------------------------------------------------------------
# composite: mask each timestep with its own quality band, then reduce
# --------------------------------------------------------------------------
CLEAR = 4  # SCLClass.VEGETATION
CLOUD = 9  # SCLClass.CLOUD_HIGH_PROBABILITY
SHADOW = 3  # SCLClass.CLOUD_SHADOWS


def s2_scene(month, red, nir, *, flagged=(), flag=CLOUD, names=("red", "nir", "scl")):
    """A three-band Sentinel-2-like scene whose SCL band flags some pixels."""
    scl = np.full((4, 4), CLEAR, dtype="uint16")
    for row, col in flagged:
        scl[row, col] = flag
    bands = np.stack(
        [
            np.full((4, 4), red, dtype="uint16"),
            np.full((4, 4), nir, dtype="uint16"),
            scl,
        ]
    )
    return scene(month, bands, names=list(names))


@pytest.fixture
def clouded_series():
    """Three timesteps whose cloud roams, plus one pixel clouded throughout.

    Pixel (3, 3) is clouded at every timestep — the one a composite cannot
    fill. Pixels (0, 0), (1, 1) and (2, 2) are each clouded once, so each has
    two clear timesteps to be built from.
    """
    scenes = [
        s2_scene(4, 1000, 2000, flagged=[(0, 0), (3, 3)]),
        s2_scene(5, 900, 3000, flagged=[(1, 1), (3, 3)]),
        s2_scene(6, 800, 4000, flagged=[(2, 2), (3, 3)]),
    ]
    ts = eeo.time_series(scenes)
    yield ts
    ts.close()


def test_the_composite_leaves_the_quality_band_out(clouded_series):
    # A median of scene-class numbers would be a class nobody assigned.
    result = clouded_series.composite()

    assert result.get_count() == 2
    assert result.band_names == ["red", "nir"]


def test_a_clear_pixel_is_the_median_of_every_timestep(clouded_series):
    values = clouded_series.composite().to_array()

    assert values[0, 0, 1] == pytest.approx(900.0)
    assert values[1, 0, 1] == pytest.approx(3000.0)


def test_a_pixel_clouded_once_is_built_from_the_timesteps_that_saw_it(clouded_series):
    values = clouded_series.composite().to_array()

    # (0, 0) is clouded in April, leaving red 900 and 800: median 850.
    assert values[0, 0, 0] == pytest.approx(850.0)
    # (2, 2) is clouded in June, leaving 1000 and 900: median 950.
    assert values[0, 2, 2] == pytest.approx(950.0)


def test_a_pixel_clouded_at_every_timestep_cannot_be_filled(clouded_series):
    values = clouded_series.composite().to_array()

    assert np.isnan(values[0, 3, 3])
    assert np.isnan(values[1, 3, 3])


def test_the_composite_records_what_it_covered(clouded_series):
    attrs = clouded_series.composite().attrs

    assert attrs["temporal_reduction"] == "median"
    assert attrs["timesteps"] == 3
    assert attrs["time_start"].month == 4
    assert attrs["time_end"].month == 6


def test_the_statistic_can_be_chosen(clouded_series):
    smallest = clouded_series.composite(how="min")

    assert smallest.get_metadata()["dtype"] == "uint16"
    # (0, 0) is clouded in April, so the smallest clear red is 800.
    assert int(smallest.to_array()[0, 0, 0]) == 800


def test_the_quality_band_can_be_named(clouded_series):
    by_name = clouded_series.composite(mask_band="scl").to_array()
    by_index = clouded_series.composite(mask_band=3).to_array()

    np.testing.assert_allclose(by_name, by_index)


def test_which_classes_count_as_cloud_can_be_chosen():
    # Cloud shadow is masked by default; asking for cloud only keeps it.
    scenes = [
        s2_scene(4, 1000, 2000, flagged=[(0, 0)], flag=SHADOW),
        s2_scene(5, 900, 3000),
    ]
    ts = eeo.time_series(scenes)

    default = ts.composite().to_array()
    cloud_only = ts.composite(classes=[CLOUD]).to_array()

    assert default[0, 0, 0] == pytest.approx(900.0)
    assert cloud_only[0, 0, 0] == pytest.approx(950.0)
    ts.close()


def test_a_series_with_no_quality_band_says_what_to_load(season_series):
    with pytest.raises(ValidationError, match="no quality band"):
        season_series.composite()


def test_a_series_of_nothing_but_a_quality_band_is_refused():
    scl = np.full((1, 4, 4), CLEAR, dtype="uint16")
    ts = eeo.time_series([scene(4, scl, names=["scl"]), scene(5, scl, names=["scl"])])

    with pytest.raises(ValidationError, match="nothing to composite"):
        ts.composite()

    ts.close()


def test_landsat_flags_are_refused_on_a_sentinel2_series(clouded_series):
    with pytest.raises(ValidationError, match="use classes="):
        clouded_series.composite(flags=[3])


def test_the_masked_timesteps_can_be_kept(clouded_series, tmp_path):
    out = tmp_path / "masked"

    result = clouded_series.composite(mask_dir=out)

    written = sorted(out.glob("*.tif"))
    assert len(written) == 3
    # Masking happens before the reduction, so what was written is masked but
    # not yet reduced: three bands each, the quality band among them.
    assert eeo.load_raster(written[0]).get_count() == 3
    assert result.to_array()[0, 0, 0] == pytest.approx(850.0)


def test_the_composite_can_be_written_straight_to_disk(clouded_series, tmp_path):
    out = tmp_path / "composite.tif"

    result = clouded_series.composite(save_path=out)

    assert out.exists()
    assert result.band_names == ["red", "nir"]
    assert result.to_array()[0, 0, 1] == pytest.approx(900.0)
    result.close()


def test_a_composite_chains_like_any_dataset(clouded_series):
    ndvi = clouded_series.composite().ndvi(red="red", nir="nir")

    # Clear pixel: red 900, nir 3000 -> (3000 - 900) / 3900.
    assert ndvi.to_array()[0, 0, 1] == pytest.approx(2100 / 3900, abs=1e-6)


def test_a_landsat_series_composites_from_qa_pixel():
    # QA_PIXEL packs flags into bits: 0 is a clear pixel, bit 3 is cloud.
    def landsat(month, value, *, clouded):
        qa = np.zeros((4, 4), dtype="uint16")
        for row, col in clouded:
            qa[row, col] = 1 << 3
        bands = np.stack([np.full((4, 4), value, dtype="uint16"), qa])
        ds = scene(month, bands, names=["red", "qa_pixel"])
        ds.attrs["mission"] = "Landsat 9"
        return ds

    ts = eeo.time_series([landsat(4, 1000, clouded=[(0, 0)]), landsat(5, 500, clouded=[])])

    result = ts.composite()

    assert result.band_names == ["red"]
    assert result.to_array()[0, 0, 0] == pytest.approx(500.0)
    assert result.to_array()[0, 1, 1] == pytest.approx(750.0)
    ts.close()
