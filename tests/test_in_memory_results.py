"""Every in-memory raster the library returns is closed and reopened read-only.

This pins the cause of a real slowdown, not a style preference. A result left
as an open writer keeps its freshly written blocks *dirty* in GDAL's block
cache, and each operation in a chain adds another raster's worth. Once that
outgrows the cache — 5% of RAM by default — GDAL flushes dirty blocks one band
at a time, which for a pixel-interleaved GeoTIFF is catastrophically slow. On
a real 6-band, 5490x5490 Sentinel-2 stack with default settings, a three-op
chain took 0.7 s, 124 s and 354 s per step before the fix.

A timing test would be the direct check, but GDAL sizes its cache once per
process, so a test cannot shrink it reliably. The read-only mode *is* the fix,
though: a raster opened for reading has no dirty blocks to leave behind. So
that is what is asserted, for every code path that builds one.
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from eeo import load_array
from eeo.core.adapters import RasterioAdapter


def _inset_bbox(ds):
    left, bottom, right, top = ds.get_bounds()
    return (left + 10, bottom + 10, right - 10, top - 10)


def _inset_frame(ds):
    return gpd.GeoDataFrame(geometry=[box(*_inset_bbox(ds))], crs=ds.get_crs())


#: Every public route to an in-memory raster. A new op that writes one should
#: be added here, or the test below says nothing about it.
PRODUCERS = [
    pytest.param(lambda ds: ds.add(1), id="algebra (block-wise engine)"),
    pytest.param(lambda ds: ds.normalized_difference(ds.add(1)), id="index"),
    pytest.param(lambda ds: ds.normalize_percentile(), id="normalization"),
    pytest.param(lambda ds: ds.clip_raster_with_bbox(_inset_bbox(ds)), id="clip_raster_with_bbox"),
    pytest.param(
        lambda ds: ds.clip_raster_with_vector(_inset_frame(ds)), id="clip_raster_with_vector"
    ),
    pytest.param(lambda ds: ds.reproject_raster(target_crs=4326), id="reproject_raster"),
    pytest.param(lambda ds: ds.resample(scale_factor=0.5), id="resample"),
    pytest.param(lambda ds: ds.mosaic(ds.add(1)), id="mosaic"),
    pytest.param(lambda ds: ds.stack(ds.add(1)), id="stack"),
]


@pytest.mark.parametrize("produce", PRODUCERS)
def test_an_op_result_is_read_only(single_band_float32, produce):
    result = produce(single_band_float32)
    try:
        assert result.ds.mode == "r"
        assert not result.ds.closed
    finally:
        result.close()


def test_promoting_a_numpy_array_gives_a_read_only_raster(numpy_backed_dataset):
    # The route every scene loader takes: load_array(...).to_rasterio().
    promoted = numpy_backed_dataset.to_rasterio()
    try:
        assert promoted.ds.mode == "r"
        assert np.array_equal(promoted.read(), numpy_backed_dataset.read())
    finally:
        promoted.close()


def test_mask_clouds_result_is_read_only(single_band_float32):
    scene = np.stack([single_band_float32.read()[0], np.full((6, 6), 4, dtype=np.float32)])
    ds = load_array(
        scene,
        transform=single_band_float32.get_transform(),
        crs=single_band_float32.get_crs(),
        band_names=["red", "scl"],
    ).to_rasterio()
    try:
        masked = ds.mask_clouds()
        assert masked.ds.mode == "r"
        masked.close()
    finally:
        ds.close()


def test_a_long_chain_ends_read_only_at_every_step(single_band_float32):
    step = single_band_float32
    for build in (
        lambda ds: ds.add(1),
        lambda ds: ds.multiply(2),
        lambda ds: ds.clip_raster_with_bbox(_inset_bbox(ds)),
        lambda ds: ds.normalize_min_max(),
    ):
        step = build(step)
        assert step.ds.mode == "r"


@pytest.mark.parametrize(
    "clip",
    [
        pytest.param(lambda ds, **kw: ds.clip_raster_with_bbox(_inset_bbox(ds), **kw), id="bbox"),
        pytest.param(
            lambda ds, **kw: ds.clip_raster_with_vector(_inset_frame(ds), **kw), id="vector"
        ),
    ],
)
def test_a_clip_preview_plots_from_the_reopened_result(single_band_float32, clip, monkeypatch):
    # The preview used to be drawn from the open writer; it now comes from the
    # finished, read-only result. Plotting closes its figure right after
    # show(), so the figure has to be inspected at the moment show() runs.
    import matplotlib.pyplot as plt

    drawn = []

    def capture(*args, **kwargs):
        images = [image for ax in plt.gcf().axes for image in ax.get_images()]
        drawn.append([image.get_array() for image in images])

    monkeypatch.setattr(plt, "show", capture)
    result = clip(single_band_float32, show_preview=True)
    try:
        assert result.ds.mode == "r"
        assert drawn, "show_preview never showed a figure"
        assert drawn[0], "the preview figure held no image"
        assert drawn[0][0].shape == result.get_shape()
    finally:
        plt.close("all")
        result.close()


def test_the_result_owns_its_memory_file(single_band_float32):
    # Reopened read-only, the dataset would be reading freed memory if the
    # MemoryFile were garbage-collected under it; the adapter must hold it.
    result = single_band_float32.add(1)
    assert result._adapter._memory_file is not None
    del single_band_float32  # the source going away must not matter
    assert result.read().shape == (1, 6, 6)
    result.close()
    assert result._adapter._memory_file.closed


def test_a_failing_fill_does_not_leak_the_memory_file(single_band_float32):
    def explode(dst):
        raise ZeroDivisionError("boom")

    with pytest.raises(ZeroDivisionError, match="boom"):
        RasterioAdapter.write_in_memory(single_band_float32.get_metadata(), explode)
