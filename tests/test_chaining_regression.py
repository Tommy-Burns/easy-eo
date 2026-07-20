"""Regression tests for the chainability bugs fixed

Two bugs made chaining onto a *derived* dataset fail:

* backend detection. ``mosaic``, ``stack``, ``clip_raster_with_vector``,
  ``clip_raster_with_bbox``, and ``reproject_raster`` gated on
  ``isinstance(backend, rio.DatasetReader)``. But the backend of any in-memory
  result (an algebra op, ``to_rasterio()`` of a NumPy-backed dataset, a prior
  clip/mosaic) is a ``rasterio.io.DatasetWriter``, which is *not* a
  ``DatasetReader``. So e.g. ``ds.add(1).clip_raster_with_bbox(...)`` raised
  ``TypeError`` even though the input was genuinely rasterio-backed. The bug
  hid because the existing tests fed these ops freshly file-loaded datasets
  (which really are ``DatasetReader``). These tests chain from a derived
  dataset instead.

* the ``eeo_raster_op`` decorator replaced any ``None`` return with
  ``self`` for chaining, so the bound ``ds.mosaic(other, save_path=...)``
  silently returned ``self`` instead of the documented ``None``. ``mosaic`` is
  now decorated with ``preserve_none=True``.
"""

import geopandas as gpd
from rasterio.io import DatasetReader
from shapely.geometry import box

from eeo.common import is_rasterio_backed


def _inset_bbox(ds):
    """Return a bbox one pixel inside the raster's bounds."""
    left, bottom, right, top = ds.get_bounds()
    return (left + 10, bottom + 10, right - 10, top - 10)


def test_derived_dataset_is_writer_backed_but_rasterio(single_band_float32):
    """The precondition that used to trip the old guard.

    A derived dataset's backend is a ``DatasetWriter`` (not a
    ``DatasetReader``), yet it is genuinely rasterio-backed.
    """
    derived = single_band_float32.add(1)
    assert not isinstance(derived.ds, DatasetReader)
    assert is_rasterio_backed(derived)


def test_chain_clip_bbox_on_derived(single_band_float32):
    derived = single_band_float32.add(1)
    clipped = derived.clip_raster_with_bbox(_inset_bbox(derived))
    assert clipped.get_width() > 0
    assert clipped.get_height() > 0


def test_chain_clip_vector_on_derived(single_band_float32):
    derived = single_band_float32.add(1)
    left, bottom, right, top = derived.get_bounds()
    gdf = gpd.GeoDataFrame(
        geometry=[box(left + 10, bottom + 10, right - 10, top - 10)],
        crs=derived.get_crs(),
    )
    clipped = derived.clip_raster_with_vector(gdf)
    assert clipped.get_count() == 1
    assert clipped.get_width() > 0


def test_chain_reproject_on_derived(single_band_float32):
    derived = single_band_float32.add(1)
    reprojected = derived.reproject_raster(target_crs=4326)
    assert reprojected.get_crs().to_epsg() == 4326


def test_chain_mosaic_on_derived(single_band_float32):
    a = single_band_float32.add(1)
    b = single_band_float32.add(2)
    mosaicked = a.mosaic(b)
    assert mosaicked.get_count() == 1
    assert mosaicked.get_width() >= single_band_float32.get_width()


def test_chain_stack_on_derived(single_band_float32):
    a = single_band_float32.add(1)
    b = single_band_float32.add(2)
    stacked = a.stack(b)
    assert stacked.get_count() == 2


def test_bound_mosaic_returns_dataset_when_not_saving(single_band_float32):
    """The chainable path still returns a dataset when no save_path is given."""
    a = single_band_float32.add(1)
    b = single_band_float32.add(2)
    result = a.mosaic(b)
    assert result is not a
    assert hasattr(result, "read")


def test_bound_mosaic_save_path_returns_none(single_band_float32, tmp_path):
    """the bound method honours mosaic's documented None-on-save.

    Without preserve_none the decorator would have swapped the None for
    ``self`` (``a``).
    """
    a = single_band_float32.add(1)
    b = single_band_float32.add(2)
    out_path = tmp_path / "bound_mosaic.tif"

    result = a.mosaic(b, save_path=str(out_path))

    assert result is None
    assert out_path.exists()
