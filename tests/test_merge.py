import numpy as np
import pytest
import rasterio as rio
from affine import Affine
from rasterio.crs import CRS

from eeo import load_raster
from eeo.core.exceptions import CRSMismatchError
from eeo.ops.merge import mosaic


def _write_tile(tmp_path, name, origin_x, value, res=10.0):
    path = tmp_path / name
    array = np.full((3, 3), value, dtype=np.float32)
    transform = Affine.translation(origin_x, 30.0) * Affine.scale(res, -res)
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(32633),
        transform=transform,
    ) as dst:
        dst.write(array, 1)
    return load_raster(str(path))


def test_mosaic_returns_dataset(tmp_path):
    left = _write_tile(tmp_path, "left.tif", origin_x=0.0, value=1.0)
    right = _write_tile(tmp_path, "right.tif", origin_x=30.0, value=2.0)

    mosaicked = left.mosaic(right)

    assert hasattr(mosaicked, "read")
    assert mosaicked.get_count() == 1
    assert mosaicked.get_width() >= 6


def test_mosaic_save_path_writes_file_and_returns_none(tmp_path):
    left = _write_tile(tmp_path, "left.tif", origin_x=0.0, value=1.0)
    right = _write_tile(tmp_path, "right.tif", origin_x=30.0, value=2.0)
    out_path = tmp_path / "mosaic_out.tif"

    result = mosaic(left, right, save_path=str(out_path))

    assert result is None
    assert out_path.exists()
    with rio.open(out_path) as saved:
        assert saved.count == 1
        assert saved.width >= 6


def test_mosaic_auto_reproject_across_crs(single_band_float32):
    """auto_reproject mosaics rasters whose CRS differs from the base.

    the auto_reproject path passed a rasterio CRS
    positionally to ``reproject_raster`` (keyword-only, int/str/pyproj.CRS
    only), which raised TypeError and meant the feature never worked.
    """
    # Same footprint as the base, re-expressed in EPSG:4326.
    other = single_band_float32.reproject_raster(target_crs=4326)
    assert other.get_crs() != single_band_float32.get_crs()

    mosaicked = single_band_float32.mosaic(other, auto_reproject=True)

    assert mosaicked.get_count() == 1
    assert mosaicked.get_crs() == single_band_float32.get_crs()


def test_mosaic_crs_mismatch_without_auto_reproject_raises(single_band_float32):
    """A CRS mismatch raises CRSMismatchError when auto_reproject is off."""
    other = single_band_float32.reproject_raster(target_crs=4326)
    with pytest.raises(CRSMismatchError, match="share the CRS"):
        single_band_float32.mosaic(other)


def test_stack_combines_bands(tmp_path):
    a = _write_tile(tmp_path, "a.tif", origin_x=0.0, value=1.0)
    b = _write_tile(tmp_path, "b.tif", origin_x=0.0, value=2.0)

    stacked = a.stack(b)

    assert stacked.get_count() == 2
