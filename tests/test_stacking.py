"""Tests for the shared multi-source reader (eeo/io/_stacking.py).

The STAC suite already exercises this code end to end, but only through
``STACItem.load``, which validates its asset list first and always hands over
sources that sit on tidy grids. These tests reach the module directly, covering
the fallback branches that decide between an exact windowed read and a warp —
the paths a local-product reader will hit as soon as it stacks a 20 m band onto
a 10 m one.

Everything is written to a temporary directory; no network, no fixtures outside
the repo.
"""

import numpy as np
import pytest
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from eeo.core.exceptions import ValidationError
from eeo.io._stacking import Grid, _aligned_window, read_onto_common_grid

CRS = "EPSG:32633"
ORIGIN = (500000.0, 5000000.0)
SIZE = 32
RES = 10.0


def write(path, *, origin=ORIGIN, size=SIZE, res=RES, count=1, crs=CRS, fill=None):
    """Write a small synthetic GeoTIFF and return its path as a string."""
    if fill is None:
        band = np.arange(size * size, dtype="uint16").reshape(size, size)
        data = np.stack([band + step for step in range(count)])
    else:
        data = np.full((count, size, size), fill, dtype="uint16")
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=count,
        dtype="uint16",
        crs=crs,
        transform=from_origin(origin[0], origin[1], res, res),
        nodata=0,
    ) as dst:
        dst.write(data)
    return str(path)


def grid_of(href):
    """Build the Grid a source defines when read whole."""
    with rio.open(href) as src:
        return Grid(src.crs, src.transform, src.width, src.height)


class TestAlignedWindowFallsBackToWarp:
    """_aligned_window returns None wherever an exact read would be wrong.

    Each None sends the read down the WarpedVRT path instead. These are the
    branches the STAC suite never reached, and a silent regression here would
    resample when it should not (or the reverse) rather than raise.
    """

    def test_matching_grid_reads_exactly(self, tmp_path):
        href = write(tmp_path / "a.tif")
        with rio.open(href) as src:
            window = _aligned_window(src, grid_of(href))
        assert window is not None
        assert (window.col_off, window.row_off) == (0, 0)

    def test_different_crs_warps(self, tmp_path):
        href = write(tmp_path / "utm34.tif", crs="EPSG:32634")
        target = grid_of(write(tmp_path / "utm33.tif"))
        with rio.open(href) as src:
            assert _aligned_window(src, target) is None

    def test_different_resolution_warps(self, tmp_path):
        href = write(tmp_path / "coarse.tif", res=20.0)
        target = grid_of(write(tmp_path / "fine.tif", res=10.0))
        with rio.open(href) as src:
            assert _aligned_window(src, target) is None

    def test_sub_pixel_column_offset_warps(self, tmp_path):
        # Origin shifted half a pixel east: the window lands on x.5 columns.
        href = write(tmp_path / "half_east.tif", origin=(ORIGIN[0] + RES / 2, ORIGIN[1]))
        target = grid_of(write(tmp_path / "target.tif"))
        with rio.open(href) as src:
            assert _aligned_window(src, target) is None

    def test_sub_pixel_row_offset_warps(self, tmp_path):
        # Origin shifted half a pixel south: whole columns, fractional rows.
        href = write(tmp_path / "half_south.tif", origin=(ORIGIN[0], ORIGIN[1] - RES / 2))
        target = grid_of(write(tmp_path / "target.tif"))
        with rio.open(href) as src:
            assert _aligned_window(src, target) is None

    def test_window_starting_before_the_source_warps(self, tmp_path):
        # Source begins one pixel east/south of the grid, so the aligned window
        # would need a negative offset and come back short.
        href = write(tmp_path / "inset.tif", origin=(ORIGIN[0] + RES, ORIGIN[1] - RES))
        target = grid_of(write(tmp_path / "target.tif"))
        with rio.open(href) as src:
            assert _aligned_window(src, target) is None

    def test_window_running_past_the_source_warps(self, tmp_path):
        # Same origin but a smaller source: the window overhangs its edge.
        href = write(tmp_path / "small.tif", size=SIZE // 2)
        target = grid_of(write(tmp_path / "target.tif"))
        with rio.open(href) as src:
            assert _aligned_window(src, target) is None


class TestReadOntoCommonGrid:
    """The first source sets the grid; the rest are placed onto it."""

    def test_empty_sources_rejected(self):
        # Unreachable through STACItem.load, which validates its asset list
        # first, but reachable for any other caller of this module.
        with pytest.raises(ValidationError, match="at least one source"):
            read_onto_common_grid([], bbox=None, resampling=Resampling.nearest)

    def test_single_source_keeps_its_own_grid(self, tmp_path):
        href = write(tmp_path / "b04.tif")
        array, grid, nodata, names = read_onto_common_grid(
            [(href, "B04")], bbox=None, resampling=Resampling.nearest
        )
        assert array.shape == (1, SIZE, SIZE)
        assert (grid.width, grid.height) == (SIZE, SIZE)
        assert nodata == 0
        assert names == ["B04"]

    def test_stacking_preserves_order_and_names(self, tmp_path):
        first = write(tmp_path / "b04.tif", fill=7)
        second = write(tmp_path / "b08.tif", fill=9)
        array, _, _, names = read_onto_common_grid(
            [(first, "B04"), (second, "B08")], bbox=None, resampling=Resampling.nearest
        )
        assert array.shape == (2, SIZE, SIZE)
        assert names == ["B04", "B08"]
        assert array[0].max() == 7 and array[1].max() == 9

    def test_coarser_source_is_warped_onto_the_fine_grid(self, tmp_path):
        fine = write(tmp_path / "fine.tif", res=RES, fill=1)
        coarse = write(tmp_path / "coarse.tif", res=RES * 2, size=SIZE // 2, fill=5)
        array, grid, _, names = read_onto_common_grid(
            [(fine, "B04"), (coarse, "B11")], bbox=None, resampling=Resampling.nearest
        )
        # The 20 m band lands on the 10 m grid rather than keeping its own size.
        assert array.shape == (2, SIZE, SIZE)
        assert (grid.width, grid.height) == (SIZE, SIZE)
        assert names == ["B04", "B11"]
        assert array[1].max() == 5

    def test_multiband_source_numbers_its_bands(self, tmp_path):
        href = write(tmp_path / "stack.tif", count=3)
        _, _, _, names = read_onto_common_grid(
            [(href, "TCI")], bbox=None, resampling=Resampling.nearest
        )
        assert names == ["TCI_1", "TCI_2", "TCI_3"]

    def test_nodata_comes_from_the_first_source(self, tmp_path):
        first = write(tmp_path / "a.tif")
        second = write(tmp_path / "b.tif")
        with rio.open(second, "r+") as dst:
            dst.nodata = 65535
        _, _, nodata, _ = read_onto_common_grid(
            [(first, "A"), (second, "B")], bbox=None, resampling=Resampling.nearest
        )
        assert nodata == 0

    def test_mixed_dtypes_promote(self, tmp_path):
        first = write(tmp_path / "u16.tif")
        second = str(tmp_path / "f32.tif")
        with rio.open(first) as src:
            profile = src.profile
        profile.update(dtype="float32")
        with rio.open(second, "w", **profile) as dst:
            dst.write(np.full((1, SIZE, SIZE), 0.5, dtype="float32"))
        array, _, _, _ = read_onto_common_grid(
            [(first, "A"), (second, "B")], bbox=None, resampling=Resampling.nearest
        )
        assert array.dtype == np.result_type(np.uint16, np.float32)

    def test_per_source_resampling_length_must_match(self, tmp_path):
        first = write(tmp_path / "a.tif")
        second = write(tmp_path / "b.tif", res=RES * 2, size=SIZE // 2)
        with pytest.raises(ValidationError, match="one resampling method per source"):
            read_onto_common_grid(
                [(first, "A"), (second, "B")],
                bbox=None,
                resampling=[Resampling.nearest],
            )

    def test_per_source_resampling_is_applied(self, tmp_path):
        # A categorical source travelling with a continuous one: the second
        # gets nearest even though the request as a whole says bilinear.
        fine = write(tmp_path / "fine.tif", res=RES, fill=1)
        classes = str(tmp_path / "classes.tif")
        with rio.open(fine) as src:
            profile = src.profile
        profile.update(
            width=SIZE // 2,
            height=SIZE // 2,
            transform=from_origin(ORIGIN[0], ORIGIN[1], RES * 2, RES * 2),
        )
        data = np.zeros((1, SIZE // 2, SIZE // 2), dtype="uint16")
        data[0, : SIZE // 4] = 4
        data[0, SIZE // 4 :] = 9
        with rio.open(classes, "w", **profile) as dst:
            dst.write(data)
        array, _, _, _ = read_onto_common_grid(
            [(fine, "A"), (classes, "SCL")],
            bbox=None,
            resampling=[Resampling.bilinear, Resampling.nearest],
        )
        assert set(np.unique(array[1])) <= {4, 9}
