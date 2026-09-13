"""Block-wise execution engine for pixel-wise raster operations.

A pixel-wise operation needs only the pixels it is currently computing, so it
does not have to hold a whole Sentinel-2 or Landsat scene in memory. This
module iterates a raster in windows, calls the operation on one window's worth
of every operand, and writes the result straight into the output — so peak
memory is set by the block size and the output, not by the input scene.

The engine is rasterio-specific on purpose: windowed reads and windowed writes
are backend features. Operations stay backend-agnostic by calling
:func:`apply_blockwise` and passing a plain NumPy callable, exactly as
``resample`` delegates decimated reads to the backend. Datasets on other
backends are promoted with ``to_rasterio()`` first.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import rasterio as rio
from rasterio.windows import Window

from eeo.common import (
    apply_nodata_mask,
    get_nodata,
    resolve_output_dtype,
    resolve_output_nodata,
)
from eeo.core.adapters import RasterioAdapter
from eeo.core.core import EEORasterDataset
from eeo.core.exceptions import AlignmentError, ValidationError
from eeo.core.types import StrPath

#: Pixels per block the engine aims for when the caller does not pick a block
#: shape. One mebipixel is ~4 MB per band in float32 — small enough that a
#: handful of operand blocks stay comfortably in cache-sized memory, large
#: enough that per-window overhead stays negligible.
DEFAULT_BLOCK_PIXELS = 1 << 20

_ALIGN_MISMATCH = (
    "block-wise execution needs every operand on the same grid as the target; "
    "operand {index} has shape {other_shape} / transform {other_transform} "
    "against {shape} / {transform}. Align it before calling apply_blockwise."
)


class BlockSource:
    """One operand of a block-wise computation.

    Wraps either a raster (read one window at a time) or a scalar (returned
    unchanged for every window), so :func:`apply_blockwise` can treat a mixed
    operand list uniformly. Build one with :meth:`from_dataset` or
    :meth:`from_scalar` rather than calling the constructor; each field is
    readable as an attribute of the same name.

    Parameters
    ----------
    dataset : EEORasterDataset or None
        The raster operand, already promoted to the rasterio backend, or None
        for a scalar operand.
    band : int or None
        1-based band index when the source reads a single band (yielding a 2-D
        block), or None to read every band (a 3-D block).
    value : int or float or None
        The scalar operand, or None for a raster operand.
    nodata : int, float, or None
        The operand's declared nodata value. Always None for a scalar, which
        carries no nodata.
    """

    __slots__ = ("band", "dataset", "nodata", "value")

    def __init__(
        self,
        *,
        dataset: EEORasterDataset | None,
        band: int | None,
        value: float | None,
        nodata: float | None,
    ) -> None:
        self.dataset = dataset
        self.band = band
        self.value = value
        self.nodata = nodata

    @classmethod
    def from_dataset(cls, ds: EEORasterDataset, *, band: int | None = None) -> BlockSource:
        """Wrap a raster operand, promoting it to the rasterio backend.

        Promotion matters for correctness, not just speed: the NumPy adapter's
        ``read`` ignores a ``window`` argument and would silently hand back the
        whole array for every block.

        Parameters
        ----------
        ds : EEORasterDataset
            Raster operand.
        band : int or None, default None
            1-based band index to read, or None to read every band.

        Returns
        -------
        BlockSource
            Source reading ``ds`` one window at a time.
        """
        return cls(dataset=ds.to_rasterio(), band=band, value=None, nodata=get_nodata(ds))

    @classmethod
    def from_scalar(cls, value: float) -> BlockSource:
        """Wrap a scalar operand.

        Parameters
        ----------
        value : int or float
            Scalar applied to every pixel.

        Returns
        -------
        BlockSource
            Source returning ``value`` for every window.
        """
        return cls(dataset=None, band=None, value=value, nodata=None)

    @property
    def is_raster(self) -> bool:
        """Whether this source reads pixels rather than yielding a scalar.

        Returns
        -------
        bool
            True for a raster operand, False for a scalar one.
        """
        return self.dataset is not None

    def read(self, window: Window) -> Any:
        """Return this operand's values over ``window``.

        Parameters
        ----------
        window : rasterio.windows.Window
            Window to read.

        Returns
        -------
        numpy.ndarray or int or float
            The window's pixels — 3-D ``(bands, rows, cols)``, or 2-D when the
            source names a single band — or the scalar, unchanged.
        """
        if self.dataset is None:
            return self.value
        if self.band is None:
            return self.dataset.read(window=window)
        return self.dataset.read(self.band, window=window)


def resolve_block_shape(
    shape: tuple[int, int], *, target_pixels: int = DEFAULT_BLOCK_PIXELS
) -> tuple[int, int]:
    """Choose a block shape for a raster of ``shape``, in pixels.

    Blocks span the full raster width and as many rows as fit the pixel
    budget. Full-width strips are sequential on disk for both striped and
    tiled sources, and they align with the stripes the output is written in,
    so no block is read or written twice. A raster smaller than the budget
    comes back as a single block, which keeps small rasters free of
    per-window overhead.

    Parameters
    ----------
    shape : tuple of int
        Raster shape as ``(height, width)`` in pixels.
    target_pixels : int, default ``DEFAULT_BLOCK_PIXELS``
        Approximate number of pixels a block should hold. A raster wider than
        this still yields one full row per block, since a block is never
        narrower than the raster.

    Returns
    -------
    tuple of int
        Block shape as ``(height, width)``.
    """
    height, width = shape
    rows = max(1, target_pixels // max(1, width))
    return min(height, rows), width


def block_windows(shape: tuple[int, int], block_shape: tuple[int, int]) -> Iterator[Window]:
    """Iterate the windows tiling a raster of ``shape`` in row-major order.

    Edge windows are truncated to the raster, so the windows partition it
    exactly — every pixel is covered once.

    Parameters
    ----------
    shape : tuple of int
        Raster shape as ``(height, width)`` in pixels.
    block_shape : tuple of int
        Block shape as ``(height, width)`` in pixels.

    Yields
    ------
    rasterio.windows.Window
        Each block's window, clipped to the raster edges.

    Raises
    ------
    ValidationError
        If either block dimension is not a positive integer.
    """
    height, width = shape
    block_height, block_width = block_shape
    if block_height < 1 or block_width < 1:
        raise ValidationError(
            f"block shape must be positive in both dimensions; got {block_shape!r}"
        )
    for row_off in range(0, height, block_height):
        rows = min(block_height, height - row_off)
        for col_off in range(0, width, block_width):
            cols = min(block_width, width - col_off)
            yield Window(col_off, row_off, cols, rows)


def _validate_grid(ds: EEORasterDataset, sources: Sequence[BlockSource]) -> None:
    """Raise if any raster source sits on a different grid than ``ds``."""
    shape, transform = ds.get_shape(), ds.get_transform()
    for index, source in enumerate(sources):
        other = source.dataset
        if other is None:
            continue
        if other.get_shape() != shape or other.get_transform() != transform:
            raise AlignmentError(
                _ALIGN_MISMATCH.format(
                    index=index,
                    other_shape=other.get_shape(),
                    other_transform=other.get_transform(),
                    shape=shape,
                    transform=transform,
                )
            )


def _as_bands(block: np.ndarray) -> np.ndarray:
    """Return ``block`` shaped ``(bands, rows, cols)``, adding a band axis if needed."""
    return block[np.newaxis, ...] if block.ndim == 2 else block


def apply_blockwise(
    ds: EEORasterDataset,
    compute: Callable[..., np.ndarray],
    *,
    sources: Sequence[BlockSource],
    fractional: bool = False,
    block_shape: tuple[int, int] | None = None,
    save_path: StrPath | None = None,
    driver: str = "GTiff",
) -> EEORasterDataset:
    """Run a pixel-wise computation over a raster one block at a time.

    For each window, every source is read, ``compute`` is called on those
    blocks, and the nodata & dtype contract is applied before the block is
    written to the output. Because the contract's output dtype and nodata
    value are fixed once — from the first block's dtype and the operands'
    declared nodata — every block lands in the same dtype and uses the same
    nodata marker, including blocks that happen to contain no nodata pixels.

    Parameters
    ----------
    ds : EEORasterDataset
        Target raster. Supplies the output grid (shape, transform, CRS) and
        the integer nodata sentinel. Every raster source must be on this grid.
    compute : callable
        Pixel-wise function called as ``compute(*blocks)`` with one block per
        entry of ``sources``, in order, returning an array over the same
        pixels. It must be element-wise: the value it computes for a pixel may
        not depend on any other pixel, or blocks will disagree at their seams.
    sources : sequence of BlockSource
        Operands to read per block, in the order ``compute`` expects them.
    fractional : bool, default False
        True for operations whose result is inherently fractional, which are
        written as float32 regardless of input dtype.
    block_shape : tuple of int or None, default None
        Block shape as ``(height, width)`` in pixels. Defaults to full-width
        strips of about ``DEFAULT_BLOCK_PIXELS`` pixels.
    save_path : str or path-like or None, default None
        Write the result to this path instead of to an in-memory raster, so
        the output never has to fit in memory either. The returned dataset
        reads from the file.
    driver : str, default "GTiff"
        GDAL driver for the output. The input's own driver is deliberately not
        reused: it records how the source was *read*, and a GDAL driver need
        not be able to create a dataset at all — many are read-only, and the
        failure would surface at the first block rather than up front.

    Returns
    -------
    EEORasterDataset
        New dataset holding the computed pixels, on ``ds``'s grid, with the
        band count ``compute`` produced and the nodata value the contract
        resolved.

    Raises
    ------
    AlignmentError
        If a raster source is not on ``ds``'s grid.
    ValidationError
        If ``sources`` is empty, or ``block_shape`` is not positive.

    Notes
    -----
    Streams block-wise: peak memory is one block per operand plus one output
    block, independent of the scene's size. With ``save_path`` the whole run
    is bounded that way; without it the in-memory output is also held, so a
    result larger than RAM needs ``save_path``.

    Examples
    --------
    >>> ds = load_array(np.arange(64, dtype="float32").reshape(8, 8), crs=4326)
    >>> doubled = apply_blockwise(
    ...     ds,
    ...     lambda block, factor: block * factor,
    ...     sources=[BlockSource.from_dataset(ds), BlockSource.from_scalar(2)],
    ... )
    >>> float(doubled.read()[0, 0, 1])
    2.0
    """
    if not sources:
        raise ValidationError("apply_blockwise needs at least one source to read")

    ds = ds.to_rasterio()
    _validate_grid(ds, sources)

    shape = ds.get_shape()
    windows = block_windows(shape, block_shape or resolve_block_shape(shape))

    meta = ds.get_metadata()
    ds_nodata = get_nodata(ds)
    raster_sources = [source for source in sources if source.is_raster]

    # The output profile is only knowable once a block has been computed: its
    # dtype comes from the computation and its band count from the result's
    # shape. So the destination is opened on the first block and reused after.
    dst: Any = None
    memfile: rio.io.MemoryFile | None = None
    out_dtype: Any = None
    out_nodata: float | None = None
    try:
        for window in windows:
            blocks = [source.read(window) for source in sources]
            result = _as_bands(compute(*blocks))

            if dst is None:
                out_dtype = resolve_output_dtype(result, fractional=fractional)
                out_nodata = resolve_output_nodata(
                    [source.nodata for source in raster_sources],
                    out_dtype=out_dtype,
                    ds_nodata=ds_nodata,
                )
                meta.update(
                    driver=driver,
                    dtype=out_dtype,
                    nodata=out_nodata,
                    count=result.shape[0],
                )
                if save_path is None:
                    memfile = rio.io.MemoryFile()
                    dst = memfile.open(**meta)
                else:
                    dst = rio.open(save_path, "w", **meta)

            operands = [
                (_as_bands(block), source.nodata)
                for block, source in zip(blocks, sources, strict=True)
                if source.is_raster
            ]
            masked = apply_nodata_mask(result, operands, out_dtype=out_dtype, out_nodata=out_nodata)
            # A whole-raster operand masking a result with fewer bands would
            # broadcast the mask up and silently widen the block, which the
            # write would then reject from inside rasterio. Say what happened.
            if masked.shape[0] != meta["count"]:
                raise ValidationError(
                    f"masking widened the result from {meta['count']} band(s) to "
                    f"{masked.shape[0]}: an operand covering more bands than "
                    "compute returns cannot mask it. Read that operand as a "
                    "single band, or return one result band per operand band."
                )
            dst.write(masked, window=window)
    except Exception:
        # Cleanup only — a half-written output would otherwise be left open,
        # and with save_path, left on disk. The error is always re-raised.
        if dst is not None:
            dst.close()
        if memfile is not None:
            memfile.close()
        raise

    assert dst is not None  # every raster has at least one block

    # Close before handing the result over, in memory as on disk: an open
    # writer leaves its blocks dirty in GDAL's cache, and a chain of results
    # left that way slows to a crawl once they outgrow it. See
    # RasterioAdapter.write_in_memory.
    dst.close()
    if save_path is None:
        assert memfile is not None
        return EEORasterDataset(adapter=RasterioAdapter.from_memory_file(memfile))
    return EEORasterDataset.from_path(save_path)
