"""Raster merging operations: mosaic and band stack."""

from collections.abc import Iterable

import numpy as np
import rasterio as rio
from rasterio.merge import merge

from eeo.common import is_rasterio_backed, normalize_resampling_method
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op


@eeo_raster_op(preserve_none=True)
def mosaic(
    ds: EEORasterDataset,
    others: EEORasterDataset | Iterable[EEORasterDataset],
    *,
    resampling_method: str = "nearest",
    save_path: str | None = None,
    auto_reproject: bool = False,
    **kwargs,
) -> EEORasterDataset | None:
    """Mosaic one or more rasters into a single raster.

    Parameters
    ----------
    ds : EEORasterDataset
        Base raster; also determines the target CRS.
    others : EEORasterDataset or Iterable[EEORasterDataset]
        One or more rasters to mosaic with ``ds``.
    resampling_method : str or rasterio.enums.Resampling, default "nearest"
        Resampling method used by ``rasterio.merge.merge`` where overlapping
        pixels require resampling.
    save_path : str or None, default None
        If given, writes the mosaic to this path and returns None instead of
        an ``EEORasterDataset``.
    auto_reproject : bool, default False
        If True, reproject any raster in ``others`` whose CRS differs from
        ``ds`` before mosaicking. If False, a CRS mismatch raises
        ``ValueError``.
    **kwargs
        Additional keyword arguments forwarded to ``rasterio.merge.merge``.

    Returns
    -------
    EEORasterDataset or None
        New rasterio-backed mosaic in the dtype ``rasterio.merge.merge``
        produces (the inputs' common dtype), carrying ``ds``'s nodata value;
        or None if ``save_path`` was given. Overlapping nodata pixels are
        filled from other tiles where possible.

    Raises
    ------
    TypeError
        If ``ds`` is not backed by rasterio.
    ValueError
        If ``others`` is empty, or a CRS mismatch is found and
        ``auto_reproject=False``.

    Notes
    -----
    Loads every input tile into memory via ``rasterio.merge.merge`` rather
    than streaming block-wise. With ``save_path`` the mosaic is written to
    disk as a side effect and None is returned.

    Examples
    --------
    >>> mosaicked = ds.mosaic([ds_tile_2, ds_tile_3])
    """
    # Ensure mosaic for only rasterio-backend datasets
    if not is_rasterio_backed(ds):
        raise TypeError("Mosaicking is only allowed on rasterio backend rasters")

    # normalize resampling
    resampling_method = normalize_resampling_method(resampling_method)

    # normalize inputs to list
    others = [others] if isinstance(others, EEORasterDataset) else list(others)

    if not others:
        raise ValueError("At least one raster must be provided for mosaicking")

    # CRS validation
    src_datasets: list[EEORasterDataset] = [ds]
    target_crs = ds.get_crs()
    for obj in others:
        if obj.get_crs() != target_crs:
            if auto_reproject:
                # target_crs is a rasterio CRS; reproject_raster takes its
                # keyword-only target_crs as int/str/pyproj.CRS, so WKT is passed here rather.
                obj = obj.reproject_raster(target_crs=target_crs.to_wkt())
            else:
                raise ValueError(
                    "All rasters must have the same CRS for mosaicking. "
                    "Set auto_reproject=True to allow reprojection."
                )

        src_datasets.append(obj)

    # extract datasets and perform mosaics
    datasets = [d.ds for d in src_datasets]
    mosaic_data, out_transform = merge(datasets, resampling=resampling_method, **kwargs)

    # modify metadata
    meta = ds.get_metadata().copy()
    meta.update(
        transform=out_transform,
        height=mosaic_data.shape[1],
        width=mosaic_data.shape[2],
        count=mosaic_data.shape[0],
        dtype=mosaic_data.dtype,
    )

    # write to memory file
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(mosaic_data)

    result = EEORasterDataset.from_rasterio(out_ds)

    # save or return EEORasterDataset
    if save_path is not None:
        result.save_raster(path=save_path)
        return None

    return result


@eeo_raster_op
def stack(
    ds: EEORasterDataset,
    others: EEORasterDataset | Iterable[EEORasterDataset],
) -> EEORasterDataset:
    """Stack rasters band-wise into a single multi-band raster.

    The bands of ``ds`` come first, followed by the bands of each dataset in
    ``others`` in order. All inputs must already share the same CRS,
    transform, and shape (no auto-alignment); this is spectral stacking, kept
    deliberately distinct from temporal stacking.

    Parameters
    ----------
    ds : EEORasterDataset
        Base raster; its bands lead the output.
    others : EEORasterDataset or Iterable[EEORasterDataset]
        One or more rasters whose bands are appended, in order.

    Returns
    -------
    EEORasterDataset
        New rasterio-backed dataset whose band count is the sum of all inputs'
        band counts, in the common dtype ``numpy.vstack`` promotes to, carrying
        ``ds``'s nodata value.

    Raises
    ------
    TypeError
        If ``ds`` is not backed by rasterio.
    ValueError
        If ``others`` is empty, or any input's CRS, transform, or shape
        differs from ``ds``.

    Notes
    -----
    Reads every input fully into memory rather than streaming block-wise.
    Nodata pixels are carried through as ordinary values; the nodata value in
    the metadata is preserved.

    Examples
    --------
    >>> rgb = ds_red.stack([ds_green, ds_blue])
    """
    # Ensure stack for only rasterio-backend datasets
    if not is_rasterio_backed(ds):
        raise TypeError("Stacking is only allowed on rasterio backend rasters")

    # normalize inputs
    others = [others] if isinstance(others, EEORasterDataset) else list(others)

    if not others:
        raise ValueError("At least one raster must be provided for stacking")

    # alignment checks
    for item in others:
        if (
            item.get_crs() != ds.get_crs()
            or item.get_transform() != ds.get_transform()
            or item.get_shape() != ds.get_shape()
        ):
            raise ValueError("All rasters must have identical CRS, transform, and shape")

    # read data
    arrays = [ds.read()]
    for obj in others:
        arrays.append(obj.read())

    # stack the arrays
    stacked = np.vstack(arrays)

    # metadata update
    meta = ds.get_metadata().copy()
    meta.update(count=stacked.shape[0], dtype=stacked.dtype)

    # save to memory file
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(stacked)

    return EEORasterDataset.from_rasterio(out_ds)
