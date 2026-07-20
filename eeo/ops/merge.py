from collections.abc import Iterable

import numpy as np
import rasterio as rio
from rasterio.merge import merge

from eeo.common import normalize_resampling_method
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op


@eeo_raster_op
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
        New rasterio-backed mosaic in the same dtype ``rasterio.merge.merge``
        produces, or None if ``save_path`` was given.

    Raises
    ------
    TypeError
        If ``ds`` is not backed by rasterio.
    ValueError
        If ``others`` is empty, or a CRS mismatch is found and
        ``auto_reproject=False``.

    Examples
    --------
    >>> mosaicked = ds.mosaic([ds_tile_2, ds_tile_3])
    """
    # Ensure mosaic for only rasterio-backend datasets
    backend = ds._adapter.backend
    if not isinstance(backend, rio.DatasetReader):
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
                obj = obj.reproject_raster(target_crs)
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
    # Ensure stack for only rasterio-backend datasets
    backend = ds._adapter.backend
    if not isinstance(backend, rio.DatasetReader):
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
