"""Reprojection to a target coordinate reference system."""

import pyproj
import rasterio as rio
from rasterio.warp import Resampling, calculate_default_transform, reproject

from eeo.common import get_nodata, is_rasterio_backed, normalize_resampling_method
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op
from eeo.core.exceptions import BackendError, ValidationError


@eeo_raster_op
def reproject_raster(
    ds: EEORasterDataset,
    *,
    target_crs: int | str | pyproj.CRS,
    resampling_method: Resampling = Resampling.nearest,
) -> EEORasterDataset:
    """Reproject a raster to a new coordinate reference system.

    Parameters
    ----------
    ds : EEORasterDataset
        Raster to reproject. Must be backed by rasterio.
    target_crs : int or str or pyproj.CRS
        Destination CRS as an EPSG code, a PROJ/WKT string, or a
        ``pyproj.CRS``.
    resampling_method : rasterio.enums.Resampling, default Resampling.nearest
        Resampling method used to warp the pixels. Defaults to nearest
        neighbour so categorical values and nodata edges are not blended.

    Returns
    -------
    EEORasterDataset
        New rasterio-backed dataset in ``target_crs``, in the same dtype as
        ``ds``, with a recomputed transform, width, and height; the nodata
        value is carried over unchanged.

    Raises
    ------
    BackendError
        If ``ds`` is not backed by rasterio.
    ValidationError
        If ``target_crs`` cannot be interpreted as a CRS.

    Notes
    -----
    Warps band-by-band with ``rasterio.warp.reproject`` through rasterio band
    handles, so the full source array is never materialized at once; the
    warped output is held in an in-memory dataset. Source nodata pixels are
    honoured and border pixels exposed by the warp are filled with the nodata
    value; if the raster declares no nodata, those border pixels are filled
    with 0.

    Examples
    --------
    >>> reprojected = ds.reproject_raster(target_crs=4326)
    """
    # Ensure reprojection for only rasterio-backend datasets
    if not is_rasterio_backed(ds):
        raise BackendError(
            "reproject requires a rasterio-backed dataset; this dataset uses "
            "the NumPy backend. Call .to_rasterio() first."
        )

    # Normalize resampling method
    resampling_method = normalize_resampling_method(resampling_method)
    # Normalise CRS
    if isinstance(target_crs, (int, str)):
        crs = pyproj.CRS.from_user_input(target_crs)
    else:
        crs = target_crs

    if not isinstance(crs, pyproj.CRS):
        raise ValidationError(
            f"target_crs must be an int, str, or pyproj.CRS; got {type(target_crs).__name__}"
        )

    # Get dataset bounds
    left, bottom, right, top = ds.get_bounds()

    # Compute transform and new size
    transform, width, height = calculate_default_transform(
        src_crs=ds.get_crs(),
        dst_crs=crs,
        width=ds.get_width(),
        height=ds.get_height(),
        left=left,
        bottom=bottom,
        right=right,
        top=top,
    )

    # Update the metadata
    meta = ds.get_metadata()
    meta.update({"crs": crs, "transform": transform, "width": width, "height": height})

    # return in-memory dataset
    memfile = rio.io.MemoryFile()
    dataset = memfile.open(**meta)

    # Pass the nodata value both ways so source nodata is not warped into
    # valid data and border pixels exposed by the warp are filled with it.
    nodata = get_nodata(ds)
    for i in range(1, ds.get_count() + 1):
        reproject(
            source=rio.band(ds.ds, i),
            destination=rio.band(dataset, i),
            src_transform=ds.get_transform(),
            src_crs=ds.get_crs(),
            dst_transform=transform,
            dst_crs=crs,
            src_nodata=nodata,
            dst_nodata=nodata,
            resampling=resampling_method,
        )
    return EEORasterDataset.from_rasterio(dataset)
