import pyproj
import rasterio as rio
from rasterio.warp import Resampling, calculate_default_transform, reproject

from eeo.common import is_rasterio_backed, normalize_resampling_method
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op


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
    TypeError
        If ``ds`` is not backed by rasterio, or ``target_crs`` cannot be
        interpreted as a CRS.

    Notes
    -----
    Reads the full raster into memory and warps it band-by-band with
    ``rasterio.warp.reproject``.

    Examples
    --------
    >>> reprojected = ds.reproject_raster(target_crs=4326)
    """
    # Ensure reprojection for only rasterio-backend datasets
    if not is_rasterio_backed(ds):
        raise TypeError("Reprojection is only allowed on rasterio backend rasters")

    # Normalize resampling method
    resampling_method = normalize_resampling_method(resampling_method)
    # Normalise CRS
    if isinstance(target_crs, (int, str)):
        crs = pyproj.CRS.from_user_input(target_crs)
    else:
        crs = target_crs

    if not isinstance(crs, pyproj.CRS):
        raise TypeError("Invalid CRS. Must be int, str, or pyproj.CRS")

    # Get dataset bounds
    left, bottom, right, top = ds.get_bounds()

    # Compute transform and new size
    transform, width, height = calculate_default_transform(
        src_crs=ds.get_crs(),
        dst_crs=crs,
        width=ds.get_width(),
        height=ds.get_shape()[1],
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

    for i in range(1, ds.get_count() + 1):
        reproject(
            source=rio.band(ds.ds, i),
            destination=rio.band(dataset, i),
            src_transform=ds.get_transform(),
            src_crs=ds.get_crs(),
            dst_transform=transform,
            dst_crs=crs,
            resampling=resampling_method,
        )
    return EEORasterDataset.from_rasterio(dataset)
