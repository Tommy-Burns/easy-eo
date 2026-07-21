"""Clipping operations by bounding box or vector geometry."""

import os

import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
from rasterio.windows import from_bounds

from eeo.common import is_rasterio_backed
from eeo.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op


@eeo_raster_op
def clip_raster_with_vector(
    ds: EEORasterDataset,
    vector_file: gpd.GeoDataFrame | str,
    *,
    crop: bool = True,
    pad: bool = False,
    all_touched: bool = False,
    invert: bool = False,
    nodata: int | float | None = None,
    show_preview: bool = False,
    plot_kwargs: dict | None = None,
) -> EEORasterDataset:
    """Clip a raster to vector geometries.

    Pixels outside the geometries are dropped (``crop=True``) or set to
    nodata (``crop=False``). The vector is reprojected to the raster's CRS
    when needed.

    Parameters
    ----------
    ds : EEORasterDataset
        Raster to clip. Must be backed by rasterio.
    vector_file : geopandas.GeoDataFrame or str
        Clip geometries, either a GeoDataFrame or a path to a vector file
        readable by GeoPandas.
    crop : bool, default True
        If True, crop the output to the geometries' bounding box; if False,
        keep the raster extent and set outside pixels to nodata.
    pad : bool, default False
        If True, pad the crop by half a pixel on each side.
    all_touched : bool, default False
        If True, include every pixel touched by the geometries; if False,
        include only pixels whose centre falls inside.
    invert : bool, default False
        If True, mask pixels inside the geometries instead of outside.
    nodata : int or float or None, default None
        Fill value for masked pixels. If None, the raster's existing nodata
        value is used.
    show_preview : bool, default False
        If True, plot the clipped raster before returning.
    plot_kwargs : dict or None, default None
        Keyword arguments forwarded to ``plot_raster`` when
        ``show_preview=True``.

    Returns
    -------
    EEORasterDataset
        New rasterio-backed dataset covering the clipped area, in the same
        dtype as ``ds``, carrying ``nodata`` (or the raster's existing nodata
        value) in its metadata.

    Raises
    ------
    TypeError
        If ``ds`` is not backed by rasterio, or ``vector_file`` is neither a
        GeoDataFrame nor a valid file path.

    Notes
    -----
    Reads the clipped region into memory via ``rasterio.mask.mask``.

    Examples
    --------
    >>> import geopandas as gpd
    >>> boundary = gpd.read_file("aoi.geojson")
    >>> clipped = ds.clip_raster_with_vector(boundary)
    """
    # Ensure clipping for only rasterio-backend datasets
    if not is_rasterio_backed(ds):
        raise TypeError("Clipping is only allowed on rasterio backend rasters")

    # Load vector data
    if isinstance(vector_file, gpd.GeoDataFrame):
        gdf = vector_file
    elif isinstance(vector_file, str) and os.path.isfile(vector_file):
        gdf = gpd.read_file(vector_file)
    else:
        raise TypeError("vector_file must be a GeoDataFrame or a valid file path")

    # Reproject vector geometries if needed
    if gdf.crs != ds.get_crs():
        gdf = gdf.to_crs(ds.get_crs())

    shapes = gdf.geometry.values

    # Perform masking
    clipped, clipped_transform = mask(
        ds.ds,
        shapes,
        crop=crop,
        pad=pad,
        all_touched=all_touched,
        invert=invert,
        nodata=nodata,
    )

    # Update metadata
    meta = ds.get_metadata().copy()
    meta.update(
        height=clipped.shape[1],
        width=clipped.shape[2],
        transform=clipped_transform,
        count=clipped.shape[0],
    )

    if nodata is not None:
        meta["nodata"] = nodata

    # Write to MemoryFile
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(clipped)

    # Optional preview
    if show_preview:
        EEORasterDataset.from_rasterio(out_ds).plot_raster(**(plot_kwargs or {}))

    return EEORasterDataset.from_rasterio(out_ds)


@eeo_raster_op
def clip_raster_with_bbox(
    ds: EEORasterDataset, bbox: tuple | list, plot_kwargs=None, show_preview: bool = False
) -> EEORasterDataset:
    """Clip a raster to a bounding box.

    Parameters
    ----------
    ds : EEORasterDataset
        Raster to clip. Must be backed by rasterio.
    bbox : tuple or list
        Bounding box as ``(minx, miny, maxx, maxy)`` in the raster's CRS
        units.
    plot_kwargs : dict or None, default None
        Keyword arguments forwarded to ``plot_raster`` when
        ``show_preview=True``.
    show_preview : bool, default False
        If True, plot the clipped raster before returning.

    Returns
    -------
    EEORasterDataset
        New rasterio-backed dataset covering the bounding box, in the same
        dtype as ``ds``, carrying its nodata value unchanged.

    Raises
    ------
    TypeError
        If ``ds`` is not backed by rasterio.
    ValueError
        If ``bbox`` is not four values, or does not intersect the raster.

    Notes
    -----
    Reads only the windowed region into memory, not the whole raster.

    Examples
    --------
    >>> clipped = ds.clip_raster_with_bbox((500000, 4100000, 510000, 4110000))
    """
    # Ensure rasterio backend
    if not is_rasterio_backed(ds):
        raise TypeError("Clipping is only allowed on rasterio backend rasters")

    # Validate bbox
    if not (isinstance(bbox, (tuple, list)) and len(bbox) == 4):
        raise ValueError("bbox must be (minx, miny, maxx, maxy)")

    minx, miny, maxx, maxy = bbox

    # Compute window
    window = from_bounds(minx, miny, maxx, maxy, ds.ds.transform)
    window = window.round_offsets().round_lengths()

    # Ensure correct bbox
    if window.width <= 0 or window.height <= 0:
        raise ValueError(
            "Bounding box does not intersect raster extent. "
            f"Raster bounds: {ds.get_bounds()}, bbox: {bbox}"
        )

    transform = rio.windows.transform(window, ds.ds.transform)

    # Read clipped data
    clipped = ds.ds.read(window=window)

    # Update metadata
    meta = ds.get_metadata().copy()
    meta.update(
        height=clipped.shape[1],
        width=clipped.shape[2],
        transform=transform,
    )

    # Write to MemoryFile
    memfile = rio.io.MemoryFile()
    dataset = memfile.open(**meta)
    dataset.write(clipped)

    if show_preview:
        EEORasterDataset.from_rasterio(dataset).plot_raster(**(plot_kwargs or {}))

    return EEORasterDataset.from_rasterio(dataset)
