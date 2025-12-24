from typing import Union
import geopandas as gpd

from eeo.core.core import EEORasterDataset


def clip_raster_with_vector(
        ds: EEORasterDataset,
        vector_file: Union[gpd.GeoDataFrame, str],
        *,
        crop: bool = True,
        pad: bool = False,
        all_touched: bool = False,
        invert: bool = False,
        nodata: Union[int, float, None] = None,
        show_preview: bool = False,
        plot_kwargs: dict | None = None,
) -> Union[None, "EEORasterDataset"]:
    """
    Clip a raster using vector geometries.

    The raster can be cropped to the geometry bounds or masked while preserving
    the original extent. Supports advanced Rasterio masking options via keyword arguments.

    :param ds: The EEORasterDataset object.
    :param vector_file: (GeoDataFrame or str)
        Vector geometries for clipping. If a string is provided, it must be a
        valid path to a vector file readable by GeoPandas.
    :param crop: (bool, default=True)
        ``True``: Crop the output raster to the minimal bounding box of the geometries.
        ``False``: Retain the original raster extent; pixels outside the geometries are set to nodata.
    :param pad: (bool, default=False)
        If ``crop=True``, expands the output extent to fully include edge pixels.
    :param all_touched: (bool, default=False)
        ``True``: Include all pixels touched by geometries.
        ``False``: Include only pixels whose center lies within geometries.
    :param invert: (bool, default=False)
        ``True``: Mask pixels inside the geometries instead of outside.
    :param nodata: (int or float, optional)
        Value assigned to masked pixels. If ``None``, uses the raster’s existing nodata value.
    :param show_preview: (bool, default=False)
        If ``True``, displays a preview of the clipped raster.
    :param plot_kwargs: (dict, optional)
        Additional keyword arguments passed to rasterio.plot.show. Common options:
            - title (str)
            - cmap (str or Colormap)
            - vmin, vmax (float)
            - ax (matplotlib.axes.Axes)
    :param **kwargs: Additional keyword arguments forwarded to rasterio.mask.mask,
        excluding parameters already exposed explicitly (crop, pad, all_touched, invert, nodata).

    :return: EEORasterDataset
    """
    ...


def clip_raster_with_bbox(
        ds: EEORasterDataset,
        bbox: tuple | list,
        plot_kwargs=None,
        show_preview: bool = False
)-> "EEORasterDataset":
    """
    Clip a raster using a bounding box.

    The raster will be subsetted to the specified bounding box. The bounding
    box must be in the same CRS as the raster.

    :param ds: The EEORasterDataset object.
    :param bbox: (tuple or list)
        Bounding box coordinates as (minx, miny, maxx, maxy).
    :param show_preview: (bool, default=False)
        If ``True``, displays a preview of the clipped raster.
    :param plot_kwargs: (dict, optional)
        Additional keyword arguments passed to rasterio.plot.show. Common options:
            - title (str)
            - cmap (str or Colormap)
            - vmin, vmax (float)
            - ax (matplotlib.axes.Axes)

    :return: EEORasterDataset
    """
    ...
