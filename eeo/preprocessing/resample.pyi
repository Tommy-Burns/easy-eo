from typing import Optional, Union

from rasterio.enums import Resampling

from eeo.core.core import EEORasterDataset
from eeo.core.types import ResamplingMethod


def resample(ds: EEORasterDataset,
             *,
             size: Optional[tuple[int, int]] = None,
             scale_factor: Optional[float] = None,
             resolution: Optional[tuple[float, float]] = None,
             save_path: str = None,
             resampling_method: Union[Resampling, ResamplingMethod] = "nearest",
             save_driver: str = "GTiff",
             plot_kwargs=None,
             show_preview: bool = False
             ) -> Union[EEORasterDataset, None]:
    """
    Resample a raster to a different resolution using a scaling factor

    :param ds: The EEORasterDataset object.
    :param size: tuple[int, int]. Output pixel dimensions.
    :param scale_factor: float. Uniform scale factor (e.g. 0.5 halves the resolution).
    :param resolution: tuple[float, float]. Output spatial resolution (xres, yres) in CRS units.
    :param save_path: (str, optional): Path to save the clipped raster. If None, returns a new EEORasterDataset. Mutually exclusive with ``show_preview``
    :param resampling_method: ``rasterio.enums.Resampling`` | Literal
        The resampling method used during reprojection. May be either a
        `Resampling` enum value or one of the following string literals:
        ``"nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average",
        "mode", "max", "min", "med", "q1", "q3"``.
        Default is ``nearest``.
    :param show_preview: (str, optional): Raster driver to use when saving. Defaults to "GTiff".
    :param save_driver: (bool, optional): If True, show a preview of the clipped raster. Defaults to False.
    :param plot_kwargs: Additional keyword arguments passed to rasterio.plot.show.
        Common options include:
            - title (str): Title of the plot.
            - cmap (str or Colormap): Colormap to use.
            - vmin, vmax (float): Value range for display.
            - ax (matplotlib.axes.Axes): Axes to plot on.
    :return: EEORasterDataset
    """
    ...
