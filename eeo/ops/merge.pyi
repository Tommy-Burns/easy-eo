from typing import Union, Iterable

from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op


@eeo_raster_op
def mosaic(
    ds: EEORasterDataset,
    others: list[EEORasterDataset],
    *,
    resampling_method: str = "nearest",
    save_path: str | None = None,
    auto_reproject: bool = False,
    **kwargs,
) -> EEORasterDataset | None:
    """
    Mosaic one or more rasters into a single raster.

    Parameters
    ----------
    ds : EEORasterDataset
        Base raster.
    others : EEORasterDataset or Iterable[EEORasterDataset]
        One or more rasters to mosaic with `ds`.
    resampling_method : str, optional
        Resampling method used during merge.
    save_path : str, optional
        If provided, saves the result to disk and returns None.
    auto_reproject : bool, optional
        Reproject rasters to match `ds` CRS if needed.
    """
    ...


@eeo_raster_op
def stack(
        ds: EEORasterDataset,
        others: Union[EEORasterDataset, Iterable[EEORasterDataset]],
) -> EEORasterDataset:
    """
        Stack rasters into a multi-band raster.

        All rasters must have identical CRS, transform, and shape.
    """
    ...
