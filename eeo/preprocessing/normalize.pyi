from typing import Union

from eeo.core.core import EEORasterDataset


def standardize(
        ds: EEORasterDataset
) -> EEORasterDataset:
    """
    Z-score standardization of raster values.

    The transformation is defined as:
        (x - mean) / standard_deviation

    Returns
    -------
    EEORasterDataset
    """
    ...


def normalize_min_max(
        ds: EEORasterDataset,
        *,
        new_min: Union[int, float] = ...,
        new_max: Union[int, float] = ...
) -> EEORasterDataset:
    """
    Min-max normalization of raster values.

    Values are linearly rescaled from the original data range
    to the interval [new_min, new_max].

    Returns
    -------
    EEORasterDataset
    """
    ...


def normalize_percentile(
        ds: EEORasterDataset,
        *,
        lower_percentile: Union[int, float] = ...,
        upper_percentile: Union[int, float] = ...
) -> EEORasterDataset:
    """
    Percentile-based normalization of raster values.

    The lower and upper percentiles are computed using NaN-aware
    percentile estimation, and values are clipped to [0, 1].

    This method is robust to outliers and commonly used for
    visualization and normalization of skewed data.

    Parameters
    ----------
    lower_percentile : int | float, default=0.0
        Lower percentile threshold (0–100).
    upper_percentile : int | float, default=1.0
        Upper percentile threshold (0–100).

    Returns
    -------
    EEORasterDataset
    """
    ...
