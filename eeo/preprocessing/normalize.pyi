from eeo.core.core import EEORasterDataset

def standardize(ds: EEORasterDataset) -> EEORasterDataset:
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
    ds: EEORasterDataset, *, new_min: int | float = ..., new_max: int | float = ...
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
    lower_percentile: int | float = ...,
    upper_percentile: int | float = ...,
) -> EEORasterDataset: ...
