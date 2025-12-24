from typing import Union

from eeo.core.core import EEORasterDataset


# ==========================================================
# ARITHMETIC AND ALGEBRA
# ==========================================================

def add(
        ds: EEORasterDataset,
        other: Union[EEORasterDataset, float, int],
        *,
        auto_align: bool = True,
        method: str = "bilinear"
) -> EEORasterDataset:
    """
    Pixel-wise addition of two rasters or a raster and a scalar.

    If `other` is an EEORasterDataset and its shape or spatial alignment
    differs from `ds`, the raster may be automatically resampled to match
    `ds` before performing the operation.


    :param ds: EEORasterDataset
        The base raster dataset.
    :param other: EEORasterDataset | float | int
        Raster or scalar value to add.
    :param auto_align: bool, default=True
        Whether to automatically resample `other` to match `ds` if their
        shapes or transforms differ.
    :param method: str, default="bilinear"
        Resampling method used when auto-alignment is required.

    :return: EEORasterDataset
        A new raster dataset containing the result.
    """
    ...


def subtract(
        ds: EEORasterDataset,
        other: Union[EEORasterDataset, float, int],
        *,
        auto_align: bool = True,
        method: str = "bilinear"
) -> EEORasterDataset:
    """
    Pixel-wise subtraction of two rasters or a raster and a scalar.

    The result is computed as `ds - other`.

    Auto-alignment behavior is identical to `add()`.

    Returns
    -------
    EEORasterDataset
    """
    ...


def multiply(
        ds: EEORasterDataset,
        other: Union[EEORasterDataset, float, int],
        *,
        auto_align: bool = True,
        method: str = "bilinear"
) -> EEORasterDataset:
    """
    Pixel-wise multiplication of two rasters or a raster and a scalar.

    Auto-alignment behavior is identical to `add()`.

    Returns
    -------
    EEORasterDataset
    """
    ...


def divide(
        ds: EEORasterDataset,
        other: Union[EEORasterDataset, float, int],
        *,
        auto_align: bool = True,
        method: str = "bilinear",
        safe: bool = True
) -> EEORasterDataset:
    """
    Pixel-wise division of two rasters or a raster and a scalar.

    If `safe=True`, division by zero and invalid values are handled safely
    by suppressing warnings and replacing NaNs with zeros.

    Parameters
    ----------
    safe : bool, default=True
        Whether to enable safe division handling.

    Returns
    -------
    EEORasterDataset
    :param ds:
    """
    ...


def power(
        ds: EEORasterDataset,
        exponent: Union[int, float]
) -> EEORasterDataset:
    """
    Pixel-wise exponentiation of raster values.

    Each pixel value is raised to the given scalar exponent.

    Returns
    -------
    EEORasterDataset
    """
    ...


# ==========================================================
# TRANSFORMATIONS
# ==========================================================

def sqrt(
        ds: EEORasterDataset
) -> EEORasterDataset:
    """
    Pixel-wise square root of raster values.

    Negative values are clipped to zero prior to computation.

    Returns
    -------
    EEORasterDataset
    """
    ...


def log(
        ds: EEORasterDataset,
        base: Union[int, float] = ...
) -> EEORasterDataset:
    """
    Pixel-wise logarithm of raster values.

    Zero and negative values are handled safely by clamping to a small
    positive constant prior to computation.

    Parameters
    ----------
    base : int | float, default=e
        Logarithmic base.

    Returns
    -------
    EEORasterDataset
    """
    ...


def absolute(
        ds: EEORasterDataset
) -> EEORasterDataset:
    """
    Pixel-wise absolute value of raster values.

    Returns
    -------
    EEORasterDataset
    """
    ...
