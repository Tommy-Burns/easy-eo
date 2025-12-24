from typing import Union

from eeo.core.core import EEORasterDataset

Coordinate = Union[tuple[float, float], list[float]]

def extract_value_at_coordinate(
    ds: EEORasterDataset,
    coordinates: Coordinate,
    band_idx: int = 1,
) -> Union[int, float]:
    """
    Extract the pixel value from an EEORasterDataset at a given geographic coordinate.

    :param ds: The raster dataset from which to extract a value.
    :type ds: EEORasterDataset
    :param coordinates: A tuple of (x, y) coordinates. Must contain exactly two values.
    :type coordinates: Coordinate
    :param band_idx: The index of the raster band to sample from in case of a multiband raster.
    :type band_idx: int

    :return: The extracted pixel value from the specified band and coordinate.
    :rtype: int or float

    :raises ValueError: If the provided coordinates do not contain exactly two elements.
    """
    ...


def get_maximum_pixel(
    ds: EEORasterDataset,
    band_idx: int = 1,
    *,
    return_position_as_pixel_coordinate: bool = False,
) -> dict:
    """
    Get the maximum pixel value and its location from a raster band.

    :param ds: Input raster dataset.
    :param band_idx: Band index to evaluate (1-based).
    :param return_position_as_pixel_coordinate: If True, return (row, col) instead of spatial coordinates.

    :return: Dictionary containing the maximum value and its position.
    :rtype: dict[str, float | tuple]
    """
    ...

def get_minimum_pixel(
    ds: EEORasterDataset,
    band_idx: int = 1,
    *,
    return_position_as_pixel_coordinate: bool = False,
) -> dict:
    """
    Get the minimum pixel value and its location from a raster band.

    :param ds: Input raster dataset.
    :param band_idx: Band index to evaluate (1-based).
    :param return_position_as_pixel_coordinate: If True, return (row, col) instead of spatial coordinates.

    :return: Dictionary containing the minimum value and its position.
    :rtype: dict[str, float | tuple]
    """
    ...

def get_mean_pixel(
    ds: EEORasterDataset,
    band_idx: int = 1,
    *,
    return_position_as_pixel_coordinate: bool = False,
) -> dict:
    """
    Get the pixel whose value is closest to the mean of the raster band.

    :param ds: Input raster dataset.
    :param band_idx: Band index to evaluate (1-based).
    :param return_position_as_pixel_coordinate: If True, return (row, col) instead of spatial coordinates.

    :return: Dictionary containing the mean value and its position.
    """
    ...

def get_percentile_pixel(
    ds: EEORasterDataset,
    percentile: float,
    band_idx: int = 1,
    *,
    return_position_as_pixel_coordinate: bool = False,
) -> dict:
    """
    Get the pixel corresponding to a given percentile of raster values.

    :param ds: Input raster dataset.
    :param percentile: Percentile in the range [0, 100].
    :param band_idx: Band index to evaluate (1-based).
    :param return_position_as_pixel_coordinate: If True, return (row, col) instead of spatial coordinates.
    :return: Dictionary containing the percentile value and its position.
    """
    ...
