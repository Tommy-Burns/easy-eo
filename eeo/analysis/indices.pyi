from typing import Union

import numpy as np

from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op


@eeo_raster_op
def normalized_difference(
        ds: EEORasterDataset,
        other: EEORasterDataset,
        *,
        auto_align: bool = True,
        method: str = "bilinear",
        return_as_ndarray: bool = False
) -> Union[np.ndarray, EEORasterDataset]:
    """
        Compute the normalized difference index: (ds - other) / (ds + other)


        :param ds: EEORasterDataset
            First raster (e.g. NIR).
        :param other: EEORasterDataset
            Second raster (e.g. Red).
        :param auto_align: bool, optional
            Automatically resample `other` to match `ds`.
        :param method: str, optional
            Resampling method used if auto-aligning.
        :param return_as_ndarray: bool, optional
            If True, return a NumPy array instead of an EEORasterDataset.

        :return: Union[np.ndarray, EEORasterDataset]
    """
    ...
