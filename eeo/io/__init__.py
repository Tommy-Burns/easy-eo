"""Data access and exchange beyond the local filesystem.

Holds STAC catalog access (:mod:`eeo.io.stac`, the ``stac`` extra) and xarray
interop (:mod:`eeo.io.xarray`, the ``xarray`` extra). Importing this package
pulls in no optional dependency; an extra is only required when one of its
features actually runs.
"""

from .stac import PLANETARY_COMPUTER_STAC_URL, STACItem, STACSearchResult, stac_search
from .xarray import from_xarray

__all__ = [
    "stac_search",
    "STACItem",
    "STACSearchResult",
    "PLANETARY_COMPUTER_STAC_URL",
    "from_xarray",
]
