"""Data access beyond the local filesystem.

Currently holds STAC catalog access (:mod:`eeo.io.stac`), which needs the
optional ``stac`` extra (``pip install "easy-eo[stac]"``). Importing this
package pulls in no optional dependency; the extra is only required when a
search actually runs.
"""

from .stac import PLANETARY_COMPUTER_STAC_URL, STACItem, STACSearchResult, stac_search

__all__ = [
    "stac_search",
    "STACItem",
    "STACSearchResult",
    "PLANETARY_COMPUTER_STAC_URL",
]
