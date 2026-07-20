from collections.abc import Iterable

from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op

@eeo_raster_op
def mosaic(
    ds: EEORasterDataset,
    others: EEORasterDataset | Iterable[EEORasterDataset],
    *,
    resampling_method: str = "nearest",
    save_path: str | None = None,
    auto_reproject: bool = False,
    **kwargs,
) -> EEORasterDataset | None: ...
@eeo_raster_op
def stack(
    ds: EEORasterDataset,
    others: EEORasterDataset | Iterable[EEORasterDataset],
) -> EEORasterDataset: ...
