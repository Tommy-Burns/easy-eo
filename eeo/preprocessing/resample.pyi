from rasterio.enums import Resampling

from eeo.core.core import EEORasterDataset
from eeo.core.types import ResamplingMethod

def resample(
    ds: EEORasterDataset,
    *,
    size: tuple[int, int] | None = None,
    scale_factor: float | None = None,
    resolution: tuple[float, float] | None = None,
    resampling_method: Resampling | ResamplingMethod = "nearest",
    plot_kwargs=None,
    show_preview: bool = False,
) -> EEORasterDataset: ...
