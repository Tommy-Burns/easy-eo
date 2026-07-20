import rasterio as rio
from rasterio.enums import Resampling
from rasterio.transform import Affine

from eeo.common import normalize_resampling_method
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op
from eeo.core.types import ResamplingMethod


@eeo_raster_op
def resample(
    ds: EEORasterDataset,
    *,
    size: tuple[int, int] | None = None,
    scale_factor: float | None = None,
    resolution: tuple[float, float] | None = None,
    resampling_method: Resampling | ResamplingMethod = "nearest",
    plot_kwargs=None,
    show_preview: bool = False,
) -> EEORasterDataset:
    """Resample a raster to a new size, scale, or resolution.

    Exactly one of ``size``, ``scale_factor``, or ``resolution`` must be
    given. NumPy-backed datasets are promoted to the rasterio backend
    automatically, since resampling requires rasterio's decimated reads.

    Parameters
    ----------
    ds : EEORasterDataset
        Input raster dataset.
    size : tuple[int, int] or None, default None
        Output shape as ``(height, width)`` in pixels.
    scale_factor : float or None, default None
        Uniform scale factor applied to both dimensions (e.g. 0.5 halves
        the resolution, 2.0 doubles it).
    resolution : tuple[float, float] or None, default None
        Target pixel resolution as ``(xres, yres)`` in the raster's CRS
        units.
    resampling_method : str or rasterio.enums.Resampling, default "nearest"
        One of ``"nearest"``, ``"bilinear"``, ``"cubic"``,
        ``"cubic_spline"``, ``"lanczos"``, ``"average"``, ``"mode"``,
        ``"max"``, ``"min"``, ``"med"``, ``"q1"``, ``"q3"``, or a
        ``rasterio.enums.Resampling`` value. Defaults to ``"nearest"`` so
        categorical values and nodata edges are not blended; pick
        ``"bilinear"`` or a higher-order method for continuous data.
    plot_kwargs : dict or None, default None
        Keyword arguments forwarded to ``plot_raster`` when
        ``show_preview=True``.
    show_preview : bool, default False
        If True, plot the resampled raster after resampling.

    Returns
    -------
    EEORasterDataset
        New rasterio-backed dataset at the requested size/scale/resolution,
        in the same dtype as the input. The nodata value is carried over
        unchanged in the output metadata.

    Raises
    ------
    ValueError
        If zero or more than one of ``size``, ``scale_factor``, and
        ``resolution`` is given.
    RuntimeError
        If resampling fails for any other reason (wraps the underlying
        error).

    Notes
    -----
    Resamples via rasterio's decimated read (``out_shape``); the full
    output array is read into memory in a single call, not block-wise.

    Examples
    --------
    >>> from affine import Affine
    >>> ds = load_array(
    ...     np.zeros((100, 100), dtype=np.float32),
    ...     crs=4326,
    ...     transform=Affine.identity(),
    ... )
    >>> resampled = ds.resample(scale_factor=0.5)
    >>> resampled.get_shape()
    (50, 50)
    """
    params = [size, scale_factor, resolution]
    if sum(p is not None for p in params) != 1:
        raise ValueError("Provide exactly one of: size=, scale_factor=, resolution=")

    # Resampling needs rasterio's decimated reads; promote non-rasterio
    # backends (no-op if the backend is already rasterio)
    ds = ds.to_rasterio()

    # Compute new dimensions
    # --- When size is provided ---
    if size is not None:
        new_height, new_width = size

    # --- When scale factor is provided ---
    elif scale_factor is not None:
        new_width = int(ds.get_width() * scale_factor)
        new_height = int(ds.get_height() * scale_factor)

    # --- When resolution is provided ---
    else:
        assert resolution is not None  # guaranteed by the exactly-one check above
        xres, yres = resolution
        bounds = ds.get_bounds()

        new_width = int((bounds.right - bounds.left) / abs(xres))
        new_height = int((bounds.top - bounds.bottom) / abs(yres))
    try:
        # Resampling using bilinear interpolation
        resampling_enum = normalize_resampling_method(resampling_method)
        data = ds.read(
            out_shape=(ds.get_count(), new_height, new_width),
            resampling=resampling_enum,
        )

        # Computing scale transform
        scale_x = ds.get_width() / new_width
        scale_y = ds.get_height() / new_height

        transform = ds.get_transform() * Affine.scale(scale_x, scale_y)

        # Save or return EEORasterDataset
        # Update metadata
        meta = ds.get_metadata()
        meta.update(
            transform=transform,
            height=new_height,
            width=new_width,
        )
        # Write to MemoryFile
        memfile = rio.io.MemoryFile()
        with memfile.open(**meta) as mem:
            mem.write(data)

        dataset = memfile.open()

        if show_preview:
            EEORasterDataset.from_rasterio(dataset).plot_raster(**(plot_kwargs or {}))

        return EEORasterDataset.from_rasterio(dataset)
    except Exception as e:
        raise RuntimeError("Could not scale raster data") from e
