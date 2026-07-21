"""Failure-mode tests asserting the custom exception hierarchy is raised.

Each operation's main failure modes must raise a specific Easy-EO exception
(and, for backward compatibility, the built-in it derives from). The class
relationships themselves are covered in ``test_exceptions.py``; here we drive
real operations to the point of failure.
"""

import numpy as np
import pytest

from eeo.core.adapters.rasterio import RasterioAdapter
from eeo.core.exceptions import (
    AlignmentError,
    BackendError,
    CRSMismatchError,
    EEOError,
    ValidationError,
)

# ---------------------------------------------------------------------
# BackendError: rasterio-only ops called on a NumPy-backed dataset
# ---------------------------------------------------------------------


def test_mosaic_on_numpy_backend_raises_backend_error(numpy_backed_dataset):
    with pytest.raises(BackendError, match="rasterio-backed"):
        numpy_backed_dataset.mosaic(numpy_backed_dataset)


def test_stack_on_numpy_backend_raises_backend_error(numpy_backed_dataset):
    with pytest.raises(BackendError, match="rasterio-backed"):
        numpy_backed_dataset.stack(numpy_backed_dataset)


def test_clip_bbox_on_numpy_backend_raises_backend_error(numpy_backed_dataset):
    with pytest.raises(BackendError, match="rasterio-backed"):
        numpy_backed_dataset.clip_raster_with_bbox((0, 0, 10, 10))


def test_clip_vector_on_numpy_backend_raises_backend_error(numpy_backed_dataset):
    # The backend guard fires before the vector file is ever read.
    with pytest.raises(BackendError, match="rasterio-backed"):
        numpy_backed_dataset.clip_raster_with_vector("does_not_matter.geojson")


def test_reproject_on_numpy_backend_raises_backend_error(numpy_backed_dataset):
    with pytest.raises(BackendError, match="rasterio-backed"):
        numpy_backed_dataset.reproject_raster(target_crs=4326)


def test_backend_error_is_runtime_error(numpy_backed_dataset):
    # Backward compatibility: still catchable as RuntimeError.
    with pytest.raises(RuntimeError):
        numpy_backed_dataset.reproject_raster(target_crs=4326)


def test_adapter_open_failure_raises_backend_error(tmp_path):
    bad = tmp_path / "not_a_raster.txt"
    bad.write_text("nope")
    with pytest.raises(BackendError, match="failed to open raster"):
        RasterioAdapter.from_path(str(bad))


# ---------------------------------------------------------------------
# AlignmentError: mismatched grids where alignment is required
# ---------------------------------------------------------------------


def test_add_misaligned_without_auto_align_raises(shape_mismatch_pair):
    fine, coarse = shape_mismatch_pair
    with pytest.raises(AlignmentError, match="share the same grid"):
        fine.add(coarse, auto_align=False)


def test_normalized_difference_misaligned_without_auto_align_raises(shape_mismatch_pair):
    fine, coarse = shape_mismatch_pair
    with pytest.raises(AlignmentError, match="share the same grid"):
        fine.normalized_difference(coarse, auto_align=False)


def test_stack_shape_mismatch_raises_alignment_error(shape_mismatch_pair):
    # Same CRS, different transform/shape -> AlignmentError (not CRSMismatch).
    fine, coarse = shape_mismatch_pair
    with pytest.raises(AlignmentError, match="transform and shape"):
        fine.stack(coarse)


def test_alignment_error_is_value_error(shape_mismatch_pair):
    fine, coarse = shape_mismatch_pair
    with pytest.raises(ValueError):
        fine.add(coarse, auto_align=False)


# ---------------------------------------------------------------------
# CRSMismatchError: incompatible CRS
# ---------------------------------------------------------------------


def test_stack_crs_mismatch_raises_crs_error(crs_mismatch_pair):
    utm, geo = crs_mismatch_pair
    with pytest.raises(CRSMismatchError, match="share the CRS"):
        utm.stack(geo)


def test_mosaic_crs_mismatch_raises_crs_error(crs_mismatch_pair):
    utm, geo = crs_mismatch_pair
    with pytest.raises(CRSMismatchError, match="share the CRS"):
        utm.mosaic(geo)


# ---------------------------------------------------------------------
# ValidationError: malformed parameters
# ---------------------------------------------------------------------


def test_clip_bbox_wrong_length_raises_validation_error(single_band_float32):
    with pytest.raises(ValidationError, match="4 values"):
        single_band_float32.clip_raster_with_bbox((0, 0, 10))


def test_clip_bbox_non_intersecting_raises_validation_error(single_band_float32):
    # The UTM fixture sits near (500000, 4200000); a bbox at the origin misses.
    with pytest.raises(ValidationError, match="does not intersect"):
        single_band_float32.clip_raster_with_bbox((0, 0, 10, 10))


def test_clip_vector_bad_type_raises_validation_error(single_band_float32):
    with pytest.raises(ValidationError, match="GeoDataFrame"):
        single_band_float32.clip_raster_with_vector(12345)


def test_reproject_bad_crs_type_raises_validation_error(single_band_float32):
    with pytest.raises(ValidationError, match="target_crs"):
        single_band_float32.reproject_raster(target_crs=[1, 2, 3])


def test_resample_invalid_method_raises_validation_error(single_band_float32):
    # A bad method must surface as ValidationError, not the BackendError the
    # rasterio read is wrapped in.
    with pytest.raises(ValidationError, match="invalid resampling method"):
        single_band_float32.resample(scale_factor=2.0, resampling_method="not_a_method")


def test_resample_no_selector_raises_validation_error(single_band_float32):
    with pytest.raises(ValidationError, match="exactly one"):
        single_band_float32.resample()


def test_empty_others_mosaic_raises_validation_error(single_band_float32):
    with pytest.raises(ValidationError, match="at least one raster"):
        single_band_float32.mosaic([])


def test_empty_others_stack_raises_validation_error(single_band_float32):
    with pytest.raises(ValidationError, match="at least one raster"):
        single_band_float32.stack([])


# ---------------------------------------------------------------------
# Intentionally-kept standard-library exceptions
# ---------------------------------------------------------------------


def test_band_out_of_range_rasterio_backend_raises_index_error(single_band_float32):
    with pytest.raises(IndexError, match="valid 1..1"):
        single_band_float32.get_band(5)


def test_band_out_of_range_numpy_backend_raises_index_error(numpy_backed_dataset):
    with pytest.raises(IndexError, match="valid 1..1"):
        numpy_backed_dataset.get_band(5)


# ---------------------------------------------------------------------
# Everything domain-specific is catchable as EEOError
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "action",
    [
        lambda ds: ds.reproject_raster(target_crs=4326),  # BackendError on numpy backend
        lambda ds: ds.resample(),  # ValidationError
    ],
)
def test_domain_failures_catchable_as_eeoerror(numpy_backed_dataset, action):
    with pytest.raises(EEOError):
        action(numpy_backed_dataset)


def test_load_array_validation_is_eeoerror():
    from eeo import load_array

    with pytest.raises(EEOError):
        load_array(np.zeros((2, 2, 2, 2)))
