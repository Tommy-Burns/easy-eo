"""Provenance metadata (timestamp + attrs) preservation across operations.

An ``EEORasterDataset`` carries an optional acquisition ``timestamp`` and a
free-form ``attrs`` dict. Both must survive every chainable operation
(Milestone 3 design constraint), copied onto each operation's result.
"""

from datetime import datetime

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from eeo import load_array

TS = datetime(2023, 6, 1, 10, 30)
UTM = CRS.from_epsg(32633)
GRID = Affine.translation(500_000.0, 4_200_000.0) * Affine.scale(10.0, -10.0)


def test_defaults_are_empty():
    ds = load_array(np.ones((4, 4), dtype=np.float32), crs=UTM)
    assert ds.timestamp is None
    assert ds.attrs == {}


def test_loaders_accept_provenance():
    ds = load_array(
        np.ones((4, 4), dtype=np.float32),
        transform=GRID,
        crs=UTM,
        timestamp=TS,
        attrs={"sensor": "Sentinel-2"},
    )
    assert ds.timestamp == TS
    assert ds.attrs == {"sensor": "Sentinel-2"}


def test_loader_copies_attrs_dict():
    source = {"sensor": "Sentinel-2"}
    ds = load_array(np.ones((4, 4), dtype=np.float32), crs=UTM, attrs=source)
    ds.attrs["extra"] = 1
    assert "extra" not in source  # the loader copied the dict


def test_provenance_survives_a_representative_chain(single_band_float32):
    single_band_float32.timestamp = TS
    single_band_float32.attrs["sensor"] = "Sentinel-2"

    result = (
        single_band_float32.add(1)
        .multiply(2)
        .normalize_min_max()
        .resample(scale_factor=0.5)
        .clip_raster_with_bbox((500_010.0, 4_199_950.0, 500_050.0, 4_199_990.0))
        .reproject_raster(target_crs=4326)
    )

    assert result.timestamp == TS
    assert result.attrs == {"sensor": "Sentinel-2"}


def test_result_attrs_isolated_from_source(single_band_float32):
    single_band_float32.attrs["sensor"] = "Sentinel-2"

    result = single_band_float32.add(1)
    result.attrs["derived"] = True

    assert "derived" not in single_band_float32.attrs  # source untouched


def test_to_rasterio_preserves_provenance(numpy_backed_dataset):
    numpy_backed_dataset.timestamp = TS
    numpy_backed_dataset.attrs["sensor"] = "Sentinel-2"

    promoted = numpy_backed_dataset.to_rasterio()

    assert promoted.timestamp == TS
    assert promoted.attrs == {"sensor": "Sentinel-2"}


def test_provenance_survives_numpy_backed_chain(numpy_backed_dataset):
    numpy_backed_dataset.timestamp = TS
    numpy_backed_dataset.attrs["sensor"] = "Sentinel-2"

    # A NumPy-backed dataset must be promoted before a rasterio-only op.
    result = numpy_backed_dataset.to_rasterio().resample(scale_factor=0.5)

    assert result.timestamp == TS
    assert result.attrs == {"sensor": "Sentinel-2"}


def test_index_op_carries_provenance_from_first_operand(single_band_float32):
    single_band_float32.timestamp = TS
    single_band_float32.attrs["sensor"] = "Sentinel-2"

    nd = single_band_float32.normalized_difference(single_band_float32)

    assert nd.timestamp == TS
    assert nd.attrs == {"sensor": "Sentinel-2"}


@pytest.mark.parametrize("op", ["__add__", "__mul__", "__sub__"])
def test_operator_forms_carry_provenance(single_band_float32, op):
    single_band_float32.timestamp = TS
    single_band_float32.attrs["sensor"] = "Sentinel-2"

    result = getattr(single_band_float32, op)(1)

    assert result.timestamp == TS
    assert result.attrs == {"sensor": "Sentinel-2"}
