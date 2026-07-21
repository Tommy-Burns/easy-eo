"""Tests for EEORasterDataset.describe() and __repr__."""

from datetime import datetime

import pytest

from eeo.core.core import _resolve_stats_mode
from eeo.core.exceptions import ValidationError


def test_repr_is_concise(single_band_float32):
    assert repr(single_band_float32) == "<EEORasterDataset 1×6×6 float32 EPSG:32633>"


def test_resolve_stats_mode_mapping():
    assert _resolve_stats_mode(False) is None
    assert _resolve_stats_mode(True) == "approx"
    assert _resolve_stats_mode("approx") == "approx"
    assert _resolve_stats_mode("exact") == "exact"


def test_describe_invalid_stats_raises(single_band_float32):
    with pytest.raises(ValidationError, match="stats must be"):
        single_band_float32.describe(stats="full")


# ---------------------------------------------------------------------
# Structural description
# ---------------------------------------------------------------------


def test_describe_structural_content(raster_with_nodata, capsys):
    raster_with_nodata.describe()
    out = capsys.readouterr().out

    assert "EEORasterDataset" in out
    assert "EPSG:32633" in out
    assert "6 × 6  (height × width)" in out
    assert "dtype       : float32" in out
    assert "nodata      : -9999" in out
    assert "500000, 4199940, 500060, 4200000" in out  # extent, no sci notation
    assert "statistics" not in out  # no stats block by default


def test_describe_shows_provenance(single_band_float32, capsys):
    single_band_float32.timestamp = datetime(2023, 6, 1, 10, 30)
    single_band_float32.attrs["sensor"] = "Sentinel-2"

    single_band_float32.describe()
    out = capsys.readouterr().out

    assert "2023-06-01T10:30:00" in out
    assert "sensor=Sentinel-2" in out


def test_describe_structural_reads_no_pixels(single_band_float32, monkeypatch):
    adapter_cls = type(single_band_float32._adapter)
    calls: list[str] = []

    def spy(name, original):
        def wrapper(self, *args, **kwargs):
            calls.append(name)
            return original(self, *args, **kwargs)

        return wrapper

    monkeypatch.setattr(adapter_cls, "read", spy("read", adapter_cls.read))
    monkeypatch.setattr(adapter_cls, "read_band", spy("read_band", adapter_cls.read_band))

    single_band_float32.describe()

    assert calls == []


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------


def test_describe_exact_stats_exclude_nodata(raster_with_nodata, capsys):
    raster_with_nodata.describe(stats="exact")
    out = capsys.readouterr().out

    assert "statistics  : exact — full read" in out
    # valid values are 2..35 (0, 1, 6, 7 are nodata): min 2, mean 19.25
    assert "min 2 " in out
    assert "mean 19.25" in out
    assert "(4 nodata)" in out


def test_describe_multiband_reports_every_band(multiband_uint16, capsys):
    multiband_uint16.describe(stats="exact")
    out = capsys.readouterr().out

    for i in range(1, 5):
        assert f"band {i}" in out


def test_describe_approx_on_small_raster_reads_full_and_labels_exact(single_band_float32, capsys):
    # below the decimation cap -> full read, so honestly labelled exact
    single_band_float32.describe(stats="approx")
    out = capsys.readouterr().out

    assert "exact — full read" in out
    assert "~" not in out


def test_describe_approx_labels_and_marks_approximate(raster_with_nodata, monkeypatch, capsys):
    # force decimation of the 6x6 raster so the approximate path is exercised
    monkeypatch.setattr("eeo.core.core._STATS_DECIMATION_CAP", 3)

    raster_with_nodata.describe(stats="approx")
    out = capsys.readouterr().out

    assert "approximate — decimated read at 3 × 3 (set stats='exact' for exact)" in out
    assert "min~" in out
    assert "valid~" in out
    assert "nodata)" not in out  # absolute counts are not reported when approximate


def test_describe_stats_true_behaves_like_approx(single_band_float32, capsys):
    single_band_float32.describe(stats=True)
    out = capsys.readouterr().out

    assert "statistics" in out


def test_describe_numpy_backed_reads_full(numpy_backed_dataset, capsys):
    # a NumPy-backed dataset cannot do a decimated read, so approx is exact
    numpy_backed_dataset.describe(stats="approx")
    out = capsys.readouterr().out

    assert "exact — full read" in out
