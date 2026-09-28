"""Grouping a series in time: monthly composites and the like.

A season of Sentinel-2 is forty acquisitions, and forty is rarely the number a
question is asked in. "How did this field green up?" is a question about months,
"was the flood gone a week later?" about weeks. Binning answers those by
reducing within each period instead of across the whole series, so a series of
forty scenes becomes a series of six monthly composites — still a series, so
everything that works on one still works.

The periods are spelled in pandas' own offset aliases, because pandas is already
a dependency and its vocabulary is one the target user knows from
:meth:`EEOTimeSeries.extract_at`. Nothing here invents a second spelling for
"monthly".
"""

from __future__ import annotations

import contextlib
import datetime as dt
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from eeo.core.core import EEORasterDataset
from eeo.core.exceptions import ValidationError
from eeo.core.types import StrPath

if TYPE_CHECKING:  # pragma: no cover - types only
    from eeo.timeseries.core import EEOTimeSeries

# Aliases pandas renamed in 2.2 ('M' to 'ME' and friends), mapped to what they
# were renamed to. Only used to explain a refusal: the alias a caller passes
# goes to pandas untouched, because translating it here would be a second
# spelling of "monthly" to keep in step with theirs.
_RENAMED = {"M": "ME", "Q": "QE", "Y": "YE", "A": "YE", "H": "h", "T": "min", "S": "s"}

_BAD_FREQ = (
    "{freq!r} is not a period pandas recognises ({detail}). Periods are pandas "
    "offset aliases: 'D' a day, '7D' seven days, 'W' a week, 'MS' a calendar "
    "month, 'QS' a quarter, 'YS' a year.{hint}"
)

_NEW_NAMES = {new: old for old, new in _RENAMED.items()}

_STABLE = (
    " The start-of-period aliases ('D', 'W', 'MS', 'QS', 'YS') were not renamed "
    "and mean the same thing on every pandas Easy-EO supports."
)


def _explain(freq: str, detail: str) -> str:
    """Compose the refusal for an unusable period, naming the pandas 2.2 renames.

    Which is the failure this is most likely to be: pandas 2.2 renamed the
    end-of-period aliases, and Easy-EO supports pandas both sides of that, so a
    period that works on one installation can be rejected on another.
    """
    hint = ""
    head = freq.lstrip("0123456789").split("-")[0] if isinstance(freq, str) else ""
    if head in _RENAMED:
        hint = f" pandas 2.2 renamed {head!r} to {_RENAMED[head]!r}.{_STABLE}"
    elif head in _NEW_NAMES:
        hint = (
            f" {head!r} needs pandas 2.2 or newer, and this installation is older; "
            f"there it is spelled {_NEW_NAMES[head]!r}.{_STABLE}"
        )
    return _BAD_FREQ.format(freq=freq, detail=detail, hint=hint)


def _grouped(series: EEOTimeSeries, freq: str) -> list[tuple[dt.datetime, list[int]]]:
    """Return the non-empty periods as ``(label, positions)``, oldest first.

    Empty periods are dropped rather than carried: a series cannot hold a
    timestep with no raster behind it, and a daily binning of a season would
    otherwise be mostly holes.
    """
    if not isinstance(freq, str) or not freq.strip():
        raise ValidationError(_explain(freq, "it is not a period string"))

    positions = pd.Series(
        range(len(series)), index=pd.DatetimeIndex(series.timestamps), dtype="int64"
    )
    # pandas' warnings are held back until the outcome is known. On 2.2-2.3 an
    # unknown alias warns before it is rejected — "'bogus' is deprecated, please
    # use 'BOGUS' instead" — which recommends something just as unknown and,
    # under an error filter, escapes as a FutureWarning instead of the refusal.
    # A deprecated alias that does resolve ('M' on 2.2+) still deserves pandas'
    # own guidance, so on success whatever pandas said is passed on.
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        try:
            resampled = positions.resample(freq)
            periods = [(label, list(values)) for label, values in resampled]
        except (ValueError, TypeError) as err:
            raise ValidationError(_explain(freq, str(err))) from err
    for warning in said:
        # Level 5 reaches the caller's line through EEOTimeSeries.resample_time,
        # resample_time and TemporalBins.__init__.
        warnings.warn(warning.message, stacklevel=5)

    return [(pd.Timestamp(label).to_pydatetime(), members) for label, members in periods if members]


class TemporalBins:
    """A series grouped into periods, waiting to be reduced within each one.

    What :meth:`EEOTimeSeries.resample_time` returns. It holds no pixels and
    reads nothing: the grouping is arithmetic on timestamps, and the work
    happens when a reducer is called on it.

    Each reducer returns a new :class:`~eeo.EEOTimeSeries` with one timestep per
    period — so the result is a series like any other, and can be reduced again,
    mapped over, sampled, or saved.

    Parameters
    ----------
    series : EEOTimeSeries
        Series being grouped. Its timesteps are shared, not copied.
    freq : str
        The pandas offset alias the grouping was asked for, kept for ``repr``
        and for the attrs of each result.

    Attributes
    ----------
    labels : list of datetime.datetime
        The period each bin covers, by its start (or end, for an alias like
        ``"ME"`` that labels periods by their end — the label is pandas' own).
    freq : str
        The alias this grouping was built with.

    Examples
    --------
    >>> import eeo
    >>> monthly = ts.resample_time("MS")  # doctest: +SKIP
    >>> len(monthly)  # doctest: +SKIP
    6
    >>> [len(bin) for bin in monthly]  # doctest: +SKIP
    [5, 7, 6, 8, 7, 5]
    """

    def __init__(self, series: EEOTimeSeries, freq: str) -> None:
        self._series = series
        self.freq = freq
        self._periods = _grouped(series, freq)

    @property
    def labels(self) -> list[dt.datetime]:
        """Return the period labels, oldest first.

        Returns
        -------
        list of datetime.datetime
            One label per non-empty period, as pandas labelled it.
        """
        return [label for label, _ in self._periods]

    def __len__(self) -> int:
        """Return the number of non-empty periods.

        Returns
        -------
        int
            Count of periods holding at least one timestep.
        """
        return len(self._periods)

    def __iter__(self) -> Any:
        """Iterate the periods as series, oldest first.

        Yields
        ------
        EEOTimeSeries
            One series per period, holding that period's timesteps.
        """
        for _, members in self._periods:
            yield self._sub(members)

    def __repr__(self) -> str:
        """Return a one-line summary: period count, alias, and time span."""
        # Never empty: a series holds at least one timestep, and that timestep
        # falls in some period.
        labels = self.labels
        sizes = ", ".join(str(len(members)) for _, members in self._periods[:6])
        if len(self._periods) > 6:
            sizes += ", ..."
        return (
            f"<TemporalBins: {len(self._periods)} periods of {self.freq!r} from "
            f"{labels[0].date()} to {labels[-1].date()}, {sizes} timesteps each>"
        )

    def _sub(self, members: list[int]) -> EEOTimeSeries:
        """Return the sub-series holding the timesteps at ``members``."""
        # Through the parent's own derivation, so the sub-series keeps its
        # backend choice and does not take ownership of the scene cache.
        return self._series._derive(
            [self._series[index] for index in members],
            [self._series.timestamps[index] for index in members],
        )

    def _reduce(self, how: str, save_dir: StrPath | None, kwargs: dict[str, Any]) -> EEOTimeSeries:
        """Reduce each period and collect the results into one series.

        Peak memory is one period's reduction where ``save_dir`` is given, and
        one per period otherwise — which is the whole result, and is why
        ``save_dir`` exists.
        """
        # Imported here and not at module scope: eeo/timeseries/core.py imports
        # this module for the method that returns a TemporalBins, so the pair is
        # circular at import time and not at call time.
        from eeo.timeseries.core import _through_file

        directory: Path | None = None
        if save_dir is not None:
            directory = Path(os.fspath(save_dir))
            directory.mkdir(parents=True, exist_ok=True)

        results: list[EEORasterDataset] = []
        labels: list[dt.datetime] = []
        try:
            for index, (label, members) in enumerate(self._periods):
                sub = self._sub(members)
                if how == "composite":
                    result = sub.composite(**kwargs)
                else:
                    result = getattr(sub, how)(**kwargs)
                # The reduction is this method's own object, and a timestep of a
                # series must be placed in time. A reducer states no timestamp
                # (a composite was not acquired at one moment), so the period's
                # label is what places it; the span it actually covers is in
                # attrs, recorded by the reducer.
                result.timestamp = label
                result.attrs["temporal_bin"] = self.freq
                if directory is not None:
                    result = _through_file(
                        result, directory, index, self._series._chunks, timestamp=label
                    )
                results.append(result)
                labels.append(label)
            return self._series._derive(results, labels)
        except BaseException:
            for result in results:
                with contextlib.suppress(Exception):
                    result.close()
            raise

    def median(self, *, save_dir: StrPath | None = None) -> EEOTimeSeries:
        """Take the median within each period.

        The one to reach for on imagery: a cloud missed by a mask is an outlier
        among the period's other acquisitions, which a median discards and a
        mean averages in.

        Parameters
        ----------
        save_dir : str or path-like or None, default None
            Write each period's result to a GeoTIFF in this directory and read
            the returned series from those files, so peak memory is one period's
            result rather than all of them. Created if absent; a rerun
            overwrites, as :meth:`eeo.EEORasterDataset.save_raster` does.

        Returns
        -------
        EEOTimeSeries
            One timestep per period, oldest first, each carrying its period's
            label as its timestamp, ``float32`` with NaN where no acquisition in
            the period saw the pixel.

        Examples
        --------
        >>> monthly = ts.resample_time("MS").median()  # doctest: +SKIP
        >>> len(monthly)  # doctest: +SKIP
        6
        """
        return self._reduce("median", save_dir, {})

    def mean(self, *, save_dir: StrPath | None = None) -> EEOTimeSeries:
        """Take the mean within each period.

        Parameters
        ----------
        save_dir : str or path-like or None, default None
            As :meth:`median`.

        Returns
        -------
        EEOTimeSeries
            One ``float32`` timestep per period, NaN where no acquisition in the
            period saw the pixel.
        """
        return self._reduce("mean", save_dir, {})

    def min(self, *, save_dir: StrPath | None = None) -> EEOTimeSeries:
        """Take the minimum within each period.

        Parameters
        ----------
        save_dir : str or path-like or None, default None
            As :meth:`median`.

        Returns
        -------
        EEOTimeSeries
            One timestep per period, keeping the timesteps' own dtype because a
            minimum selects a measured value rather than computing one.
        """
        return self._reduce("min", save_dir, {})

    def max(self, *, save_dir: StrPath | None = None) -> EEOTimeSeries:
        """Take the maximum within each period.

        The one that answers "how green did this ever get this month", which is
        what a per-period peak of an index series is.

        Parameters
        ----------
        save_dir : str or path-like or None, default None
            As :meth:`median`.

        Returns
        -------
        EEOTimeSeries
            One timestep per period, keeping the timesteps' own dtype.
        """
        return self._reduce("max", save_dir, {})

    def composite(self, *, save_dir: StrPath | None = None, **kwargs: Any) -> EEOTimeSeries:
        """Build one cloud-free raster per period.

        A monthly cloud-free composite, which is what most temporal analysis
        actually wants: each timestep is masked with its own quality band and the
        period's masked timesteps are reduced across time, so a month of partly
        clouded acquisitions becomes one clear image. The quality band is not in
        the result, as :meth:`eeo.EEOTimeSeries.composite` documents.

        Parameters
        ----------
        save_dir : str or path-like or None, default None
            As :meth:`median`. Worth using here: masking reads a whole scene,
            so a season of full scenes is the case this is for.
        **kwargs
            Passed to :meth:`eeo.EEOTimeSeries.composite` unchanged — ``how``,
            ``mask_band``, ``classes``, ``flags``, ``min_cloud_confidence``,
            ``mission``, ``nodata`` and ``mask_dir``.

        Returns
        -------
        EEOTimeSeries
            One cloud-free timestep per period, without the quality band. A
            pixel clouded at every acquisition in a period is nodata there —
            which a longer period is the fix for.

        Raises
        ------
        ValidationError
            For anything :meth:`eeo.EEOTimeSeries.composite` rejects, such as a
            series with no quality band. Raised on the first period, before the
            rest are read.

        Examples
        --------
        >>> monthly = ts.resample_time("MS").composite()  # doctest: +SKIP
        >>> greenest = monthly.map(eeo.ndvi, red="B04", nir="B08").max()  # doctest: +SKIP
        """
        if "save_path" in kwargs:
            raise ValidationError(
                "save_path= writes one raster, and a binned composite produces one "
                "per period; pass save_dir= instead"
            )
        return self._reduce("composite", save_dir, kwargs)


def resample_time(series: EEOTimeSeries, freq: str) -> TemporalBins:
    """Group a series into periods, to be reduced within each one.

    Parameters
    ----------
    series : EEOTimeSeries
        Series to group.
    freq : str
        Period length, as a pandas offset alias.

    Returns
    -------
    TemporalBins
        The grouping, with a reducer to call on it.

    Raises
    ------
    ValidationError
        If ``freq`` is not a period pandas recognises.
    """
    return TemporalBins(series, freq)
