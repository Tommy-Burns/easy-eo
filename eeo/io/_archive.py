"""Where a product's files live, so the readers do not have to care.

A downloaded scene arrives as a directory, a Copernicus ``.zip``, or a USGS
``.tar``, and identifying it means the same four questions in every case: does
this file exist, which files match this pattern, what does this metadata file
say, and what do I hand rasterio to read this image? :class:`ProductSource`
answers those four and nothing else, so :mod:`eeo.io._sentinel2` and
:mod:`eeo.io._landsat` can be written once against a product rather than once
per container it might be shipped in.

Paths inside a source are relative :class:`~pathlib.PurePosixPath` values, and
purely nominal — an archive member has no filesystem path to be relative to.
Only :meth:`ProductSource.href` produces something openable, which keeps the
one place that has to know about ``/vsizip/`` down to a single line.

The read half of the loader never needed this: rasterio opens a URL as happily
as a path. It is discovery that is filesystem-bound, so it is discovery this
covers. A remote product is the same shape of problem as an archive — the
product is not a plain directory — and would arrive as one more implementation
here rather than as changes to the readers.

Notes
-----
**A compressed tar is refused rather than read.** The three containers here all
allow a single member to be addressed on its own: a directory by definition, a
zip through its central directory, an uncompressed tar through its member
headers. Wrapping a tar in gzip removes that. The compressed stream has no
index, so reaching any member means decompressing everything before it, and a
windowed read of one band decompresses the entire archive ahead of it — every
time, for every band. The cost is not a constant factor, so accepting it
silently would make the loader mysteriously slow rather than honestly refuse.

Zip is not free either, only bounded: a deflated member is decompressed from
its own start, not from the start of the archive. That is one image at worst,
and it is the format Copernicus actually ships, so it is supported with the
cost noted rather than refused.
"""

from __future__ import annotations

import functools
import re
import tarfile
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath

from eeo.core.exceptions import ValidationError
from eeo.core.types import StrPath

#: Final suffixes marking a compressed archive, which is refused. ``.tgz`` and
#: friends are the one-piece spellings of the same thing.
_COMPRESSED_SUFFIXES = (
    ".gz",
    ".bz2",
    ".xz",
    ".z",
    ".zst",
    ".tgz",
    ".tbz",
    ".tbz2",
    ".txz",
)

#: Archive suffixes, mapped to the GDAL virtual filesystem that reads them.
_ARCHIVE_VSI = {".zip": "vsizip", ".tar": "vsitar"}


@functools.lru_cache(maxsize=64)
def _anchored(pattern: str) -> re.Pattern[str]:
    """Compile a glob pattern the way :meth:`pathlib.Path.glob` reads it.

    Anchored at the root and with ``*`` stopping at a separator, so an archive
    answers a pattern exactly as the equivalent directory would. Matching
    archive members with :meth:`PurePosixPath.match` instead would match from
    the right, and quietly find things a directory source would not.
    """
    parts = []
    for char in pattern:
        if char == "*":
            parts.append("[^/]*")
        elif char == "?":
            parts.append("[^/]")
        else:
            parts.append(re.escape(char))
    return re.compile("".join(parts))


class ProductSource(ABC):
    """The files of one product, wherever they physically live.

    Attributes
    ----------
    name : str
        A name for the product, used where a message or a provenance record
        needs to identify it.
    location : Path
        What the caller actually passed, for error messages.
    named_entry : PurePosixPath or None
        The file the caller named, when they named one rather than a container.
        A reader honours it instead of searching, so pointing at a specific
        metadata file selects that file and not a sibling it prefers.
    """

    name: str
    location: Path
    named_entry: PurePosixPath | None

    @abstractmethod
    def exists(self, relative: PurePosixPath | str) -> bool:
        """Whether the source holds a file at this relative path."""

    @abstractmethod
    def glob(self, pattern: str) -> list[PurePosixPath]:
        """Relative paths of the files matching a glob pattern, sorted."""

    @abstractmethod
    def read_bytes(self, relative: PurePosixPath | str) -> bytes:
        """Read one file whole. For metadata, never for imagery."""

    @abstractmethod
    def href(self, relative: PurePosixPath | str) -> str:
        """Return what rasterio should open to read this file."""


class DirectorySource(ProductSource):
    """A product unpacked into a directory."""

    def __init__(self, root: Path, named_entry: PurePosixPath | None = None) -> None:
        self.root = root
        self.name = root.name
        self.named_entry = named_entry
        self.location = root if named_entry is None else root / named_entry

    def exists(self, relative: PurePosixPath | str) -> bool:
        return (self.root / str(relative)).is_file()

    def glob(self, pattern: str) -> list[PurePosixPath]:
        return sorted(
            PurePosixPath(match.relative_to(self.root).as_posix())
            for match in self.root.glob(pattern)
            if match.is_file()
        )

    def read_bytes(self, relative: PurePosixPath | str) -> bytes:
        try:
            return (self.root / str(relative)).read_bytes()
        except OSError as err:
            raise ValidationError(f"could not read {relative} under {self.root}: {err}") from err

    def href(self, relative: PurePosixPath | str) -> str:
        return str(self.root / str(relative))


class ArchiveSource(ProductSource):
    """A product still inside the zip or tar it was downloaded as.

    The member list is read once and kept; the archive itself is reopened for
    each metadata read, of which there are one or two. Imagery is never read
    through here — :meth:`href` hands GDAL a virtual filesystem path and GDAL
    addresses the member directly.
    """

    def __init__(self, archive: Path, suffix: str, entries: tuple[str, ...]) -> None:
        self.archive = archive
        self.suffix = suffix
        self.name = archive.name
        self.location = archive
        self.named_entry = None
        self._entries = entries
        self._set = frozenset(entries)

    def exists(self, relative: PurePosixPath | str) -> bool:
        return str(relative) in self._set

    def glob(self, pattern: str) -> list[PurePosixPath]:
        matches = _anchored(pattern)
        return sorted(PurePosixPath(e) for e in self._entries if matches.fullmatch(e))

    def read_bytes(self, relative: PurePosixPath | str) -> bytes:
        try:
            if self.suffix == ".zip":
                with zipfile.ZipFile(self.archive) as archive:
                    return archive.read(str(relative))
            with tarfile.open(self.archive) as tar:
                member = tar.extractfile(str(relative))
                if member is None:
                    raise ValidationError(f"{relative} is not a file in {self.archive.name}")
                return member.read()
        except (OSError, KeyError, zipfile.BadZipFile, tarfile.TarError) as err:
            raise ValidationError(
                f"could not read {relative} from {self.archive.name}: {err}"
            ) from err

    def href(self, relative: PurePosixPath | str) -> str:
        # GDAL addresses an archive member as /vsizip/<archive path>/<member>.
        # An absolute archive path leaves a doubled slash after the prefix,
        # which is the documented spelling rather than an accident.
        return f"/{_ARCHIVE_VSI[self.suffix]}/{self.archive.as_posix()}/{relative}"


def _archive_entries(archive: Path, suffix: str) -> tuple[str, ...]:
    """List an archive's file members, without extracting anything."""
    try:
        if suffix == ".zip":
            with zipfile.ZipFile(archive) as zipped:
                return tuple(info.filename for info in zipped.infolist() if not info.is_dir())
        with tarfile.open(archive) as tarred:
            return tuple(member.name for member in tarred.getmembers() if member.isfile())
    except (OSError, zipfile.BadZipFile, tarfile.TarError) as err:
        raise ValidationError(f"{archive.name} is not a readable archive: {err}") from err


def open_product(path: StrPath, mission: str) -> ProductSource:
    """Open whatever the caller pointed at as a source of product files.

    Parameters
    ----------
    path : str or pathlib.Path
        A product directory, a directory holding one, a ``.zip`` or ``.tar``
        holding one, or a single file inside a product directory.
    mission : str
        How to name the mission when the path turns out to hold nothing, e.g.
        ``"Sentinel-2"``.

    Returns
    -------
    ProductSource
        A directory-backed source, or an archive-backed one.

    Raises
    ------
    ValidationError
        If the path does not exist, if the archive cannot be read, or if it is
        a compressed archive.

    Notes
    -----
    A compressed archive is refused with instructions rather than read. See the
    module docstring for why the cost is not a constant factor.

    Examples
    --------
    >>> open_product("S2B_MSIL2A_20240830T100559.SAFE", "Sentinel-2")  # doctest: +SKIP
    <eeo.io._archive.DirectorySource object at ...>
    """
    candidate = Path(path)
    if not candidate.exists():
        raise ValidationError(f"no such {mission} product: {str(path)!r}")
    if candidate.is_dir():
        return DirectorySource(candidate)

    suffix = candidate.suffix.lower()
    if suffix in _COMPRESSED_SUFFIXES:
        stem = candidate.name[: -len(candidate.suffix)]
        raise ValidationError(
            f"{candidate.name!r} is a compressed archive, which cannot be read in "
            f"place: it holds no index, so reading one band would decompress every "
            f"byte before it, once per band. Extract it first — "
            f"`tar -xf {candidate.name}` — and pass the {stem!r} directory."
        )
    if suffix in _ARCHIVE_VSI:
        return ArchiveSource(candidate, suffix, _archive_entries(candidate, suffix))

    # A plain file: the caller named one file of a product, so read the
    # directory around it and remember which file they meant.
    return DirectorySource(candidate.parent, named_entry=PurePosixPath(candidate.name))
