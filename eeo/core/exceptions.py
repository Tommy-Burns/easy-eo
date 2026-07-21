"""Custom exception hierarchy for Easy-EO.

Every error Easy-EO raises for its own failure modes derives from
:class:`EEOError`, so callers can catch any library-specific problem with a
single ``except EEOError`` while letting unrelated exceptions propagate. Each
subclass additionally derives from the built-in exception it historically
replaced (``ValueError`` or ``RuntimeError``), so existing
``except ValueError`` / ``except RuntimeError`` handlers keep working.

Two failure modes intentionally keep their standard-library exceptions rather
than joining this hierarchy, because remapping them would break universal
Python idioms: a missing raster file raises :class:`FileNotFoundError`, and an
out-of-range band index raises :class:`IndexError`.
"""


class EEOError(Exception):
    """Base class for every error raised by Easy-EO."""


class ValidationError(EEOError, ValueError):
    """Raised when a function receives invalid or malformed input.

    Covers out-of-range parameters, wrong argument types, and malformed
    arrays or coordinates. Subclasses :class:`ValueError` for backward
    compatibility.
    """


class CRSMismatchError(EEOError, ValueError):
    """Raised when rasters have incompatible coordinate reference systems.

    Subclasses :class:`ValueError` for backward compatibility.
    """


class AlignmentError(EEOError, ValueError):
    """Raised when rasters are not aligned on the same pixel grid.

    Signals a mismatch in shape and/or affine transform between rasters that an
    operation requires to share a grid. Subclasses :class:`ValueError` for
    backward compatibility.
    """


class BackendError(EEOError, RuntimeError):
    """Raised for backend problems: an unsupported op or a failed read/open.

    Raised when an operation is not supported by the dataset's active backend
    (for example, a rasterio-only op called on a NumPy-backed dataset), or when
    a backend fails to open, read, or transform data. Subclasses
    :class:`RuntimeError` for backward compatibility.
    """
