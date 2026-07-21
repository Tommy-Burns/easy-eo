Core Module
===========

.. automodule:: eeo.core
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: EEOError, ValidationError, CRSMismatchError, AlignmentError, BackendError

Utilities
---------

.. autofunction:: eeo.show_versions

Exceptions
----------

Easy-EO raises a small hierarchy of exceptions so you can catch precisely what
you need. Every library-specific error derives from :class:`~eeo.EEOError`, so a
single ``except`` clause handles any of them while letting unrelated errors
propagate:

.. code-block:: python

    import eeo

    try:
        ds = eeo.load_raster("scene.tif")
        result = ds.normalized_difference(other, auto_align=False)
    except eeo.AlignmentError as err:
        # rasters were not on the same grid
        print(err)
    except eeo.EEOError as err:
        # any other Easy-EO failure
        print(err)

For backward compatibility, each subclass also derives from the built-in
exception it historically replaced: :class:`~eeo.ValidationError`,
:class:`~eeo.CRSMismatchError`, and :class:`~eeo.AlignmentError` are
``ValueError``\ s, and :class:`~eeo.BackendError` is a ``RuntimeError``. Two
failure modes intentionally keep their standard-library exceptions rather than
joining the hierarchy: a missing raster file raises ``FileNotFoundError``, and
an out-of-range band index raises ``IndexError``.

.. automodule:: eeo.core.exceptions
    :members:
    :show-inheritance:
