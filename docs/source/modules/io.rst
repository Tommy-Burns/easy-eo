IO Module
=========

Data access beyond the local filesystem. STAC search needs the optional
``stac`` extra (``pip install "easy-eo[stac]"``); without it, calling
:func:`eeo.stac_search` raises :class:`~eeo.MissingDependencyError` with the
install command.

STAC search
-----------

.. autofunction:: eeo.stac_search

.. autoclass:: eeo.io.STACSearchResult
    :members:

.. autoclass:: eeo.io.STACItem
    :members:

.. autodata:: eeo.io.PLANETARY_COMPUTER_STAC_URL
