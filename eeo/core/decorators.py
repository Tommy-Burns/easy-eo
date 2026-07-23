"""Decorators for Easy-EO."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar, overload

from eeo.core.core import EEORasterDataset

# Generis types for type safety
P = ParamSpec("P")  # parameters of the operation
R = TypeVar("R")  # return type of the operation (Usually EEORasterDataset)

# Registry of every function bound onto EEORasterDataset via the decorators
# below. Each entry is ``(func, kind)`` where ``kind`` is ``"op"`` (chainable,
# @eeo_raster_op) or ``"viz"`` (terminal, @eeo_raster_viz). This is the single
# source of truth consumed by scripts/generate_core_stub.py to (re)generate
# eeo/core/core.pyi, which exposes these dynamically-bound methods to type
# checkers. It is not part of the public API.
_OP_REGISTRY: list[tuple[Callable[..., object], str]] = []


@overload
def eeo_raster_op(func: Callable[P, R]) -> Callable[P, R]: ...


@overload
def eeo_raster_op(
    *, preserve_none: bool = ..., propagate_band_names: bool = ...
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def eeo_raster_op(func=None, *, preserve_none=False, propagate_band_names=True):
    """Attach a free function to EEORasterDataset as a chainable method.

    Usable bare (``@eeo_raster_op``) or with arguments
    (``@eeo_raster_op(preserve_none=True)``).

    By default a ``None`` result from the wrapped function is replaced with
    ``self`` so an operation that returns nothing still chains. Set
    ``preserve_none=True`` for operations whose ``None`` return is meaningful
    (e.g. ``mosaic`` returns ``None`` when it writes to ``save_path``); the
    bound method then returns that ``None`` unchanged instead of ``self``.

    Band names are propagated only for identity-preserving operations — those
    whose result has the same band count as the input, where output band *i*
    still means what input band *i* meant (scalar algebra, clipping,
    resampling, reprojection, normalization). Operations that synthesize a new
    band (the spectral indices) or rearrange bands (``stack``, ``mosaic``) must
    set ``propagate_band_names=False`` and name their output themselves;
    inheriting names there would mislabel the data. The names of a result that
    already carries its own are never overwritten.
    """
    from .core import EEORasterDataset

    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        # ``func`` already takes the dataset as its first parameter, so the
        # bound method cannot reuse ``P`` (which includes that parameter)
        # without mypy rejecting the ``func(self, ...)`` call. Reference it as
        # a plain callable internally; the returned ``func`` keeps its precise
        # ``Callable[P, R]`` signature for callers.
        op: Callable[..., R] = func

        @wraps(func)
        def method(self: EEORasterDataset, *args: object, **kwargs: object) -> R | EEORasterDataset:
            result = op(self, *args, **kwargs)
            if result is None:
                # A meaningful None (preserve_none) passes through; otherwise
                # return self so a no-op still chains.
                return result if preserve_none else self
            # Carry provenance (timestamp + attrs) onto a freshly produced
            # dataset unless the operation set its own.
            if isinstance(result, EEORasterDataset) and result is not self:
                if result.timestamp is None:
                    result.timestamp = self.timestamp
                if not result.attrs:
                    result.attrs = dict(self.attrs)
                if (
                    propagate_band_names
                    and result.get_count() == self.get_count()
                    and not any(result.band_names)
                ):
                    result.band_names = self.band_names
            return result

        # Bind to EEORasterDataset
        setattr(EEORasterDataset, func.__name__, method)
        _OP_REGISTRY.append((func, "op"))

        return func  # the original function is not altered

    # Called bare: @eeo_raster_op
    if func is not None:
        return decorate(func)
    # Called with arguments: @eeo_raster_op(preserve_none=True)
    return decorate


def eeo_raster_viz(func: Callable[..., R]) -> Callable[..., R]:
    """Bind a visualization function to EEORasterDataset as a terminal method.

    Visualization methods:
        - operate on the dataset
        - return None or non-dataset objects
        - do not participate in chaining like the ``eeo_raster_op`` decorator
    """

    @wraps(func)
    def method(self: EEORasterDataset, *args, **kwargs) -> R:
        return func(self, *args, **kwargs)

    # Bind to EEORasterDataset
    setattr(EEORasterDataset, func.__name__, method)
    _OP_REGISTRY.append((func, "viz"))
    return func
