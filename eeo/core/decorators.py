"""
Decorators for easy-eo
"""

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
def eeo_raster_op(*, preserve_none: bool = ...) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def eeo_raster_op(func=None, *, preserve_none=False):
    """Attach a free function to EEORasterDataset as a chainable method.

    Usable bare (``@eeo_raster_op``) or with arguments
    (``@eeo_raster_op(preserve_none=True)``).

    By default a ``None`` result from the wrapped function is replaced with
    ``self`` so an operation that returns nothing still chains. Set
    ``preserve_none=True`` for operations whose ``None`` return is meaningful
    (e.g. ``mosaic`` returns ``None`` when it writes to ``save_path``); the
    bound method then returns that ``None`` unchanged instead of ``self``.
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
            if preserve_none:
                return result
            # allow functions to be chained
            return self if result is None else result

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
    """
    Decorator that binds a visualization function to EEORasterDataset
    as a terminal (non-chainable) method.

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
