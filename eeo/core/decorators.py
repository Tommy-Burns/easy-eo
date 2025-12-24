"""
Decorators for easy-eo
"""

from __future__ import annotations
from functools import wraps
from typing import Callable, TypeVar, ParamSpec

from eeo.core.core import EEORasterDataset

# Generis types for type safety
P = ParamSpec("P") # parameters of the operation
R = TypeVar("R") # return type of the operation (Usually EEORasterDataset)


def eeo_raster_op(func: Callable[P, R]) -> Callable[P, R]:
    """
        Decorator that attaches a free function to EEORasterDataset
        as a chainable method with full type safety.
    """

    from .core import EEORasterDataset

    @wraps(func)
    def method(self: EEORasterDataset, *args: P.args, **kwargs: P.kwargs) -> R:
        result = func(self, *args, **kwargs)
        # allow functions to be chained
        return self if result is None else result

    # Bind to EEORasterDataset
    setattr(EEORasterDataset, func.__name__, method)

    return func # the original function is not altered


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
    return func
