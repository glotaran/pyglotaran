"""Helper functions."""
from __future__ import annotations

from typing import Any
from typing import Callable
from typing import TypeVar
from typing import cast

DecoratedFunc = TypeVar("DecoratedFunc", bound=Callable[..., Any])  # decorated function


def wrap_func_as_method(
    cls: Any, name: str = None, annotations: dict[str, type] = None, doc: str = None
) -> DecoratedFunc:
    """A decorator to wrap a function as class method.

    Notes
    -----

    Only for internal use.

    Parameters
    ----------
    cls :
        The class in which the function will be wrapped.
    name :
        The name of method. If `None`, the original function's name is used.
    annotations :
        The annotations of the method. If `None`, the original function's annotations are used.
    doc :
        The documentation of the method. If `None`, the original function's documentation is used.
    """

    def decorator(func: DecoratedFunc) -> DecoratedFunc:
        if name:
            func.__name__ = name
        if annotations:
            setattr(func, "__annotations__", annotations)
        if doc:
            func.__doc__ = doc
        func.__qualname__ = cls.__qualname__ + "." + func.__name__
        func.__module__ = cls.__module__

        return func

    return cast(DecoratedFunc, decorator)
