"""Helper functions."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Mapping
from typing import Sequence
from typing import Union

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable
    from typing import TypeVar

    DecoratedFunc = TypeVar("DecoratedFunc", bound=Callable[..., Any])  # decorated function


class ModelError(Exception):
    """Raised when a model contains errors."""

    def __init__(self, error: str):
        super().__init__(f"ModelError: {error}")


def wrap_func_as_method(
    cls: Any, name: str = None, annotations: dict[str, type] = None, doc: str = None
) -> Callable[[DecoratedFunc], DecoratedFunc]:
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

    def wrapper(func: DecoratedFunc) -> DecoratedFunc:
        if name:
            func.__name__ = name
        if annotations:
            setattr(func, "__annotations__", annotations)
        if doc:
            func.__doc__ = doc
        func.__qualname__ = cls.__qualname__ + "." + func.__name__
        func.__module__ = cls.__module__

        return func

    return wrapper


def is_scalar_type(t: type) -> bool:
    """Check if the type is scalar.

    Scalar means the type is neither a sequence nor a mapping.

    Parameters
    ----------
    t : type
        The type to check.

    Returns
    -------
    bool
        Whether the type is scalar.
    """
    if hasattr(t, "__origin__"):
        # Union can for some reason not be used in issubclass
        return t.__origin__ is Union or not issubclass(t.__origin__, (Sequence, Mapping))
    return True


def is_sequence_type(t: type) -> bool:
    """Check if the type is a sequence.

    Parameters
    ----------
    t : type
        The type to check.

    Returns
    -------
    bool
        Whether the type is a sequence.
    """
    return not is_scalar_type(t) and issubclass(t.__origin__, Sequence)


def is_mapping_type(t: type) -> bool:
    """Check if the type is mapping.

    Parameters
    ----------
    t : type
        The type to check.

    Returns
    -------
    bool
        Whether the type is a mapping.
    """
    return not is_scalar_type(t) and issubclass(t.__origin__, Mapping)


def get_subtype(t: type) -> type:
    """Gets the subscribed type of a generic type.

    If the type is scalar, the type itself will be returned. If the type is a mapping,
    the value type will be returned.

    Parameters
    ----------
    t : type
        The origin type.

    Returns
    -------
    type
        The subscribed type.
    """
    if is_sequence_type(t):
        return t.__args__[0]
    elif is_mapping_type(t):
        return t.__args__[1]
    return t
