"""Helper functions."""

import typing


def wrap_func_as_method(
    cls: typing.Any, name: str = None, annotations: str = None, doc: str = None
) -> typing.Callable:
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

    def decorator(func):
        if name:
            func.__name__ = name
        if annotations:
            setattr(func, "__annotations__", annotations)
        if doc:
            func.__doc__ = doc
        func.__qualname__ = cls.__qualname__ + "." + func.__name__
        func.__module__ = cls.__module__

        return func

    return decorator
