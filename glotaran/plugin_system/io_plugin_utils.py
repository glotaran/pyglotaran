"""Utility functions for io plugin."""

from __future__ import annotations

import functools
import os
from typing import Any
from typing import Callable
from typing import TypeVar
from typing import cast

DecoratedFunc = TypeVar("DecoratedFunc", bound=Callable[..., Any])  # decorated function


def inferr_file_format(file_path: str | os.PathLike[str]) -> str:
    """Inferr format of a file if it exists.

    Parameters
    ----------
    file_path : str
        Path/str to the file.

    Returns
    -------
    str
        File extension without the leading dot.

    Raises
    ------
    ValueError
        If file doesn't exists.
    ValueError
        If file has no extension.
    """
    if not os.path.isfile(file_path):
        raise ValueError(f"There is no file {file_path!r}.")

    _, file_format = os.path.splitext(file_path)
    if file_format == "":
        raise ValueError(
            f"Cannot determine format of file {file_path!r}, please provide an explicit format."
        )
    else:
        return file_format.lstrip(".")


def not_implemented_to_value_error(func: DecoratedFunc) -> DecoratedFunc:
    """Decorate a function to raise ValueError instead of NotImplementedError.

    This decorator is supposed to be used on functions which call functions
    that might raise a NotImplementedError, but raise ValueError instead with
    the same error text.

    Parameters
    ----------
    func : DecoratedFunc
        Function to be decorated.

    Returns
    -------
    DecoratedFunc
        Wrapped function.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except NotImplementedError as error:
            raise ValueError(error.args)

    return cast(DecoratedFunc, wrapper)
