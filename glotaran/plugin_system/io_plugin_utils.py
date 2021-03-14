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


def protect_from_overwrite(path: str | os.PathLike[str], *, allow_overwrite: bool = False) -> None:
    """Raise FileExistsError if files already exists and allow_overwrite isn't True.

    Parameters
    ----------
    path : str
        Path to a file or folder.
    allow_overwrite : bool
        Whether or not to allow overwriting existing files, by default False

    Raises
    ------
    FileExistsError
        If path points to an existing file.
    FileExistsError
        If path points to an existing folder which is not empty.
    """
    user_info = (
        "To protect users overwriting existing files is deactivated by default. "
        "If you are absolutely sure this is what you want and need to do you can "
        "use the argument 'allow_overwrite=True'."
    )
    if allow_overwrite:
        pass
    elif os.path.isfile(path):
        raise FileExistsError(f"The file {path!r} already exists. \n{user_info}")
    elif os.path.isdir(path) and os.listdir(str(path)):
        raise FileExistsError(
            f"The folder {path!r} already exists and is not empty. \n{user_info}"
        )
