"""Utility functions for io plugin."""

from __future__ import annotations

import os
from functools import partial
from functools import wraps
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Iterator
from typing import TypeVar
from typing import cast

DecoratedFunc = TypeVar("DecoratedFunc", bound=Callable[..., Any])  # decorated function


def inferr_file_format(
    file_path: str | os.PathLike[str], *, needs_to_exist: bool = True, allow_folder=False
) -> str:
    """Inferr format of a file if it exists.

    Parameters
    ----------
    file_path : str
        Path/str to the file.
    needs_to_exist : bool
        Whether or not a file need to exists for an successful format inferring.
        While write functions don't need the file to exists, load functions do.
    allow_folder: bool
        Whether or not to allow the format to be ``folder``.
        This is only used in ``save_result``.

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
    if not os.path.isfile(file_path) and needs_to_exist and not allow_folder:
        raise ValueError(f"There is no file {file_path!r}.")

    _, file_format = os.path.splitext(file_path)
    if file_format == "":
        if allow_folder:
            return "folder"
        else:
            raise ValueError(
                f"Cannot determine format of file {file_path!r}, "
                "please provide an explicit format."
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

    @wraps(func)
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
        return
    elif os.path.isfile(path):
        raise FileExistsError(f"The file {path!r} already exists. \n{user_info}")
    elif os.path.isdir(path) and os.listdir(str(path)):
        raise FileExistsError(
            f"The folder {path!r} already exists and is not empty. \n{user_info}"
        )


def bool_str_repr(value: Any, true_repr: str = "*", false_repr: str = "/") -> Any:
    """Replace boolean value with string repr.

    This function is a helper for table representation (e.g. with tabulate)
    of boolean values.

    Parameters
    ----------
    value : Any
        Arbitrary value
    true_repr : str
        Desired repr for ``True``, by default "*"
    false_repr : str
        Desired repr for ``False``, by default "/"

    Returns
    -------
    Any
        Original value or desired repr for bool

    Examples
    --------
    >>> table_data = [["foo", True, False], ["bar", False, True]]
    >>> print(tabulate(map(lambda x: map(bool_table_repr, x), table_data)))
    ---  -  -
    foo  *  /
    bar  /  *
    ---  -  -
    """
    # since bool is a subclass of int we can't use isinstance
    if type(value) == bool:
        if value:
            return true_repr
        else:
            return false_repr
    else:
        return value


def bool_table_repr(
    table_data: Iterable[Iterable[Any]], true_repr: str = "*", false_repr: str = "/"
) -> Iterator[Iterator[Any]]:
    """Replace boolean value with string repr for all table values.

    This function is an implementation of :func:`bool_str_repr` for a
    2D table, for easy usage with tabulate.

    Parameters
    ----------
    table_data : Iterable[Iterable[Any]]
        Data of the table e.g. a list of lists.
    true_repr : str
        Desired repr for ``True``, by default "*"
    false_repr : str
        Desired repr for ``False``, by default "/"

    Returns
    -------
    Iterator[Iterator[Any]]
        ``table_data`` with original values or desired repr for bool

    See Also
    --------
    bool_str_repr

    Examples
    --------
    >>> table_data = [["foo", True, False], ["bar", False, True]]
    >>> print(tabulate(bool_table_repr(table_data))
    ---  -  -
    foo  *  /
    bar  /  *
    ---  -  -
    """
    bool_repr = partial(bool_str_repr, true_repr=true_repr, false_repr=false_repr)
    return map(lambda value: map(bool_repr, value), table_data)
