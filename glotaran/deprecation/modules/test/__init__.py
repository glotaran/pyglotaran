"""Package with deprecation tests and helper functions."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable
    from typing import Mapping
    from typing import Sequence


def deprecation_warning_on_call_test_helper(
    deprecated_callable: Callable[..., Any],
    *,
    raise_exception=False,
    args: Sequence[Any] = [],
    kwargs: Mapping[str, Any] = {},
) -> Any:
    """Helperfunction to quickly test that a deprecated class or function warns.

    By default this ignores error when calling the function/class,
    since those tests are only supposed to test the deprecation itself.

    However if the code is reimplemented it should be at least tested
    to return the proper type.


    Parameters
    ----------
    deprecated_callable : Callable[..., Any]
        Deprecated function or class.
    raise_exception : bool
        Whether or not to reraise an exception e.g. by calling with wrong args.
    args : Sequence[Any]
        Positional arguments for deprecated_callable.
    kwargs : Mapping[str, Any], optional
        Keyword arguments for deprecated_callable.

    Returns
    -------
    Any
        Return value of deprecated_callable

    Raises
    ------
    Exception
        Exception caused by deprecated_callable if raise_exception is True.
    """
    with pytest.warns(DeprecationWarning):
        try:
            return deprecated_callable(*args, **kwargs)
        except Exception:
            if raise_exception:
                raise
