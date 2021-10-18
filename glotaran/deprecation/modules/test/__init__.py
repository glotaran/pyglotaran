"""Package with deprecation tests and helper functions."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from glotaran.deprecation.deprecation_utils import GlotaranApiDeprecationWarning

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable
    from typing import Mapping
    from typing import Sequence

    from _pytest.recwarn import WarningsRecorder


def deprecation_warning_on_call_test_helper(
    deprecated_callable: Callable[..., Any],
    *,
    raise_exception=False,
    args: Sequence[Any] = [],
    kwargs: Mapping[str, Any] = {},
) -> tuple[WarningsRecorder, Any]:
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
    tuple[WarningsRecorder, Any]
        Tuple of the WarningsRecorder and return value of deprecated_callable

    Raises
    ------
    Exception
        Exception caused by deprecated_callable if raise_exception is True.
    """
    with pytest.warns(GlotaranApiDeprecationWarning) as record:
        try:
            result = deprecated_callable(*args, **kwargs)

            assert len(record) >= 1, f"{len(record)=}"
            assert Path(record[0].filename) == Path(
                __file__
            ), f"{Path(record[0].filename)=}, {Path(__file__)=}"

            return record, result

        except Exception as e:
            if raise_exception:
                raise e
            return record, None
