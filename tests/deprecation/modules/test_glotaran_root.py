"""Test deprecated imports from 'glotaran/__init__.py' """
from __future__ import annotations

from warnings import warn

import pytest

from glotaran.deprecation.deprecation_utils import GlotaranApiDeprecationWarning
from glotaran.deprecation.modules.test import deprecation_warning_on_call_test_helper


def dummy_warn(foo, bar=False):
    warn(GlotaranApiDeprecationWarning("foo"), stacklevel=2)
    if not isinstance(bar, bool):
        raise ValueError("not a bool")
    return foo, bar


def dummy_no_warn(foo, bar=False):
    return foo, bar


def test_deprecation_warning_on_call_test_helper():
    """Correct result passed on"""
    record, result = deprecation_warning_on_call_test_helper(
        dummy_warn, args=["foo"], kwargs={"bar": True}
    )
    assert len(record) == 1
    assert result == ("foo", True)


def test_deprecation_warning_on_call_test_helper_error_reraise():
    """Raise if raise_exception and args or kwargs"""

    with pytest.raises(ValueError, match="not a bool"):
        deprecation_warning_on_call_test_helper(
            dummy_warn, args=["foo"], kwargs={"bar": "baz"}, raise_exception=True
        )


@pytest.mark.xfail(strict=True, reason="Function did not warn.")
def test_deprecation_warning_on_call_test_helper_no_warn():
    """Fail no warning"""
    deprecation_warning_on_call_test_helper(dummy_no_warn, args=["foo"], kwargs={"bar": True})
