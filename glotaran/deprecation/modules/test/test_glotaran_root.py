"""Test deprecated imports from 'glotaran/__init__.py' """
from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import pytest

from glotaran import read_model_from_yaml
from glotaran import read_model_from_yaml_file
from glotaran import read_parameters_from_csv_file
from glotaran import read_parameters_from_yaml
from glotaran import read_parameters_from_yaml_file
from glotaran.deprecation.deprecation_utils import GlotaranApiDeprecationWarning
from glotaran.deprecation.modules.test import deprecation_warning_on_call_test_helper
from glotaran.model import Model
from glotaran.parameter import ParameterGroup

if TYPE_CHECKING:
    from pathlib import Path


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


def test_read_model_from_yaml():
    """read_model_from_yaml raises warning"""
    yaml = """
    type: kinetic-spectrum
    megacomplex: {}
    """
    _, result = deprecation_warning_on_call_test_helper(
        read_model_from_yaml, args=[yaml], raise_exception=True
    )

    assert isinstance(result, Model)


def test_read_model_from_yaml_file(tmp_path: Path):
    """read_model_from_yaml_file raises warning"""
    yaml = """
    type: kinetic-spectrum
    megacomplex: {}
    """
    model_file = tmp_path / "model.yaml"
    model_file.write_text(yaml)
    _, result = deprecation_warning_on_call_test_helper(
        read_model_from_yaml_file, args=[str(model_file)], raise_exception=True
    )

    assert isinstance(result, Model)


def test_read_parameters_from_csv_file(tmp_path: Path):
    """read_parameters_from_csv_file raises warning"""
    parameters_file = tmp_path / "parameters.csv"
    parameters_file.write_text("label,value\nfoo,123")
    _, result = deprecation_warning_on_call_test_helper(
        read_parameters_from_csv_file,
        args=[str(parameters_file)],
        raise_exception=True,
    )

    assert isinstance(result, ParameterGroup)
    assert result.get("foo") == 123


def test_read_parameters_from_yaml():
    """read_parameters_from_yaml raises warning"""
    _, result = deprecation_warning_on_call_test_helper(
        read_parameters_from_yaml, args=["foo:\n  - 123"], raise_exception=True
    )

    assert isinstance(result, ParameterGroup)
    assert isinstance(result["foo"], ParameterGroup)
    assert result["foo"].get("1") == 123


def test_read_parameters_from_yaml_file(tmp_path: Path):
    """read_parameters_from_yaml_file raises warning"""
    parameters_file = tmp_path / "parameters.yaml"
    parameters_file.write_text("foo:\n  - 123")
    _, result = deprecation_warning_on_call_test_helper(
        read_parameters_from_yaml_file, args=[str(parameters_file)], raise_exception=True
    )

    assert isinstance(result, ParameterGroup)
    assert isinstance(result["foo"], ParameterGroup)
    assert result["foo"].get("1") == 123
