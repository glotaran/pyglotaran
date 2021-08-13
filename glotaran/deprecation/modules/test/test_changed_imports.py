from __future__ import annotations

import warnings
from importlib import import_module
from typing import TYPE_CHECKING

import pytest

from glotaran.deprecation.deprecation_utils import GlotaranApiDeprecationWarning
from glotaran.deprecation.deprecation_utils import module_attribute
from glotaran.io import load_dataset
from glotaran.parameter import ParameterGroup
from glotaran.project import Result
from glotaran.project import Scheme
from glotaran.project import result as project_result
from glotaran.project import scheme as project_scheme

if TYPE_CHECKING:
    from _pytest.recwarn import WarningsRecorder


def check_recwarn(records: WarningsRecorder, warn_nr=1):

    for record in records:
        print(record)

    assert len(records) == warn_nr
    assert records[0].category == GlotaranApiDeprecationWarning

    records.clear()


def changed_import_test_warn(
    recwarn: WarningsRecorder, module_qual_name: str, *, attribute_name: str = None, warn_nr=1
):
    """Helper for testing changed imports, returning the imported item.

    Parameters
    ----------
    module_qual_name : str
        Fully qualified name for a module e.g. ``glotaran.model.base_model``
    attribute_name : str, optional
        Name of the attribute e.g. ``Model``

    Returns
    -------
    Any
        Module attribute or module
    """

    warnings.simplefilter("always")

    recwarn.clear()

    if attribute_name is not None:
        result = module_attribute(module_qual_name, attribute_name)
    else:
        result = import_module(module_qual_name)
    check_recwarn(recwarn, warn_nr=warn_nr)
    return result


@pytest.mark.xfail(strict=True, reason="Fail if no warning")
def test_changed_import_test_warn_attribute_no_warn(
    recwarn: WarningsRecorder,
):
    """Module attribute import not warning"""
    changed_import_test_warn(recwarn, "glotaran.parameter", attribute_name="ParameterGroup")


@pytest.mark.xfail(strict=True, reason="Fail if no warning")
def test_changed_import_test_warn_module_no_warn(
    recwarn: WarningsRecorder,
):
    """Module import not warning"""
    changed_import_test_warn(recwarn, "glotaran.parameter")


def test_root_ParameterGroup(recwarn: WarningsRecorder):
    """glotaran.ParameterGroup"""
    result = changed_import_test_warn(recwarn, "glotaran", attribute_name="ParameterGroup")

    assert result == ParameterGroup


def test_analysis_result(recwarn: WarningsRecorder):
    """Usage of glotaran.analysis.result"""
    warnings.simplefilter("always")

    from glotaran.analysis import result

    assert len(recwarn) == 0
    assert result.Result == project_result.Result  # type: ignore [attr-defined]

    check_recwarn(recwarn)


def test_analysis_result_from_import(recwarn: WarningsRecorder):
    """Same as 'from glotaran.analysis.result import Result' as analysis_result"""
    analysis_result = changed_import_test_warn(
        recwarn, "glotaran.analysis.result", attribute_name="Result"
    )

    assert analysis_result == Result


def test_analysis_scheme(recwarn: WarningsRecorder):
    """Usage of glotaran.analysis.scheme"""
    warnings.simplefilter("always")

    from glotaran.analysis import scheme as analysis_scheme

    assert len(recwarn) == 0
    assert analysis_scheme.Scheme == project_scheme.Scheme  # type: ignore [attr-defined]

    check_recwarn(recwarn)


def test_analysis_scheme_from_import(recwarn: WarningsRecorder):
    """Same as 'from glotaran.analysis.scheme import Scheme as analysis_scheme'"""
    analysis_scheme = changed_import_test_warn(
        recwarn, "glotaran.analysis.scheme", attribute_name="Scheme"
    )

    assert analysis_scheme == Scheme


def test_io_read_data_file(recwarn: WarningsRecorder):
    """glotaran.io.read_data_file"""
    result = changed_import_test_warn(recwarn, "glotaran.io", attribute_name="read_data_file")

    assert result.__code__ == load_dataset.__code__
