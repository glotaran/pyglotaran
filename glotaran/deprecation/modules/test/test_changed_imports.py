from __future__ import annotations

import warnings
from importlib import import_module
from typing import TYPE_CHECKING

import pytest

from glotaran.deprecation.deprecation_utils import GlotaranApiDeprecationWarning
from glotaran.deprecation.deprecation_utils import module_attribute
from glotaran.optimization import optimize as optimize_module
from glotaran.simulation import simulation as simulation_module

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
    changed_import_test_warn(recwarn, "glotaran.parameter", attribute_name="Parameters")


@pytest.mark.xfail(strict=True, reason="Fail if no warning")
def test_changed_import_test_warn_module_no_warn(
    recwarn: WarningsRecorder,
):
    """Module import not warning"""
    changed_import_test_warn(recwarn, "glotaran.parameter")


def test_analysis_optimization(recwarn: WarningsRecorder):
    """Usage of glotaran.analysis.optimization"""
    warnings.simplefilter("always")

    from glotaran.analysis import optimize as analysis_optimize

    assert len(recwarn) == 0
    assert analysis_optimize.optimize == optimize_module.optimize

    check_recwarn(recwarn)


def test_analysis_optimization_from_import(recwarn: WarningsRecorder):
    """Same as 'from glotaran.analysis.optimize import optimizations analysis_scheme'"""
    changed_import_test_warn(recwarn, "glotaran.analysis.optimize", attribute_name="optimize")


def test_analysis_simulation(recwarn: WarningsRecorder):
    """Usage of glotaran.analysis.simulation"""
    warnings.simplefilter("always")

    from glotaran.analysis import simulation as analysis_simulation

    assert len(recwarn) == 0
    assert analysis_simulation.simulate == simulation_module.simulate

    check_recwarn(recwarn)


def test_analysis_simulation_from_import(recwarn: WarningsRecorder):
    """Same as 'from glotaran.analysis.simulation import simulate as analysis_scheme'"""
    changed_import_test_warn(recwarn, "glotaran.analysis.simulation", attribute_name="simulate")


@pytest.mark.parametrize(
    "attribute_name", ("sim_model", "dataset", "model", "scheme", "wanted_parameter", "parameter")
)
def test_examples_sequential(recwarn: WarningsRecorder, attribute_name: str):
    """glotaran.examples.sequential exported attributes"""
    from glotaran.examples import sequential  # noqa: F401

    recwarn.clear()

    changed_import_test_warn(
        recwarn, "glotaran.examples.sequential", attribute_name=attribute_name
    )
