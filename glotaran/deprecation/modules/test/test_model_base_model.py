"""Test deprecated functionality in 'glotaran.model.base_model'."""
from __future__ import annotations

import pytest

from glotaran.deprecation.modules.test import deprecation_warning_on_call_test_helper
from glotaran.model.base_model import Model
from glotaran.model.test.test_model import MockModel


def test_model_simulate_method():
    """Model.simulate raises deperecation warning"""
    deprecation_warning_on_call_test_helper(Model().simulate)


def test_model_index_dependent_method():
    """Model.index_dependent raises deperecation warning"""
    deprecation_warning_on_call_test_helper(MockModel().index_dependent)


def test_model_global_dimension_property():
    """Model.global_dimension raises deperecation warning"""
    with pytest.warns(DeprecationWarning):
        MockModel().global_dimension


def test_model_model_dimension_property():
    """Model.model_dimension raises deperecation warning"""
    with pytest.warns(DeprecationWarning):
        MockModel().model_dimension
