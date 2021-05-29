"""Test deprecated functionality in 'glotaran.model.base_model'."""
from __future__ import annotations

from glotaran.deprecation.modules.test import deprecation_warning_on_call_test_helper
from glotaran.model.base_model import Model


def test_model_simulate_method():
    """Model.simulate raises deperecation warning"""
    deprecation_warning_on_call_test_helper(Model().simulate)
