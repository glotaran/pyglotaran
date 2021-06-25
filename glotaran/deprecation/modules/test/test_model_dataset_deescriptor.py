"""Test deprecated functionality in 'glotaran.model.base_model'."""
from __future__ import annotations

from glotaran.deprecation.modules.test import deprecation_warning_on_call_test_helper
from glotaran.model.dataset_descriptor import DatasetDescriptor


def test_dataset_descriptor_overwrite_index_dependent_method():
    """DatasetDescriptor.overwrite_index_dependent raises deperecation warning"""
    deprecation_warning_on_call_test_helper(DatasetDescriptor().overwrite_index_dependent)
