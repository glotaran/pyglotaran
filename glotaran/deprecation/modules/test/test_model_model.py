"""Tests for deprecated methods in ``glotaran.model.model``."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from glotaran.deprecation.deprecation_utils import GlotaranDeprectedApiError
from glotaran.testing.model_generators import SimpleModelGenerator

if TYPE_CHECKING:
    from glotaran.model import Model


@pytest.fixture(scope="module")
def dummy_model() -> Model:
    """Minimal model instance for testing."""
    generator = SimpleModelGenerator(
        rates=[300e-3],
        k_matrix="parallel",
    )
    return generator.model


def test_model_model_dimension(dummy_model: Model):
    """Raise ``GlotaranApiDeprecationWarning``."""
    expected = (
        "Usage of 'Model.model_dimension' was deprecated, "
        "use \"Scheme.model_dimensions['<dataset_name>']\" instead.\n"
        "It wasn't possible to restore the original behavior of this usage "
        "(mostlikely due to an object hierarchy change)."
        "This usage change message won't be show as of version: '0.7.0'."
    )

    with pytest.raises(GlotaranDeprectedApiError) as excinfo:
        dummy_model.model_dimension

    assert str(excinfo.value) == expected


def test_model_global_dimension(dummy_model: Model):
    """Raise ``GlotaranApiDeprecationWarning``."""
    expected = (
        "Usage of 'Model.global_dimension' was deprecated, "
        "use \"Scheme.global_dimensions['<dataset_name>']\" instead.\n"
        "It wasn't possible to restore the original behavior of this usage "
        "(mostlikely due to an object hierarchy change)."
        "This usage change message won't be show as of version: '0.7.0'."
    )

    with pytest.raises(GlotaranDeprectedApiError) as excinfo:
        dummy_model.global_dimension

    assert str(excinfo.value) == expected
