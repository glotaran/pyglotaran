from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.builtin.elements.kinetic.element import KineticElement
from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.io import load_scheme
from glotaran.io import save_scheme

if TYPE_CHECKING:
    from pathlib import Path

test_scheme_yml = """
# Just a comment
library:
    parallel:
        type: kinetic
        rates:
            (s1, s1): rates.1

experiments:
    myexp:
        datasets:
            kinetic_parallel:
                elements: [parallel]
                activations:
                    irf:
                        type: instant
                        compartments:
                            "s1": 1
"""


def test_load_scheme():
    """Load scheme from string."""
    scheme = load_scheme(test_scheme_yml, format_name="yml_str")
    assert isinstance(scheme.library["parallel"], KineticElement)
    assert isinstance(
        scheme.experiments["myexp"].datasets["kinetic_parallel"], ActivationDataModel
    )


def test_save_scheme_from_string(tmp_path: Path):
    """Save and load scheme from file."""
    input_scheme = load_scheme(test_scheme_yml, format_name="yml_str")
    save_path = tmp_path / "test_scheme.yml"
    save_scheme(input_scheme, save_path)
    loaded_scheme = load_scheme(save_path)
    assert loaded_scheme.model_dump() == input_scheme.model_dump()


def test_save_scheme_from_file(tmp_path: Path):
    """YAML file roundtrip preserves comments."""
    input_path = tmp_path / "input_scheme.yml"
    input_path.write_text(test_scheme_yml)

    save_path = tmp_path / "test_scheme.yml"
    save_scheme(load_scheme(input_path), save_path)

    assert save_path.read_text() == test_scheme_yml
