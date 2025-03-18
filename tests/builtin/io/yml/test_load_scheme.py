from __future__ import annotations

from glotaran.builtin.elements.kinetic.element import KineticElement
from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.io import load_scheme

test_scheme_yml = """
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
    scheme = load_scheme(test_scheme_yml, format_name="yml_str")
    assert isinstance(scheme.library["parallel"], KineticElement)
    assert isinstance(
        scheme.experiments["myexp"].datasets["kinetic_parallel"], ActivationDataModel
    )
