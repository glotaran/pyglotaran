from __future__ import annotations

from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.builtin.models.kinetic import KineticModel
from glotaran.io import load_scheme

test_scheme_yml = """
library:
    parallel:
        type: kinetic
        rates:
            (s1, s1): rates.1

experiments:
    - datasets:
        kinetic_parallel:
            models: [parallel]
            activation:
                - type: instant
                  compartments:
                      "s1": 1
"""


def test_load_scheme():
    scheme = load_scheme(test_scheme_yml, format_name="yml_str")
    assert isinstance(scheme.library["parallel"], KineticModel)
    assert isinstance(scheme.experiments[0].datasets["kinetic_parallel"], ActivationDataModel)
