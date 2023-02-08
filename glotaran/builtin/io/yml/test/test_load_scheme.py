from __future__ import annotations

from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.builtin.megacomplexes.kinetic import KineticMegacomplex
from glotaran.io import load_scheme

test_scheme_yml = """
library:
    megacomplex:
        parallel:
            type: kinetic
            kinetic: [parallel]
    kinetic:
        parallel:
            rates:
                (s1, s1): rates.1

experiments:
    - datasets:
        kinetic_parallel:
            megacomplex: [parallel]
            activation:
                - type: instant
                  compartments:
                      "s1": 1
"""


def test_load_scheme():
    scheme = load_scheme(test_scheme_yml, format_name="yml_str")
    assert isinstance(scheme.library.megacomplex["parallel"], KineticMegacomplex)
    assert isinstance(scheme.experiments[0].datasets["kinetic_parallel"], ActivationDataModel)
