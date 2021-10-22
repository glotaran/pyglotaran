from __future__ import annotations

from copy import deepcopy

import pytest
from rich import pretty
from rich import print  # pylint: disable=W0622

from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.testing.model_generators import SimpleModelGenerator

pretty.install()


REF_PARAMETER_DICT = {
    "rates": [
        ["1", 501e-3],
        ["2", 202e-4],
        ["3", 105e-5],
        {"non-negative": True},
    ],
    "irf": [["center", 1.3], ["width", 7.8]],
    "inputs": [
        ["1", 1],
        ["0", 0],
        {"vary": False},
    ],
}

REF_MODEL_DICT = {
    "default_megacomplex": "decay",
    "initial_concentration": {
        "j1": {
            "compartments": ["s1", "s2", "s3"],
            "parameters": ["inputs.1", "inputs.0", "inputs.0"],
        },
    },
    "megacomplex": {
        "mc1": {"k_matrix": ["k1"]},
    },
    "k_matrix": {
        "k1": {
            "matrix": {
                ("s2", "s1"): "rates.1",
                ("s3", "s2"): "rates.2",
                ("s3", "s3"): "rates.3",
            }
        }
    },
    "irf": {
        "irf1": {
            "type": "multi-gaussian",
            "center": ["irf.center"],
            "width": ["irf.width"],
        },
    },
    "dataset": {
        "dataset1": {
            "initial_concentration": "j1",
            "irf": "irf1",
            "megacomplex": ["mc1"],
        },
    },
}


def simple_diff_between_string(string1, string2):
    return "".join(c2 for c1, c2 in zip(string1, string2) if c1 != c2)


def test_three_component_sequential_model():
    ref_model = Model.from_dict(deepcopy(REF_MODEL_DICT))
    ref_parameters = ParameterGroup.from_dict(deepcopy(REF_PARAMETER_DICT))
    generator = SimpleModelGenerator(
        rates=[501e-3, 202e-4, 105e-5, {"non-negative": True}],
        irf={"center": 1.3, "width": 7.8},
        k_matrix="sequential",
    )
    for key, _ in REF_PARAMETER_DICT.items():
        assert key in generator.parameters_dict
        # TODO: check contents

    model, parameters = generator.model_and_parameters
    assert str(ref_model) == str(model), print(
        simple_diff_between_string(str(model), str(ref_model))
    )
    assert str(ref_parameters) == str(parameters), print(
        simple_diff_between_string(str(parameters), str(ref_parameters))
    )


def test_only_rates_no_irf():
    generator = SimpleModelGenerator(rates=[0.1, 0.02, 0.003])
    assert "irf" not in generator.model_dict.keys()


def test_no_rates():
    generator = SimpleModelGenerator()
    assert generator.valid is False


def test_one_rate():
    generator = SimpleModelGenerator([1])
    assert generator.valid is True
    assert "is valid" in generator.validate()


def test_rates_not_a_list():
    generator = SimpleModelGenerator(1)
    assert generator.valid is False
    with pytest.raises(ValueError):
        print(generator.validate())


def test_set_rates_delayed():
    generator = SimpleModelGenerator()
    generator.rates = [1, 2, 3]
    assert generator.valid is True
