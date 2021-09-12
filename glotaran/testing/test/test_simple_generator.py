from __future__ import annotations

from copy import deepcopy

from rich import pretty
from rich import print

from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.testing.simple_generator import SimpleGenerator

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
    "default-megacomplex": "decay",
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
    # finding the common chars from both strings
    result = ""
    for c1, c2 in zip(string1, string2):
        if not c1 == c2:
            result += c2
    return result


def test_simple_model_3comp_seq():
    ref_model = Model.from_dict(deepcopy(REF_MODEL_DICT))
    ref_parameters = ParameterGroup.from_dict(deepcopy(REF_PARAMETER_DICT))
    generator = SimpleGenerator(
        rates=[501e-3, 202e-4, 105e-5, {"non-negative": True}],
        irf={"center": 1.3, "width": 7.8},
        k_matrix="sequential",
    )
    for key, value in REF_PARAMETER_DICT.items():
        assert key in generator.parameters_dict
        # TODO: check contents

    model, parameters = generator.model_and_parameters
    assert str(ref_model) == str(model), print(
        simple_diff_between_string(str(model), str(ref_model))
    )
    assert str(ref_parameters) == str(parameters), print(
        simple_diff_between_string(str(parameters), str(ref_parameters))
    )


if __name__ == "__main__":
    test_simple_model_3comp_seq()
