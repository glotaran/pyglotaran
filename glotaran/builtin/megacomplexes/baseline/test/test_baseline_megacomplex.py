import numpy as np

from glotaran.analysis.util import calculate_matrix
from glotaran.builtin.megacomplexes.baseline import BaselineMegacomplex
from glotaran.builtin.megacomplexes.decay import DecayMegacomplex
from glotaran.model import Model
from glotaran.parameter import ParameterGroup


def test_baseline():
    model = Model.from_dict(
        {
            "initial_concentration": {
                "j1": {"compartments": ["s1"], "parameters": ["2"]},
            },
            "megacomplex": {
                "mc1": {"type": "decay", "k_matrix": ["k1"]},
                "mc2": {"type": "baseline", "dimension": "time"},
            },
            "k_matrix": {
                "k1": {
                    "matrix": {
                        ("s1", "s1"): "1",
                    }
                }
            },
            "dataset": {
                "dataset1": {
                    "initial_concentration": "j1",
                    "megacomplex": ["mc1", "mc2"],
                },
            },
        },
        megacomplex_types={"decay": DecayMegacomplex, "baseline": BaselineMegacomplex},
    )

    parameter = ParameterGroup.from_list(
        [
            101e-4,
            [1, {"vary": False, "non-negative": False}],
            [42, {"vary": False, "non-negative": False}],
        ]
    )

    time = np.asarray(np.arange(0, 50, 1.5))
    pixel = np.asarray([0])
    coords = {"time": time, "pixel": pixel}
    dataset_model = model.dataset["dataset1"].fill(model, parameter)
    dataset_model.overwrite_global_dimension("pixel")
    dataset_model.set_coordinates(coords)
    matrix = calculate_matrix(dataset_model, {})
    compartments = matrix.clp_labels

    assert len(compartments) == 2
    assert "dataset1_baseline" in compartments

    assert matrix.matrix.shape == (time.size, 2)
    assert np.all(matrix.matrix[:, 1] == 1)
