import numpy as np
import xarray as xr

from glotaran.analysis.util import calculate_matrix
from glotaran.builtin.models.kinetic_image import KineticImageModel
from glotaran.parameter import ParameterGroup


def test_baseline():
    model = KineticImageModel.from_dict(
        {
            "initial_concentration": {
                "j1": {"compartments": ["s1"], "parameters": ["2"]},
            },
            "megacomplex": {
                "mc1": {"type": "kinetic-decay", "k_matrix": ["k1"]},
                "mc2": {"type": "kinetic-baseline"},
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
        }
    )

    parameter = ParameterGroup.from_list(
        [
            101e-4,
            [1, {"vary": False, "non-negative": False}],
            [42, {"vary": False, "non-negative": False}],
        ]
    )

    time = xr.DataArray(np.asarray(np.arange(0, 50, 1.5)))
    pixel = xr.DataArray([0])
    coords = {"time": time, "pixel": pixel}
    dataset_model = model.dataset["dataset1"].fill(model, parameter)
    dataset_model.overwrite_global_dimension("pixel")
    dataset_model.set_coords(coords)
    matrix = calculate_matrix(dataset_model, {})
    compartments = matrix.coords["clp_label"]

    assert len(compartments) == 2
    assert compartments[0] == "dataset1_baseline"

    assert matrix.shape == (time.size, 2)
    assert np.all(matrix[:, 0] == 1)
