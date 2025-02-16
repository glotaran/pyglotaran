from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from glotaran.builtin.elements.kinetic import KineticElement
from glotaran.builtin.items.activation import Activation
from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.builtin.items.activation import GaussianActivation
from glotaran.builtin.items.activation import InstantActivation
from glotaran.builtin.items.activation import MultiGaussianActivation
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization import Optimization
from glotaran.parameter import Parameters
from glotaran.simulation import simulate

test_library = {
    "parallel": KineticElement(
        **{
            "label": "parallel",
            "type": "kinetic",
            "rates": {
                ("s1", "s1"): "rates.1",
                ("s2", "s2"): "rates.2",
            },
        }
    ),
    "sequential": KineticElement(
        **{
            "label": "sequential",
            "type": "kinetic",
            "rates": {
                ("s2", "s1"): "rates.1",
                ("s2", "s2"): "rates.2",
            },
        }
    ),
    "equilibrium": KineticElement(
        **{
            "label": "equilibrium",
            "type": "kinetic",
            "rates": {
                ("s2", "s1"): "rates.1",
                ("s2", "s2"): "rates.2",
                ("s1", "s2"): "rates.3",
            },
            "clp_constraints": [
                {"type": "zero", "target": "s1", "interval": (1, 1)},
                {"type": "zero", "target": "s2", "interval": (0, 0)},
            ],
        }
    ),
}


test_parameters_simulation = Parameters.from_dict(
    {"rates": [0.2, 0.01, 0.09], "gaussian": [["center", 50], ["width", 20]]}
)
test_parameters = Parameters.from_dict(
    {"rates": [0.1, 0.02, 0.08, {"min": 0}], "gaussian": [["center", 60], ["width", 8]]}
)

test_global_axis = np.array([0, 1])
test_model_axis = np.arange(-10, 1500, 1)
test_axies = {"spectral": test_global_axis, "time": test_model_axis}
test_clp = xr.DataArray(
    [
        [1, 0],
        [0, 1],
    ],
    coords=[("clp_label", ["s1", "s2"]), ("spectral", test_global_axis.data)],
).T


@pytest.mark.parametrize("decay_method", ["parallel", "sequential", "equilibrium"])
@pytest.mark.parametrize(
    "activation",
    [
        InstantActivation(type="instant", compartments={}),
        GaussianActivation(
            type="gaussian",
            compartments={},
            center="gaussian.center",
            width="gaussian.width",
        ),
        GaussianActivation(
            type="gaussian",
            compartments={},
            center="gaussian.center",
            width="gaussian.width",
            shift=[1, 0],
        ),
        MultiGaussianActivation(
            type="multi-gaussian",
            compartments={},
            center=["gaussian.center"],
            width=["gaussian.width", "gaussian.width"],
        ),
    ],
)
def test_decay(decay_method: str, activation: Activation):
    dataset_label = "dataset1"
    if decay_method == "parallel":
        activation.compartments = {"s1": 1, "s2": 1}
    else:
        activation.compartments = {"s1": 1}
    data_model = ActivationDataModel(elements=[decay_method], activations={"irf": activation})
    data_model.data = simulate(
        data_model, test_library, test_parameters_simulation, test_axies, clp=test_clp
    )
    experiments = [ExperimentModel(datasets={"dataset1": data_model})]
    optimization = Optimization(
        experiments,
        test_parameters,
        test_library,
        raise_exception=True,
        maximum_number_function_evaluations=25,
    )
    optimized_parameters, optimization_results, optimization_info = optimization.run()
    assert optimization_info.success
    assert optimized_parameters.close_or_equal(test_parameters_simulation)
    assert dataset_label in optimization_results
    optimization_result = optimization_results[dataset_label]

    assert optimization_result.residuals is not None
    assert optimization_result.elements is not None
    assert optimization_result.elements[decay_method] is not None
    decay_result = optimization_result.elements[decay_method]
    assert "amplitudes" in decay_result
    assert "concentrations" in decay_result
    assert "initial_concentrations" in decay_result
    assert "k_matrix" in decay_result
    assert "reduced_k_matrix" in decay_result
    assert "a_matrix" in decay_result
    assert "kinetic_amplitudes" in decay_result

    if isinstance(activation, MultiGaussianActivation):
        assert optimization_result.activations["irf"] is not None
        # TODO: check if we need more tests here
        # assert "gaussian_activation" in optimization_result["dataset1"].coords
        # assert "gaussian_activation_function" in optimization_result["dataset1"]
