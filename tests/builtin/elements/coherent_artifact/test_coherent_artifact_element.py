from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from glotaran.builtin.elements.coherent_artifact import CoherentArtifactElement
from glotaran.builtin.items.activation import Activation
from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.builtin.items.activation import GaussianActivation
from glotaran.builtin.items.activation import MultiGaussianActivation
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization import Optimization
from glotaran.parameter import Parameters
from glotaran.simulation import simulate

test_library = {
    "ca": CoherentArtifactElement(label="ca", type="coherent-artifact", order=3),
}


test_parameters_simulation = Parameters.from_dict({"irf": [["center", 50], ["width", 20]]})
test_parameters = Parameters.from_dict({"irf": [["center", 60], ["width", 8]]})

test_global_axis = np.array([0])
test_model_axis = np.arange(-10, 1500, 1)
test_axies = {"spectral": test_global_axis, "time": test_model_axis}
test_clp = xr.DataArray(
    [
        [1],
        [1],
        [1],
    ],
    coords=[
        (
            "clp_label",
            [
                "ca_derivative_0",
                "ca_derivative_1",
                "ca_derivative_2",
            ],
        ),
        ("spectral", test_global_axis.data),
    ],
).T


@pytest.mark.parametrize(
    "activation",
    [
        GaussianActivation(
            type="gaussian",
            compartments={"ca": 1},
            center="irf.center",
            width="irf.width",
        ),
        GaussianActivation(
            type="gaussian",
            compartments={"ca": 1},
            center="irf.center",
            width="irf.width",
            shift=[0],
        ),
        MultiGaussianActivation(
            type="multi-gaussian",
            compartments={"ca": 1},
            center=["irf.center"],
            width=["irf.width", "irf.width"],
        ),
    ],
)
def test_coherent_artifact(activation: Activation):
    element_label = "ca"
    dataset_label = "dataset1"
    data_model = ActivationDataModel(elements=[element_label], activations={"irf": activation})
    data_model.data = simulate(
        data_model, test_library, test_parameters_simulation, test_axies, clp=test_clp
    )
    experiments = [
        ExperimentModel(
            datasets={dataset_label: data_model},
        )
    ]
    optimization = Optimization(
        models=experiments,
        parameters=test_parameters,
        library=test_library,
        raise_exception=True,
        maximum_number_function_evaluations=25,
    )
    optimized_parameters, optimization_results, optimization_info = optimization.run()
    assert optimization_info.success
    assert optimized_parameters.close_or_equal(test_parameters_simulation)

    assert dataset_label in optimization_results
    assert element_label in optimization_results[dataset_label].elements
    ca_result = optimization_results[dataset_label].elements[element_label]
    assert "amplitudes" in ca_result.data_vars
    assert "concentrations" in ca_result.data_vars


if __name__ == "__main__":
    pytest.main([__file__])
