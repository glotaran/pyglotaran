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
    "coherent-artifact": CoherentArtifactElement(
        label="coherent-artifact", type="coherent-artifact", order=3
    ),
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
                "coherent_artifact_coherent-artifact_order_1",
                "coherent_artifact_coherent-artifact_order_2",
                "coherent_artifact_coherent-artifact_order_3",
            ],
        ),
        ("spectral", test_global_axis.data),
    ],
).T


@pytest.mark.parametrize(
    "activation",
    (
        GaussianActivation(
            type="gaussian",
            compartments={"coherent-artifact": 1},
            center="irf.center",
            width="irf.width",
        ),
        GaussianActivation(
            type="gaussian",
            compartments={"coherent-artifact": 1},
            center="irf.center",
            width="irf.width",
            shift=[0],
        ),
        MultiGaussianActivation(
            type="multi-gaussian",
            compartments={"coherent-artifact": 1},
            center=["irf.center"],
            width=["irf.width", "irf.width"],
        ),
    ),
)
def test_coherent_artifact(activation: Activation):
    element_label = "coherent-artifact"
    dataset_label = "dataset1"
    data_model = ActivationDataModel(
        elements=[element_label], activations={"irf": activation}
    )
    data_model.data = simulate(
        data_model, test_library, test_parameters_simulation, test_axies, clp=test_clp
    )
    experiments = [
        ExperimentModel(
            datasets={dataset_label: data_model},
        )
    ]
    optimization = Optimization(
        experiments,
        test_parameters,
        test_library,
        raise_exception=True,
        maximum_number_function_evaluations=25,
    )
    optimized_parameters, optimization_result, optimization_info = optimization.run()
    assert optimization_info.success
    assert optimized_parameters.close_or_equal(test_parameters_simulation)

    assert dataset_label in optimization_result
    assert element_label in optimization_result[dataset_label].elements
    ca_result = optimization_result[dataset_label].elements[element_label]
    assert "amplitudes" in ca_result.data_vars
    assert "concentrations" in ca_result.data_vars


if __name__ == "__main__":
    test_coherent_artifact(
        GaussianActivation(
            type="gaussian",
            compartments={"coherent-artifact": 1},
            center="irf.center",
            width="irf.width",
        )
    )
    test_coherent_artifact(
        GaussianActivation(
            type="gaussian",
            compartments={"coherent-artifact": 1},
            center="irf.center",
            width="irf.width",
            shift=[0],
        )
    )
    test_coherent_artifact(
        MultiGaussianActivation(
            type="multi-gaussian",
            compartments={"coherent-artifact": 1},
            center=["irf.center"],
            width=["irf.width", "irf.width"],
        )
    )
