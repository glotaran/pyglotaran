import numpy as np
import pytest
import xarray as xr

from glotaran.builtin.items.activation import Activation
from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.builtin.items.activation import GaussianActivation
from glotaran.builtin.items.activation import MultiGaussianActivation
from glotaran.builtin.models.coherent_artifact import CoherentArtifactModel
from glotaran.model import ExperimentModel
from glotaran.optimization import Optimization
from glotaran.parameter import Parameters
from glotaran.simulation import simulate

test_library = {
    "coherent-artifact": CoherentArtifactModel(
        label="coherent-artifact", type="coherent-artifact", order=3
    ),
}


test_parameters_simulation = Parameters.from_dict({"gaussian": [["center", 50], ["width", 20]]})
test_parameters = Parameters.from_dict({"gaussian": [["center", 60], ["width", 8]]})

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
                "coherent_artifact_coherent-artifact_order_1_activation_0",
                "coherent_artifact_coherent-artifact_order_2_activation_0",
                "coherent_artifact_coherent-artifact_order_3_activation_0",
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
            center="gaussian.center",
            width="gaussian.width",
        ),
        GaussianActivation(
            type="gaussian",
            compartments={"coherent-artifact": 1},
            center="gaussian.center",
            width="gaussian.width",
            shift=[0],
        ),
        MultiGaussianActivation(
            type="multi-gaussian",
            compartments={"coherent-artifact": 1},
            center=["gaussian.center"],
            width=["gaussian.width", "gaussian.width"],
        ),
    ),
)
def test_coherent_artifact(activation: Activation):
    data_model = ActivationDataModel(models=["coherent-artifact"], activation=[activation])
    data_model.data = simulate(
        data_model, test_library, test_parameters_simulation, test_axies, clp=test_clp
    )
    experiments = [
        ExperimentModel(
            datasets={"coherent_artifact": data_model},
        )
    ]
    optimization = Optimization(
        experiments,
        test_parameters,
        test_library,
        raise_exception=True,
        maximum_number_function_evaluations=25,
    )
    optimized_parameters, optimized_data, result = optimization.run()
    assert result.success
    print(test_parameters_simulation)
    print(optimized_parameters)
    assert optimized_parameters.close_or_equal(test_parameters_simulation)

    assert "coherent_artifact" in optimized_data
    assert "coherent_artifact_response" in optimized_data["coherent_artifact"]
    assert "coherent_artifact_associated_estimation" in optimized_data["coherent_artifact"]
