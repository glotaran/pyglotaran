from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from glotaran.builtin.elements.damped_oscillation import DampedOscillationElement
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
    "doas": DampedOscillationElement(
        label="damped-oscillation",
        type="damped-oscillation",
        oscillations={
            "osc": {
                "frequency": "osc.frequency",
                "rate": "osc.rate",
            },
        },
    ),
}


test_parameters_simulation = Parameters.from_dict(
    {
        "osc": [["frequency", 3], ["rate", 1]],
        "gaussian": [["center", 0], ["width", 10]],
    }
)
test_parameters = Parameters.from_dict(
    {
        "osc": [["frequency", 3], ["rate", 1, {"min": 0}]],
        "gaussian": [["center", 0], ["width", 10]],
    }
)

test_global_axis = np.array([0])
test_model_axis = np.arange(-10, 150, 1)
test_axies = {"spectral": test_global_axis, "time": test_model_axis}
test_clp = xr.DataArray(
    [
        [1],
        [1],
    ],
    coords=[
        (
            "clp_label",
            [
                "osc_sin",
                "osc_cos",
            ],
        ),
        ("spectral", test_global_axis.data),
    ],
).T


@pytest.mark.parametrize(
    "activation",
    (
        InstantActivation(
            type="instant",
            compartments={"osc": 1},
        ),
        GaussianActivation(
            type="gaussian",
            compartments={"osc": 1},
            center="irf.center",
            width="irf.width",
        ),
        GaussianActivation(
            type="gaussian",
            compartments={"osc": 1},
            center="irf.center",
            width="irf.width",
            shift=[0],
        ),
        MultiGaussianActivation(
            type="multi-gaussian",
            compartments={},
            center=["irf.center"],
            width=["irf.width", "irf.width"],
        ),
    ),
)
def test_coherent_artifact(activation: Activation):
    data_model = ActivationDataModel(elements=["does"], activations={"irf": activation})
    data_model.data = simulate(
        data_model, test_library, test_parameters_simulation, test_axies, clp=test_clp
    )
    experiments = [
        ExperimentModel(
            datasets={"dataset1": data_model},
        )
    ]
    optimization = Optimization(
        experiments,
        test_parameters,
        test_library,
        raise_exception=True,
        maximum_number_function_evaluations=25,
    )
    optimized_parameters, optimization_results, optimization_info = optimization.run()
    assert optimization_info.success
    print(test_parameters_simulation)
    print(optimized_parameters)
    assert optimized_parameters.close_or_equal(test_parameters_simulation)

    assert "dataset1" in optimization_results
    assert (
        "damped_oscillation_associated_amplitudes_damped-oscillation"
        in optimization_results["dataset1"]
    )
    assert (
        "damped_oscillation_associated_concentrations_damped-oscillation"
        in optimization_results["damped_oscillation"]
    )
    assert (
        "damped_oscillation_frequency_damped-oscillation" in optimization_results["damped_oscillation"]
    )
    assert "damped_oscillation_rate_damped-oscillation" in optimization_results["damped_oscillation"]
    assert (
        "damped_oscillation_phase_associated_amplitudes_damped-oscillation"
        in optimization_results["damped_oscillation"]
    )
    assert (
        "damped_oscillation_sin_associated_amplitudes_damped-oscillation"
        in optimization_results["damped_oscillation"]
    )
    assert (
        "damped_oscillation_cos_associated_amplitudes_damped-oscillation"
        in optimization_results["damped_oscillation"]
    )
    assert (
        "damped_oscillation_sin_associated_concentrations_damped-oscillation"
        in optimization_results["damped_oscillation"]
    )
    assert (
        "damped_oscillation_cos_associated_concentrations_damped-oscillation"
        in optimization_results["damped_oscillation"]
    )

if __name__ == "__main__":
    test_coherent_artifact(InstantActivation(
        type="instant",
        compartments={"osc": 1},
    ))
    test_coherent_artifact(GaussianActivation(
        type="gaussian",
        compartments={"osc": 1},
        center="gaussian.center",
        width="gaussian.width",
    ))
    test_coherent_artifact(GaussianActivation(
        type="gaussian",
        compartments={"osc": 1},
        center="gaussian.center",
        width="gaussian.width",
        shift=[0],
    ))
    test_coherent_artifact(MultiGaussianActivation(
        type="multi-gaussian",
        compartments={},
        center=["gaussian.center"],
        width=["gaussian.width", "gaussian.width"],
    ))