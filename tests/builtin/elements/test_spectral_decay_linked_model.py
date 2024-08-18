from __future__ import annotations

import numpy as np

from glotaran.builtin.elements.kinetic import KineticElement
from glotaran.builtin.elements.spectral import SpectralElement
from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.builtin.items.activation.gaussian import GaussianActivation
from glotaran.model.data_model import DataModel
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization import Optimization
from glotaran.parameter import Parameters
from glotaran.simulation import simulate

test_library = {
    "decay": KineticElement(
        **{
            "label": "decay",
            "type": "kinetic",
            "rates": {
                ("s1", "s1"): "rates.1",
                ("s2", "s2"): "rates.2",
            },
        }
    ),
    "spectral": SpectralElement(
        **{
            "label": "spectral",
            "type": "spectral",
            "shapes": {
                "s1": {
                    "type": "gaussian",
                    "amplitude": 250,
                    "location": 620,
                    "width": 50,
                },
                "s2": {
                    "type": "gaussian",
                    "amplitude": 300,
                    "location": 650,
                    "width": 60,
                },
            },
        }
    ),
}

test_parameters_simulation = Parameters.from_dict(
    {
        "rates": [0.5, 0.3],
        "activation": [
            ["center", 0.3],
            ["width", 0.1],
        ],
    },
)

test_parameters = Parameters.from_dict(
    {
        "rates": [0.55, 0.1, {"min": 0}],
        "activation": [
            ["center", 0.35],
            ["width", 0.05],
        ],
    },
)

test_global_axis = np.arange(600, 700, 1.4)
test_model_axis = np.arange(-1, 20, 0.01)
test_axies = {"spectral": test_global_axis, "time": test_model_axis}


test_data_model_simulation_cls = DataModel.create_class_for_elements(
    (KineticElement, SpectralElement)
)

test_activation_1 = [
    GaussianActivation(
        type="gaussian",
        compartments={"s1": 1, "s2": 0.75},
        center="activation.center",
        width="activation.width",
    ),
]

test_activation_2 = [
    GaussianActivation(
        type="gaussian",
        compartments={"s1": 1, "s2": 0.1},
        center="activation.center",
        width="activation.width",
    ),
]

test_data_1 = simulate(
    test_data_model_simulation_cls(
        elements=["decay"],
        global_elements=["spectral"],
        activation=test_activation_1,
    ),
    test_library,
    test_parameters_simulation,
    test_axies,
    noise=True,
    noise_seed=42,
    noise_std_dev=2,
)

test_data_2 = simulate(
    test_data_model_simulation_cls(
        elements=["decay"],
        global_elements=["spectral"],
        activation=test_activation_2,
    ),
    test_library,
    test_parameters_simulation,
    test_axies,
    noise=True,
    noise_seed=42,
    noise_std_dev=2,
)

test_experiments = [
    ExperimentModel(
        datasets={
            "decay_1": ActivationDataModel(
                elements=["decay"],
                data=test_data_1,
                activation=test_activation_1,
            ),
            "decay_2": ActivationDataModel(
                elements=["decay"],
                data=test_data_2,
                activation=test_activation_2,
            ),
        },
    ),
]


def test_spectral_decay_linking():
    optimization = Optimization(
        test_experiments,
        test_parameters,
        test_library,
        raise_exception=True,
        maximum_number_function_evaluations=25,
    )
    optimized_parameters, optimized_data, result = optimization.run()
    assert result.success
    print(optimized_parameters)
    print(test_parameters_simulation)
    assert optimized_parameters.close_or_equal(test_parameters_simulation, rtol=1e-1)
    sas_ds1 = optimized_data["decay_1"].species_associated_amplitude_decay.to_numpy()
    sas_sd2 = optimized_data["decay_2"].species_associated_amplitude_decay.to_numpy()
    print("Diff SAS: ", sas_ds1.sum() - sas_sd2.sum())
    print(sas_ds1[0, 0], sas_sd2[0, 0])
    assert not np.allclose(sas_ds1, np.zeros_like(sas_ds1))
    assert np.allclose(sas_ds1, sas_sd2)
