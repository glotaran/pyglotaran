from __future__ import annotations

import numpy as np

from glotaran.builtin.elements.kinetic import KineticElement
from glotaran.builtin.elements.spectral import SpectralElement
from glotaran.builtin.elements.spectral.element import SpectralDataModel
from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.builtin.items.activation import InstantActivation
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
                ("s2", "s1"): "rates.1",
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
                    "amplitude": "shape.1.amplitude",
                    "location": "shape.1.location",
                    "width": "shape.1.width",
                },
                "s2": {
                    "type": "gaussian",
                    "amplitude": "shape.2.amplitude",
                    "location": "shape.2.location",
                    "width": "shape.2.width",
                },
            },
        }
    ),
}

test_parameters_simulation = Parameters.from_dict(
    {
        "rates": [0.2, 0.01],
        "shape": {
            "1": [
                ["amplitude", 10],
                ["location", 10],
                ["width", 10],
            ],
            "2": [
                ["amplitude", 10],
                ["location", 10],
                ["width", 10],
            ],
        },
    },
)

test_parameters = Parameters.from_dict(
    {
        "rates": [0.2, 0.01, {"min": 0}],
        "shape": {
            "1": [
                ["amplitude", 10],
                ["location", 10],
                ["width", 10],
            ],
            "2": [
                ["amplitude", 10],
                ["location", 10],
                ["width", 10],
            ],
        },
    },
)

test_global_axis = np.arange(0, 50)
test_model_axis = np.arange(-10, 1500, 1)
test_axies = {"spectral": test_global_axis, "time": test_model_axis}
test_activation = {"no_irf": InstantActivation(type="instant", compartments={"s1": 1})}
test_data_model_cls = DataModel.create_class_for_elements((KineticElement, SpectralElement))
test_data = simulate(
    test_data_model_cls(
        elements=["decay"],
        global_elements=["spectral"],
        activations=test_activation,
    ),
    test_library,
    test_parameters_simulation,
    test_axies,
)

test_experiments = [
    ExperimentModel(
        datasets={
            "decay": ActivationDataModel(
                elements=["decay"], data=test_data, activations=test_activation
            ),
        },
    ),
    ExperimentModel(
        datasets={
            "spectral": SpectralDataModel(elements=["spectral"], data=test_data),
        },
    ),
    ExperimentModel(
        datasets={
            "spectral-decay": test_data_model_cls(
                elements=["decay"],
                global_elements=["spectral"],
                data=test_data,
                activations=test_activation,
            ),
        },
    ),
    ExperimentModel(
        datasets={
            "decay-spectral": test_data_model_cls(
                elements=["spectral"],
                global_elements=["decay"],
                data=test_data,
                activations=test_activation,
            ),
        },
    ),
]


def test_spectral_decay():
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
    assert optimized_parameters.close_or_equal(test_parameters_simulation)
