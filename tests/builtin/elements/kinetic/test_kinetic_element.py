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
from glotaran.model.clp_constraint import ZeroConstraint
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


@pytest.mark.parametrize("decay", ("parallel", "sequential", "equilibrium"))
@pytest.mark.parametrize(
    "activation",
    (
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
    ),
)
def test_decay(decay: str, activation: Activation):
    if decay == "parallel":
        activation.compartments = {"s1": 1, "s2": 1}
    else:
        activation.compartments = {"s1": 1}
    data_model = ActivationDataModel(elements=[decay], activation=[activation])
    data_model.data = simulate(
        data_model, test_library, test_parameters_simulation, test_axies, clp=test_clp
    )
    experiments = [
        ExperimentModel(
            datasets={"decay": data_model},
            clp_constraints=[
                ZeroConstraint(type="zero", target="s1", interval=(1, 1)),
                ZeroConstraint(type="zero", target="s2", interval=(0, 0)),
            ]
            if decay == "equilibrium"
            else [],
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
    assert optimized_parameters.close_or_equal(test_parameters_simulation)
    assert "decay" in optimized_data
    print(optimized_data["decay"])
    assert "residual" in optimized_data["decay"]
    assert f"species_associated_concentrations_{decay}" in optimized_data["decay"]
    assert f"species_associated_amplitudes_{decay}" in optimized_data["decay"]
    assert f"kinetic_associated_amplitudes_{decay}" in optimized_data["decay"]
    assert f"k_matrix_{decay}" in optimized_data["decay"]
    if isinstance(activation, MultiGaussianActivation):
        assert "gaussian_activation" in optimized_data["decay"].coords
        assert "gaussian_activation_function" in optimized_data["decay"]
