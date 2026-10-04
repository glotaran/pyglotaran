from __future__ import annotations

import numpy as np
import xarray as xr

from glotaran.builtin.elements.kinetic import KineticElement
from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.parameter import Parameters
from glotaran.project.scheme import Scheme
from glotaran.simulation import simulate
from glotaran.testing.simulated_data.sequential_spectral_decay import DATASET
from glotaran.testing.simulated_data.sequential_spectral_decay import SCHEME_DICT
from glotaran.testing.simulated_data.shared_decay import PARAMETERS

test_scheme_dict = {
    "library": {
        "parallel": {
            "type": "kinetic",
            "rates": {("s1", "s1"): "rates.1", ("s2", "s2"): "rates.2"},
        },
        "extended": {
            "type": "kinetic",
            "rates": {("s1", "s1"): "rates.4", ("s3", "s3"): "rates.3"},
            "extends": ["parallel"],
        },
        "nested_extend": {"type": "kinetic", "extends": ["extended"], "rates": {}},
    },
    "experiments": {
        "test_experiment": {
            "datasets": {
                "kinetic_parallel": {
                    "elements": ["parallel"],
                    "activations": {
                        "irf": {"type": "instant", "compartments": {"s1": 1}},
                    },
                }
            }
        }
    },
}


test_parameters = Parameters.from_dict(
    {"rates": [0.1, 0.02, 0.08, 0.2, {"min": 0}], "gaussian": [["center", 60], ["width", 8]]}
)
test_global_axis = np.array([0])
test_model_axis = np.arange(100)
test_axies = {"spectral": test_global_axis, "time": test_model_axis}
test_clp = xr.DataArray(
    [[1], [2]],
    coords=[("clp_label", ["s1", "s2"]), ("spectral", test_global_axis.data)],
).T


def test_scheme():
    scheme = Scheme.from_dict(test_scheme_dict)
    assert "parallel" in scheme.library
    assert isinstance(scheme.library["parallel"], KineticElement)
    assert "extended" in scheme.library
    extended = scheme.library["extended"]
    assert isinstance(extended, KineticElement)
    assert ("s1", "s1") in extended.rates
    assert extended.rates[("s1", "s1")] == "rates.4"
    assert ("s2", "s2") in extended.rates
    assert extended.rates[("s2", "s2")] == "rates.2"
    assert ("s3", "s3") in extended.rates
    assert extended.rates[("s3", "s3")] == "rates.3"
    assert isinstance(
        scheme.experiments["test_experiment"].datasets["kinetic_parallel"], ActivationDataModel
    )
    datasets = {
        "kinetic_parallel": simulate(
            scheme.experiments["test_experiment"].datasets["kinetic_parallel"],
            scheme.library,
            test_parameters,
            test_axies,
            clp=test_clp,
        )
    }
    result = scheme.optimize(test_parameters, datasets)
    assert result.optimization_info.success


def test_scheme_round_trip_serialization():
    """Dumping a scheme returns the original dict."""
    assert Scheme.from_dict(test_scheme_dict).model_dump(exclude_unset=True) == test_scheme_dict


def test_optimize_result_does_not_share_inputs():
    """A later fit, a scheme edit or an in-place data change leave an earlier result unchanged."""
    scheme = Scheme.from_dict(SCHEME_DICT)
    parameters = PARAMETERS.copy()
    data = DATASET.copy(deep=True)
    result = scheme.optimize(parameters, {"sequential-decay": data}, verbose=False)

    assert result.scheme is not scheme
    assert result.initial_parameters is not parameters
    input_data = result.input_data["sequential-decay"].copy(deep=True)
    fitted_data = result.optimization_results["sequential-decay"].fitted_data.copy(deep=True)
    scheme_dump = result.scheme.model_dump()
    initial_parameters = result.initial_parameters.copy()

    scheme.optimize(parameters, {"sequential-decay": data * 2}, verbose=False)
    scheme.library["sequential"].rates[("s2", "s1")] = "rates.species_2"
    parameters.get("rates.species_1").value = 10
    data["data"].data[:] = 0

    assert result.input_data["sequential-decay"].equals(input_data)
    assert result.optimization_results["sequential-decay"].fitted_data.equals(fitted_data)
    assert result.scheme.model_dump() == scheme_dump
    assert (
        result.scheme.experiments["sequential-decay"]
        .datasets["sequential-decay"]
        .data["data"]
        .any()
    )
    assert result.initial_parameters == initial_parameters
