import numpy as np
import xarray as xr

from glotaran.builtin.elements.kinetic import KineticElement
from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.parameter import Parameters
from glotaran.project.scheme import Scheme
from glotaran.simulation import simulate

test_scheme_dict = {
    "library": {
        "parallel": {"type": "kinetic", "rates": {("s1", "s1"): "rates.1"}},
    },
    "experiments": {
        "test_experiment": {
            "datasets": {
                "kinetic_parallel": {
                    "elements": ["parallel"],
                    "activation": [
                        {"type": "instant", "compartments": {"s1": 1}},
                    ],
                }
            }
        }
    },
}


test_parameters = Parameters.from_dict(
    {"rates": [0.1, 0.02, 0.08, {"min": 0}], "gaussian": [["center", 60], ["width", 8]]}
)
test_global_axis = np.array([0])
test_model_axis = np.arange(100)
test_axies = {"spectral": test_global_axis, "time": test_model_axis}
test_clp = xr.DataArray(
    [[1]],
    coords=[("clp_label", ["s1"]), ("spectral", test_global_axis.data)],
).T


def test_scheme():
    scheme = Scheme.from_dict(test_scheme_dict)
    assert "parallel" in scheme.library
    assert isinstance(scheme.library["parallel"], KineticElement)
    assert isinstance(
        scheme.experiments["test_experiment"].datasets["kinetic_parallel"], ActivationDataModel
    )
    scheme.experiments["test_experiment"].datasets["kinetic_parallel"].data = simulate(
        scheme.experiments["test_experiment"].datasets["kinetic_parallel"],
        scheme.library,
        test_parameters,
        test_axies,
        clp=test_clp,
    )
    result = scheme.optimize(test_parameters)
    assert result.optimization.success
