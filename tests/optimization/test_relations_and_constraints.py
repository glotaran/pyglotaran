from __future__ import annotations

from copy import deepcopy

import numpy as np
import xarray as xr

from glotaran.model.clp_constraint import ZeroConstraint
from glotaran.model.data_model import DataModel
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization.optimization import Optimization
from glotaran.parameter import Parameters
from glotaran.simulation import simulate
from tests.optimization.library import test_library


def test_zero_contraint():
    library = deepcopy(test_library)
    library["decay_independent"].clp_constraints = [ZeroConstraint(type="zero", target="c1")]
    library["constant"].clp_constraints = [ZeroConstraint(type="zero", target="c1")]
    data_model_one = DataModel(elements=["decay_independent"])
    data_model_two = DataModel(elements=["constant"])
    experiment = ExperimentModel(
        datasets={"decay_independent": data_model_one, "constant": data_model_two},
    )
    parameters = Parameters.from_dict({"rates": {"decay": [0.8, 0.04]}})

    global_axis = np.arange(10)
    model_axis = np.arange(0, 150, 1)
    clp = xr.DataArray(
        [[1, 10, 2]] * global_axis.size,
        coords=(("global", global_axis), ("clp_label", ["c1", "c2", "cc"])),
    )
    data_model_one.data = simulate(
        data_model_one,
        library,
        parameters,
        {"global": global_axis, "model": model_axis},
        clp,
    )
    data_model_two.data = simulate(
        data_model_two,
        library,
        parameters,
        {"global": global_axis, "model": model_axis},
        clp,
    )

    optimization = Optimization(
        models=[experiment],
        parameters=parameters,
        library=library,
        raise_exception=True,
        maximum_number_function_evaluations=1,
    )
    optimized_parameters, optimized_data, result = optimization.run()
    assert "decay_independent" in optimized_data
    assert "constant" in optimized_data
    print(optimized_parameters)
    assert result.success
    assert optimized_parameters.close_or_equal(parameters)
