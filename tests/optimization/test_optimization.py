from __future__ import annotations

import numpy as np
import xarray as xr

from glotaran.model.data_model import DataModel
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization.optimization import Optimization
from glotaran.parameter import Parameters
from glotaran.simulation import simulate
from tests.optimization.library import test_library


def test_single_data():
    data_model = DataModel(elements=["decay_independent"])
    experiment = ExperimentModel(datasets={"decay_independent": data_model})
    parameters = Parameters.from_dict({"rates": {"decay": [0.8, 0.04]}})

    global_axis = np.arange(10)
    model_axis = np.arange(0, 150, 1)
    clp = xr.DataArray(
        [[1, 10]] * global_axis.size,
        coords=(("global", global_axis), ("clp_label", ["c1", "c2"])),
    )
    data_model.data = simulate(
        data_model, test_library, parameters, {"global": global_axis, "model": model_axis}, clp
    )

    initial_parameters = Parameters.from_dict({"rates": {"decay": [0.9, 0.02]}})
    print(initial_parameters)
    optimization = Optimization(
        [experiment],
        initial_parameters,
        test_library,
        raise_exception=True,
        maximum_number_function_evaluations=10,
    )
    optimized_parameters, optimized_data, result = optimization.run()
    print(optimized_parameters)
    assert result.success
    assert initial_parameters != optimized_parameters
    assert optimized_parameters.close_or_equal(parameters)
    assert "decay_independent" in optimized_data
    result_data = optimized_data["decay_independent"]
    print(result_data)
    assert "residual" in result_data
    assert "fit" in result_data


def test_multiple_experiments():
    data_model = DataModel(elements=["decay_independent"])
    experiments = [
        ExperimentModel(datasets={"decay_independent_1": data_model}),
        ExperimentModel(datasets={"decay_independent_2": data_model}),
    ]
    parameters = Parameters.from_dict({"rates": {"decay": [0.8, 0.04]}})

    global_axis = np.arange(10)
    model_axis = np.arange(0, 150, 1)
    clp = xr.DataArray(
        [[1, 10]] * global_axis.size,
        coords=(("global", global_axis), ("clp_label", ["c1", "c2"])),
    )
    data_model.data = simulate(
        data_model, test_library, parameters, {"global": global_axis, "model": model_axis}, clp
    )

    initial_parameters = Parameters.from_dict({"rates": {"decay": [0.9, 0.02]}})
    print(initial_parameters)
    optimization = Optimization(
        experiments,
        initial_parameters,
        test_library,
        raise_exception=True,
        maximum_number_function_evaluations=10,
    )
    optimized_parameters, optimized_data, result = optimization.run()
    assert "decay_independent_1" in optimized_data
    assert "decay_independent_2" in optimized_data
    print(optimized_parameters)
    assert result.success
    assert initial_parameters != optimized_parameters
    assert optimized_parameters.close_or_equal(parameters)


def test_global_data():
    data_model = DataModel(elements=["decay_independent"], global_elements=["gaussian"])
    experiment = ExperimentModel(datasets={"decay_independent": data_model})
    parameters = Parameters.from_dict(
        {
            "rates": {"decay": [0.8, 0.04]},
            "gaussian": {
                "amplitude": [2.0, 3.0],
                "location": [3.0, 6.0],
                "width": [2.0, 4.0],
            },
        }
    )

    global_axis = np.arange(10)
    model_axis = np.arange(0, 150, 1)
    data_model.data = simulate(
        data_model, test_library, parameters, {"global": global_axis, "model": model_axis}
    )

    initial_parameters = Parameters.from_dict(
        {
            "rates": {"decay": [0.8, 0.04]},
            "gaussian": {
                "amplitude": [2.0, 3.0],
                "location": [3.0, 6.0],
                "width": [2.0, 4.0],
            },
        }
    )
    print(initial_parameters)
    optimization = Optimization(
        [experiment],
        initial_parameters,
        test_library,
        raise_exception=True,
        maximum_number_function_evaluations=10,
    )
    optimized_parameters, optimized_data, result = optimization.run()
    assert "decay_independent" in optimized_data
    print(optimized_parameters)
    assert result.success
    assert initial_parameters != optimized_parameters
    assert optimized_parameters.close_or_equal(parameters)


def test_multiple_data():
    data_model_one = DataModel(elements=["decay_independent"])
    data_model_two = DataModel(elements=["decay_dependent"])
    experiment = ExperimentModel(
        datasets={"decay_independent": data_model_one, "decay_dependent": data_model_two}
    )
    parameters = Parameters.from_dict({"rates": {"decay": [0.8, 0.04]}})

    global_axis = np.arange(10)
    model_axis = np.arange(0, 150, 1)
    clp = xr.DataArray(
        [[1, 10]] * global_axis.size,
        coords=(("global", global_axis), ("clp_label", ["c1", "c2"])),
    )
    data_model_one.data = simulate(
        data_model_one, test_library, parameters, {"global": global_axis, "model": model_axis}, clp
    )
    data_model_two.data = simulate(
        data_model_two, test_library, parameters, {"global": global_axis, "model": model_axis}, clp
    )

    initial_parameters = Parameters.from_dict({"rates": {"decay": [0.9, 0.02]}})
    print(initial_parameters)
    optimization = Optimization(
        [experiment],
        initial_parameters,
        test_library,
        raise_exception=True,
        maximum_number_function_evaluations=10,
    )
    optimized_parameters, optimized_data, result = optimization.run()
    assert "decay_independent" in optimized_data
    assert "decay_dependent" in optimized_data
    print(optimized_parameters)
    assert result.success
    assert initial_parameters != optimized_parameters
    assert optimized_parameters.close_or_equal(parameters)
