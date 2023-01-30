import numpy as np
import xarray as xr

from glotaran.model import DataModel
from glotaran.model import ExperimentModel
from glotaran.optimization.optimization import Optimization
from glotaran.optimization.test.library import TestLibrary
from glotaran.parameter import Parameters
from glotaran.simulation import simulate


def test_single_data():
    data_model = DataModel(megacomplex=["decay_independent"])
    experiment = ExperimentModel(datasets={"decay_independent": data_model})
    parameters = Parameters.from_dict({"rates": {"decay": [0.8, 0.04]}})

    global_axis = np.arange(10)
    model_axis = np.arange(0, 150, 1)
    clp = xr.DataArray(
        [[1, 10]] * global_axis.size,
        coords=(("global", global_axis), ("clp_label", ["c1", "c2"])),
    )
    data_model.data = simulate(
        data_model, TestLibrary, parameters, {"global": global_axis, "model": model_axis}, clp
    )

    initial_parameters = Parameters.from_dict({"rates": {"decay": [0.9, 0.02]}})
    print(initial_parameters)
    optimization = Optimization(
        [experiment],
        initial_parameters,
        TestLibrary,
        raise_exception=True,
        maximum_number_function_evaluations=10,
    )
    optimized_parameters, optimized_data, result = optimization.run()
    assert "decay_independent" in optimized_data
    print(optimized_parameters)
    assert result.success
    assert initial_parameters != optimized_parameters
    for param in optimized_parameters.all():
        if param.vary:
            assert np.allclose(param.value, parameters.get(param.label).value, rtol=1e-1)


def test_multiple_data():
    data_model_one = DataModel(megacomplex=["decay_independent"])
    data_model_two = DataModel(megacomplex=["decay_dependent"])
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
        data_model_one, TestLibrary, parameters, {"global": global_axis, "model": model_axis}, clp
    )
    data_model_two.data = simulate(
        data_model_two, TestLibrary, parameters, {"global": global_axis, "model": model_axis}, clp
    )

    initial_parameters = Parameters.from_dict({"rates": {"decay": [0.9, 0.02]}})
    print(initial_parameters)
    optimization = Optimization(
        [experiment],
        initial_parameters,
        TestLibrary,
        raise_exception=True,
        maximum_number_function_evaluations=10,
    )
    optimized_parameters, optimized_data, result = optimization.run()
    assert "decay_independent" in optimized_data
    assert "decay_dependent" in optimized_data
    print(optimized_parameters)
    print(optimized_parameters)
    assert result.success
    assert initial_parameters != optimized_parameters
    for param in optimized_parameters.all():
        if param.vary:
            assert np.allclose(param.value, parameters.get(param.label).value, rtol=1e-1)
