from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import xarray as xr

from glotaran.model.data_model import DataModel
from glotaran.model.errors import GlotaranUserError
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization.info import OptimizationInfo
from glotaran.optimization.objective import OptimizationObjective
from glotaran.optimization.optimization import Optimization
from glotaran.optimization.optimization_history import OptimizationHistory
from glotaran.parameter import ParameterHistory
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
        models=[experiment],
        parameters=initial_parameters,
        library=test_library,
        raise_exception=True,
        maximum_number_function_evaluations=10,
    )
    optimized_parameters, optimization_results, optimization_info = optimization.run()
    print(optimized_parameters)
    assert optimization_info.success
    assert initial_parameters != optimized_parameters
    assert optimized_parameters.close_or_equal(parameters)
    assert "decay_independent" in optimization_results
    optimization_result = optimization_results["decay_independent"]
    print(optimization_result)
    assert optimization_result.residuals is not None
    assert optimization_result.fitted_data is not None


@pytest.mark.parametrize("verbose", [True, False])
def test_no_free_parameters_records_the_evaluation(verbose: bool):
    """Without free parameters the one evaluation is in the cost and parameter histories."""
    data_model = DataModel(elements=["decay_independent"])
    parameters = Parameters.from_dict(
        {"rates": {"decay": [[0.8, {"vary": False}], [0.04, {"vary": False}]]}}
    )
    clp = xr.DataArray(
        [[1, 10]] * 10, coords=(("global", np.arange(10)), ("clp_label", ["c1", "c2"]))
    )
    data_model.data = simulate(
        data_model,
        test_library,
        parameters,
        {"global": np.arange(10), "model": np.arange(0, 150, 1)},
        clp,
    )
    optimization = Optimization(
        models=[ExperimentModel(datasets={"decay_independent": data_model})],
        parameters=parameters,
        library=test_library,
        verbose=verbose,
    )
    _, _, optimization_info = optimization.run()

    assert optimization_info.number_of_function_evaluations == 1
    assert optimization.cost_history == [pytest.approx(optimization_info.cost, rel=1e-12)]
    assert optimization_info.parameter_history.number_of_records == (2 if verbose else 1)


def test_only_unused_free_parameter_evaluates_model_successfully():
    data_model = DataModel(elements=["decay_independent"])
    experiment = ExperimentModel(datasets={"decay_independent": data_model})
    parameters = Parameters.from_dict(
        {
            "rates": {
                "decay": [
                    [0.8, {"vary": False}],
                    [0.04, {"vary": False}],
                ]
            },
            "unused": [1.0],
        }
    )

    global_axis = np.arange(10)
    model_axis = np.arange(0, 150, 1)
    clp = xr.DataArray(
        [[1, 10]] * global_axis.size,
        coords=(("global", global_axis), ("clp_label", ["c1", "c2"])),
    )
    data_model.data = simulate(
        data_model, test_library, parameters, {"global": global_axis, "model": model_axis}, clp
    )

    optimized_parameters, optimization_results, optimization_info = Optimization(
        models=[experiment],
        parameters=parameters,
        library=test_library,
        raise_exception=True,
    ).run()

    assert optimization_info.success
    assert optimization_info.termination_reason == "No free parameters to optimize."
    assert optimization_info.free_parameter_labels == []
    assert optimization_info.number_of_function_evaluations == 1
    assert optimization_info.number_of_jacobian_evaluations == 0
    assert optimization_info.number_of_parameters == 0
    assert optimization_info.jacobian.shape == (optimization_info.number_of_data_points, 0)
    assert optimization_info.covariance_matrix.shape == (0, 0)
    assert parameters.get_label_value_and_bounds_arrays(exclude_non_vary=True)[0] == ["unused.1"]
    assert [parameter.label for parameter in optimized_parameters.all()] == [
        "rates.decay.1",
        "rates.decay.2",
    ]
    assert "decay_independent" in optimization_results


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
        models=experiments,
        parameters=initial_parameters,
        library=test_library,
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


def test_dataset_label_repeated_across_experiments_is_rejected():
    """Results are stored by dataset label, so one experiment's result would be lost."""
    experiments = [
        ExperimentModel(datasets={"shared": DataModel(elements=["decay_independent"])}),
        ExperimentModel(datasets={"shared": DataModel(elements=["decay_independent"])}),
    ]

    with pytest.raises(GlotaranUserError, match=r"\['shared'\] are used in more than one"):
        Optimization(
            models=experiments,
            parameters=Parameters.from_dict({"rates": {"decay": [0.8, 0.04]}}),
            library=test_library,
        )


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
        models=[experiment],
        parameters=initial_parameters,
        library=test_library,
        raise_exception=True,
        maximum_number_function_evaluations=10,
    )
    optimized_parameters, optimized_data, result = optimization.run()
    assert "decay_independent" in optimized_data
    print(optimized_parameters)
    assert result.success
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
        models=[experiment],
        parameters=initial_parameters,
        library=test_library,
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


def create_single_data_optimization(**kwargs: Any) -> Optimization:
    """Create the optimization of ``test_single_data`` with ``kwargs`` for ``Optimization``."""
    data_model = DataModel(elements=["decay_independent"])
    experiment = ExperimentModel(datasets={"decay_independent": data_model})
    parameters = Parameters.from_dict({"rates": {"decay": [0.8, 0.04]}})
    global_axis = np.arange(10)
    clp = xr.DataArray(
        [[1, 10]] * global_axis.size,
        coords=(("global", global_axis), ("clp_label", ["c1", "c2"])),
    )
    data_model.data = simulate(
        data_model,
        test_library,
        parameters,
        {"global": global_axis, "model": np.arange(0, 150, 1)},
        clp,
    )
    return Optimization(
        models=[experiment],
        parameters=Parameters.from_dict({"rates": {"decay": [0.9, 0.02]}}),
        library=test_library,
        **kwargs,
    )


def test_failed_objective_reports_the_optimization_error(monkeypatch: pytest.MonkeyPatch):
    """The evaluation after an exception in the objective raises the original error."""
    calls = []

    def calculate(self: OptimizationObjective) -> np.ndarray:
        calls.append(None)
        if len(calls) > 1:
            msg = f"evaluation {len(calls)}"
            raise ValueError(msg)
        return np.zeros(10)

    monkeypatch.setattr(OptimizationObjective, "calculate", calculate)
    optimization = create_single_data_optimization()

    with (
        pytest.warns(UserWarning, match="Optimization failed"),
        pytest.raises(ValueError, match="evaluation 2") as error,
    ):
        optimization.run()
    assert str(error.value.__cause__) == "evaluation 3"


@pytest.mark.parametrize("method", ["TrustRegionReflection", "Dogbox", "Levenberg-Marquardt"])
@pytest.mark.parametrize("verbose", [True, False])
def test_histories(method: str, verbose: bool):
    """The cost of every evaluation is collected; parameter values only with ``verbose``."""
    optimization = create_single_data_optimization(optimization_method=method, verbose=verbose)
    optimized_parameters, _, optimization_info = optimization.run()

    cost_history = optimization.cost_history
    # The Jacobian evaluations are counted too, SciPy's nfev does not count them
    assert len(cost_history) > optimization_info.number_of_function_evaluations
    # The result is at SciPy's solution, one of the evaluated points, often not the last one
    assert optimization_info.cost in cost_history
    assert optimization_info.chi_square == pytest.approx(2 * optimization_info.cost, rel=1e-13)

    parameter_history = optimization_info.parameter_history
    if verbose:
        history = parameter_history.to_dataframe()
        # One row with the initial values, then one row per evaluation
        assert list(history["iteration"]) == list(range(len(cost_history) + 1))
        assert optimized_parameters.get("rates.decay.1").value in list(history["rates.decay.1"])
    else:
        assert parameter_history.number_of_records == 1


def test_parameter_history_in_user_coordinates():
    """Non-negative parameters are stored with their value, not its logarithm."""
    optimization = create_single_data_optimization(verbose=True)
    optimization._parameters.get("rates.decay.1").non_negative = True
    optimized_parameters, _, optimization_info = optimization.run()

    history = optimization_info.parameter_history.to_dataframe()
    assert history["rates.decay.1"].iloc[0] == 0.9
    assert optimized_parameters.get("rates.decay.1").value in list(history["rates.decay.1"])


def test_dry_run_statistics():
    """A dry run reports the statistics of its one evaluation and is not successful."""
    _, _, info = create_single_data_optimization().dry_run()

    assert info.success is False
    assert info.termination_reason == "Dry run."
    assert info.number_of_data_points == 150 * 10
    assert info.number_of_parameters == 2
    assert info.degrees_of_freedom == (
        info.number_of_data_points - info.number_of_parameters - info.number_of_clps
    )
    assert info.chi_square == pytest.approx(2 * info.cost, rel=1e-12)
    assert info.root_mean_square_error == pytest.approx(
        np.sqrt(info.chi_square / info.degrees_of_freedom)
    )
    assert info.covariance_matrix is None


def test_dry_run_without_degrees_of_freedom():
    """A dry run with as many parameters and clps as data points has no reduced chi-square."""
    info = OptimizationInfo.from_least_squares_result(
        None,
        ParameterHistory(),
        OptimizationHistory.from_stdout_str(""),
        np.ones(3),
        0.0,
        ["a", "b"],
        "Dry run.",
        1,
        number_of_function_evaluations=1,
        dry_run=True,
    )

    assert info.degrees_of_freedom == 0
    assert info.chi_square == 3.0
    assert info.reduced_chi_square is None
    assert info.root_mean_square_error is None
