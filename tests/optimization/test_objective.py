from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from glotaran.io import SAVING_OPTIONS_MINIMAL
from glotaran.io import save_dataset
from glotaran.model.clp_penalties import EqualAreaPenalty
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization.data import LinkedOptimizationData
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.objective import OptimizationObjective
from glotaran.optimization.objective import OptimizationResult
from glotaran.optimization.objective import OptimizationResultMetaData
from glotaran.optimization.objective import calculate_root_mean_square_error
from glotaran.plugin_system.data_io_registration import get_data_io
from glotaran.testing.plugin_system import monkeypatch_plugin_registry_data_io
from tests.optimization.data import TestDataModelConstantIndexDependent
from tests.optimization.data import TestDataModelConstantIndexIndependent
from tests.optimization.data import TestDataModelGlobal

STUB_META_DATA = OptimizationResultMetaData(
    global_dimension="global_dim",
    model_dimension="model_dim",
    root_mean_square_error=0.1,
)


# Values taken from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
# and calculated sqrt.
# The current root_mean_squared_error of sklearn currently has a bug where the mean of the sqrt
#  is calculated instead of the sqrt of the mean.
# Ref.: https://github.com/scikit-learn/scikit-learn/blob/1eb422d6c5f46a98a318f341de3e4709f9521bfe/sklearn/metrics/_regression.py
@pytest.mark.parametrize(
    ("residual", "expected_rmse"),
    [
        (xr.DataArray(np.array([0.5, -0.5, 0, -1]), dims=("flat")), 0.612372),
        (
            xr.DataArray(
                np.array([[0.5, -1.0], [0.0, -1.0], [-1.0, -1.0]]),
                dims=("model_dim", "global_dim"),
            ),
            0.841625,
        ),
    ],
)
def test_calculate_root_mean_square_error(residual: xr.DataArray, expected_rmse: float):
    """Test calculation of root mean square error in OptimizationResultMetaData."""
    assert calculate_root_mean_square_error(residual) == pytest.approx(expected_rmse)


def test_optimization_result_default_serde(tmp_path: Path):
    """Default serialization of ``OptimizationResult`` is round-trippable."""
    save_folder = tmp_path / "save_folder"
    foo_data = xr.Dataset({"data": ("dim_0", np.arange(2))})
    bar_data = xr.Dataset({"data": ("dim_0", np.arange(3) + 10)})
    input_data = xr.Dataset({"data": ("dim_0", np.arange(4))})
    residuals = xr.Dataset({"data": ("dim_0", np.arange(4) * 0.001)})
    optimization_result = OptimizationResult(
        elements={"foo": foo_data},
        activations={"bar": bar_data},
        input_data=input_data,
        residuals=residuals,
        meta=STUB_META_DATA,
    )
    # Instance validation passes without error
    OptimizationResult.model_validate(optimization_result)

    serialized = optimization_result.model_dump(context={"save_folder": save_folder}, mode="json")

    assert save_folder.is_dir() is True
    assert serialized["elements"]["foo"] == "foo.nc"
    assert (save_folder / "elements" / "foo.nc").is_file() is True
    assert serialized["activations"]["bar"] == "bar.nc"
    assert (save_folder / "activations" / "bar.nc").is_file() is True
    assert serialized["input_data"] == "input_data.nc"
    assert (save_folder / "input_data.nc").is_file() is True
    assert serialized["residuals"] == "residuals.nc"
    assert (save_folder / "residuals.nc").is_file() is True

    # Round Trip
    deserialized = OptimizationResult.model_validate(
        serialized, context={"save_folder": save_folder}
    )
    expected_dataset_attrs = STUB_META_DATA.model_dump(exclude_defaults=True)
    assert deserialized.elements["foo"].equals(foo_data)
    assert deserialized.elements["foo"].attrs.items() >= expected_dataset_attrs.items()
    assert deserialized.activations["bar"].equals(bar_data)
    assert deserialized.activations["bar"].attrs.items() >= expected_dataset_attrs.items()
    assert deserialized.input_data.equals(input_data)
    assert deserialized.residuals is not None
    assert deserialized.residuals.equals(residuals)
    assert deserialized.residuals.attrs.items() >= expected_dataset_attrs.items()
    assert deserialized.fitted_data is not None
    assert deserialized.fitted_data.equals(optimization_result.fitted_data)
    assert deserialized.fitted_data.attrs.items() >= expected_dataset_attrs.items()


def test_optimization_result_minimal_serde(tmp_path: Path):
    """Default serialization of ``OptimizationResult`` is round-trippable."""
    save_folder = tmp_path / "save_folder"
    input_data = xr.Dataset({"data": ("dim_0", np.arange(4))})
    save_dataset(input_data, tmp_path / "my_input_data.nc")

    assert input_data.attrs["source_path"] == (tmp_path / "my_input_data.nc").resolve().as_posix()
    assert input_data.attrs["io_plugin_name"] == "glotaran.builtin.io.netCDF.netCDF.NetCDFDataIo"

    optimization_result = OptimizationResult(
        elements={"foo": xr.Dataset({"data": ("dim_0", np.arange(2))})},
        activations={"bar": xr.Dataset({"data": ("dim_0", np.arange(3) + 10)})},
        input_data=input_data,
        residuals=xr.Dataset({"data": ("dim_0", np.arange(4) * 0.001)}),
        meta=STUB_META_DATA,
    )

    serialized = optimization_result.model_dump(
        context={"save_folder": save_folder, "saving_options": SAVING_OPTIONS_MINIMAL}, mode="json"
    )

    assert save_folder.exists() is False
    assert len(serialized["elements"]) == 0
    assert (save_folder / "elements").is_dir() is False
    assert len(serialized["activations"]) == 0
    assert (save_folder / "activations").is_dir() is False
    assert serialized["input_data"] == "../my_input_data.nc"
    assert (save_folder / "input_data.nc").is_file() is False
    assert (save_folder / "../my_input_data.nc").resolve().is_file() is True
    assert serialized["residuals"] is None

    deserialized = OptimizationResult.model_validate(
        serialized, context={"save_folder": save_folder}
    )

    assert deserialized.elements == {}
    assert deserialized.activations == {}
    assert deserialized.input_data.equals(optimization_result.input_data)
    assert deserialized.residuals is None


def test_optimization_result_noop_validation():
    """Self validation of already initialized object works."""
    optimization_result = OptimizationResult(
        elements={"foo": xr.Dataset({"data": ("dim_0", np.arange(2))})},
        activations={"bar": xr.Dataset({"data": ("dim_0", np.arange(3) + 10)})},
        input_data=xr.Dataset({"data": ("dim_0", np.arange(4))}),
        residuals=xr.Dataset({"data": ("dim_0", np.arange(4) * 0.001)}),
        meta=STUB_META_DATA,
    )
    OptimizationResult.model_validate(optimization_result)


def test_optimization_result_fitted_data_warn_on_missing_residuals():
    """Warn when fitted data is accessed without residuals."""
    optimization_result = OptimizationResult(
        input_data=xr.Dataset({"data": ("dim_0", np.arange(4))}), meta=STUB_META_DATA
    )

    with pytest.warns(
        UserWarning, match=r"Residuals must be set to calculate fitted data\."
    ) as warn_records:
        assert optimization_result.fitted_data is None
    assert len(warn_records) == 1
    assert Path(warn_records[0].filename).samefile(__file__), warn_records[0]


def test_optimization_result_error_missing_serialization_context():
    """Raise value error when serializing in ``json`` mode without ``save_folder`` in context."""
    optimization_result = OptimizationResult(
        input_data=xr.Dataset({"data": ("dim_0", np.arange(4))}),
        residuals=xr.Dataset({"data": ("dim_0", np.arange(4) * 0.001)}),
        meta=STUB_META_DATA,
    )

    with pytest.raises(ValueError) as exec_info:
        optimization_result.model_dump(mode="json")
    assert "SerializationInfo context is missing 'save_folder':" in str(exec_info.value)

    # No error with default python serialization
    optimization_result.model_dump()


def test_optimization_result_error_missing_missing_input_data():
    """Raise validation error when ``input_data`` is None."""
    with pytest.raises(ValidationError) as exec_info:
        OptimizationResult.model_validate({"input_data": None})
    assert "Input data cannot be None." in str(exec_info.value)


def test_optimization_result_error_bad_input_data_tuple(tmp_path: Path):
    """Raise validation error when ``input_data`` is a tuple but shape is unexpected."""
    context = {"save_folder": tmp_path}
    with pytest.raises(ValidationError) as exec_info:
        OptimizationResult.model_validate({"input_data": ("foo", "bar", "baz")}, context=context)
    assert (
        "Expected a tuple/list of relative file path and io plugin name for deserializing "
        "'input_data' dataset, got: ('foo', 'bar', 'baz')" in str(exec_info.value)
    )

    with pytest.raises(ValidationError) as exec_info:
        OptimizationResult.model_validate({"input_data": (1, "bar")}, context=context)
    assert (
        "Expected a tuple/list of relative file path and io plugin name for deserializing "
        "'input_data' dataset, got: (1, 'bar')" in str(exec_info.value)
    )


@pytest.mark.filterwarnings("ignore:Residuals must be set to calculate fitted data")
def test_optimization_result_input_data_read_with_3rd_party_plugin(tmp_path: Path):
    """Load input data which where originally saved with a 3rd party plugin."""
    input_data = xr.Dataset({"data": ("time", np.arange(4))})
    save_path = tmp_path / "input_data.foo"
    nc_plugin = get_data_io("nc")
    mock_plugin_name = "foo.FooNc"
    save_folder = tmp_path / "save_folder"

    with monkeypatch_plugin_registry_data_io({mock_plugin_name: nc_plugin}):
        save_dataset(input_data, save_path, format_name=mock_plugin_name)
        input_data.attrs["io_plugin_name"] = mock_plugin_name

        assert input_data.attrs["source_path"] == save_path.as_posix()

        serialized_optimization_result = OptimizationResult(
            input_data=input_data, meta=STUB_META_DATA
        ).model_dump(
            context={"save_folder": save_folder, "saving_options": SAVING_OPTIONS_MINIMAL},
            mode="json",
        )
        assert serialized_optimization_result["input_data"] == [
            "../input_data.foo",
            mock_plugin_name,
        ]

        loaded_optimization_result = OptimizationResult.model_validate(
            serialized_optimization_result,
            context={"save_folder": save_folder},
        )
        assert loaded_optimization_result.input_data.equals(input_data)


def test_single_data():
    data_model = deepcopy(TestDataModelConstantIndexIndependent)
    experiment = ExperimentModel(datasets={"test_data": data_model})
    objective = OptimizationObjective(experiment)
    assert isinstance(objective._data, OptimizationData)

    penalty = objective.calculate()
    data_size = data_model.data["model_dim"].size * data_model.data["global_dim"].size
    assert penalty.size == data_size

    result = objective.get_result().optimization_results
    assert "test_data" in result
    result_data = result["test_data"]
    print(result_data)
    assert "test_ele" in result_data.elements
    element_result = result_data.elements["test_ele"]

    assert "concentrations" in element_result
    assert element_result.concentrations.shape == (
        data_model.data["model_dim"].size,
        1,
    )
    assert "amplitudes" in element_result
    assert element_result.amplitudes.shape == (
        data_model.data["global_dim"].size,
        1,
    )
    assert result_data.residuals is not None
    assert result_data.input_data is not None
    assert result_data.input_data.shape == data_model.data.data.shape
    assert result_data.residuals.shape == data_model.data.data.shape


@pytest.mark.parametrize("weight", {True, False})
def test_global_data(weight: bool):
    dataset_label = "dataset1"
    data_model = deepcopy(TestDataModelGlobal)
    if weight:
        data_model.data["weight"] = xr.ones_like(data_model.data.data) * 0.5
    experiment = ExperimentModel(datasets={dataset_label: data_model})
    objective = OptimizationObjective(experiment)
    assert isinstance(objective._data, OptimizationData)
    model_coord = data_model.data.coords["model_dim"]
    global_coord = data_model.data.coords["global_dim"]

    penalty = objective.calculate()
    data_size = model_coord.size * global_coord.size
    assert penalty.size == data_size

    result = objective.get_result().optimization_results
    assert dataset_label in result

    optimization_result = result[dataset_label]
    # TODO: something to figure out
    # when the full matrix is calculated, from the "elements" and "global_elements", the name
    # given to the results is the same as the dataset name
    element_result = optimization_result.elements[dataset_label]
    print(element_result)
    assert "model_concentrations" in element_result
    assert element_result["model_concentrations"].shape == (
        (global_coord.size, model_coord.size, 1) if weight else (model_coord.size, 1)
    )
    assert "global_concentrations" in element_result
    assert element_result["global_concentrations"].shape == (
        (model_coord.size, global_coord.size, 1) if weight else (global_coord.size, 1)
    )
    assert "amplitudes" in element_result
    assert element_result["amplitudes"].shape == (1, 1)
    assert optimization_result.residuals is not None
    assert optimization_result.residuals.shape == data_model.data.data.shape


def test_multiple_data():
    data_model_one = deepcopy(TestDataModelConstantIndexIndependent)
    data_model_two = deepcopy(TestDataModelConstantIndexDependent)
    experiment = ExperimentModel(
        datasets={
            "independent": data_model_one,
            "dependent": data_model_two,
        }
    )
    objective = OptimizationObjective(experiment)
    assert isinstance(objective._data, LinkedOptimizationData)

    model_coord_one = data_model_one.data.coords["model_dim"]
    model_coord_two = data_model_two.data.coords["model_dim"]
    global_coord_one = data_model_one.data.coords["global_dim"]
    global_coord_two = data_model_two.data.coords["global_dim"]

    penalty = objective.calculate()
    data_size_one = model_coord_one.size * global_coord_one.size
    data_size_two = model_coord_two.size * global_coord_two.size
    assert penalty.size == data_size_one + data_size_two

    result = objective.get_result().optimization_results

    assert "independent" in result
    optimization_result_independent = result["independent"]
    assert optimization_result_independent.residuals is not None
    assert optimization_result_independent.residuals.shape == data_model_one.data.data.shape

    element_result_independent = optimization_result_independent.elements["test_ele"]
    assert "concentrations" in element_result_independent
    assert element_result_independent["concentrations"].shape == (
        model_coord_one.size,
        1,
    )
    assert "amplitudes" in element_result_independent
    assert element_result_independent["amplitudes"].shape == (
        global_coord_one.size,
        1,
    )

    assert "dependent" in result
    optimization_result_dependent = result["dependent"]
    assert optimization_result_dependent.residuals is not None
    # this datamodel has transposed input
    assert optimization_result_dependent.residuals.shape == data_model_two.data.data.T.shape

    element_result_dependent = optimization_result_dependent.elements["test_ele_index_dependent"]
    assert "concentrations" in element_result_dependent
    assert element_result_dependent["concentrations"].shape == (
        global_coord_two.size,
        model_coord_two.size,
        1,
    )
    assert "amplitudes" in element_result_dependent
    assert element_result_dependent["amplitudes"].shape == (
        global_coord_two.size,
        1,
    )


@pytest.mark.parametrize("weight", {True, False})
def test_result_data(weight: bool):
    dataset_label = "dataset1"
    data_model = deepcopy(TestDataModelConstantIndexIndependent)
    if weight:
        data_model.data["weight"] = xr.ones_like(data_model.data.data) * 0.5
    experiment = ExperimentModel(datasets={dataset_label: data_model})
    objective = OptimizationObjective(experiment)
    assert isinstance(objective._data, OptimizationData)

    penalty = objective.calculate()
    data_size = (
        data_model.data.coords["model_dim"].size * data_model.data.coords["global_dim"].size
    )
    assert penalty.size == data_size

    result = objective.get_result().optimization_results
    assert dataset_label in result
    optimization_result = result[dataset_label]
    element_results = optimization_result.elements["test_ele"]
    assert "concentrations" in element_results
    assert "amplitudes" in element_results

    assert np.array_equal(data_model.data.coords["model_dim"], element_results.coords["model_dim"])
    assert np.array_equal(
        data_model.data.coords["global_dim"], element_results.coords["global_dim"]
    )
    assert optimization_result.input_data.shape == data_model.data.data.shape
    assert np.allclose(optimization_result.input_data, data_model.data.data)
    if weight:
        # TODO: find the lost weights
        # assert "weight" in result_data
        # assert "weighted_residual" in result_data
        pass


def test_penalty():
    data_model_one = deepcopy(TestDataModelConstantIndexIndependent)
    data_model_two = deepcopy(TestDataModelConstantIndexDependent)
    experiment = ExperimentModel(
        datasets={
            "independent": data_model_one,
            "dependent": data_model_two,
        },
        clp_link_tolerance=1,
        clp_penalties=[
            EqualAreaPenalty(type="equal_area", source="c1", target="c2", parameter=2, weight=4)
        ],
    )
    objective = OptimizationObjective(experiment)
    assert isinstance(objective._data, LinkedOptimizationData)

    penalty = objective.calculate()
    data_size_one = (
        data_model_one.data.coords["model_dim"].size
        * data_model_one.data.coords["global_dim"].size
    )
    data_size_two = (
        data_model_two.data.coords["model_dim"].size
        * data_model_two.data.coords["global_dim"].size
    )
    assert penalty.size == data_size_one + data_size_two + 1
    assert penalty[-1] == 20  # TODO: investigate


if __name__ == "__main__":
    pytest.main([__file__])
