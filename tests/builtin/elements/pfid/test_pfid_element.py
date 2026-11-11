from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from glotaran.builtin.elements.kinetic import KineticElement
from glotaran.builtin.elements.pfid import PFIDDataModel
from glotaran.builtin.elements.pfid import PFIDElement
from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.builtin.items.activation import InstantActivation
from glotaran.builtin.items.activation import MultiGaussianActivation
from glotaran.model.data_model import DataModel
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization import Optimization
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.parameter import Parameters
from glotaran.project.library import ModelLibrary
from glotaran.simulation import simulate


def create_activation(*, shift: list[float] | None = None) -> MultiGaussianActivation:
    return MultiGaussianActivation(
        type="multi-gaussian",
        compartments={},
        center=[0.01],
        width=[0.05],
        shift=shift,
    )


def create_element(*, frequency: float | str = 1500, rate: float | str = -2) -> PFIDElement:
    return PFIDElement(
        label="pfid",
        type="pfid",
        oscillations={"osc": {"frequency": frequency, "rate": rate}},
    )


def test_pfid_plugin_and_data_model_configuration():
    library = ModelLibrary.from_dict(
        {
            "pfid": {
                "type": "pfid",
                "oscillations": {"osc": {"frequency": 1500, "rate": -2}},
            }
        }
    )
    model = DataModel.from_dict(
        library,
        {
            "elements": ["pfid"],
            "activation": {
                "type": "multi-gaussian",
                "compartments": {},
                "center": [0.01],
                "width": [0.05],
            },
        },
    )

    assert isinstance(library["pfid"], PFIDElement)
    assert isinstance(model, PFIDDataModel)

    with pytest.raises(ValidationError):
        DataModel.from_dict(library, {"elements": ["pfid"]})


def test_pfid_matrix_is_index_dependent_and_supports_shift():
    element = create_element()
    global_axis = np.array([1490.0, 1510.0])
    model_axis = np.linspace(-0.2, 1, 25)
    model = PFIDDataModel(elements=[element], activation=create_activation(shift=[0.0, 0.1]))

    labels, matrix = element.calculate_matrix(model, global_axis, model_axis)

    assert labels == ["osc_cos", "osc_sin"]
    assert matrix.shape == (2, 25, 2)
    assert np.all(np.isfinite(matrix))
    assert np.any(matrix != 0)
    assert not np.array_equal(matrix[0], matrix[1])


@pytest.mark.parametrize(
    ("model_options", "equivalent_frequency"),
    [
        ({"spectral_axis_scale": 2}, 3000),
        (
            {"spectral_axis_inverted": True, "spectral_axis_scale": 3e6},
            2000,
        ),
    ],
)
def test_pfid_spectral_axis_transform(model_options: dict, equivalent_frequency: float):
    global_axis = np.array([1490.0, 1510.0])
    model_axis = np.linspace(-0.2, 1, 25)
    transformed = create_element()
    reference = create_element(frequency=equivalent_frequency)

    _, transformed_matrix = transformed.calculate_matrix(
        PFIDDataModel(elements=[transformed], activation=create_activation(), **model_options),
        global_axis,
        model_axis,
    )
    _, reference_matrix = reference.calculate_matrix(
        PFIDDataModel(elements=[reference], activation=create_activation()),
        global_axis,
        model_axis,
    )

    assert np.allclose(transformed_matrix, reference_matrix)


def test_pfid_result_contains_labeled_decompositions():
    element = create_element()
    model = PFIDDataModel(elements=[element], activation=create_activation())
    amplitudes = xr.DataArray(
        [[3.0, 4.0], [4.0, 3.0]],
        coords={"spectral": [1490, 1510], "amplitude_label": ["osc_cos", "osc_sin"]},
    )
    concentrations = xr.DataArray(
        np.ones((2, 3, 2)),
        coords={
            "spectral": [1490, 1510],
            "time": [-0.1, 0.0, 0.1],
            "amplitude_label": ["osc_cos", "osc_sin"],
        },
    )

    result = element.create_result(model, "spectral", "time", amplitudes, concentrations)

    assert set(result.data_vars) == {
        "amplitudes",
        "phase",
        "sin_amplitudes",
        "cos_amplitudes",
        "concentrations",
        "phase_concentrations",
        "sin_concentrations",
        "cos_concentrations",
    }
    assert result.oscillation.to_numpy().tolist() == ["osc"]
    assert np.array_equal(result.amplitudes, np.full((2, 1), 5.0))


def test_pfid_combines_with_kinetic_matrix_without_reordering():
    pfid = create_element()
    kinetic = KineticElement(
        label="kinetic",
        type="kinetic",
        rates={("s2", "s1"): 1.0},
    )
    kinetic_model = ActivationDataModel(
        elements=[kinetic],
        activations={
            "initial": InstantActivation(type="instant", compartments={"s1": 1.0, "s2": 0.0})
        },
    )
    global_axis = np.array([1490.0, 1510.0])
    model_axis = np.linspace(-0.2, 1, 25)
    pfid_matrix = OptimizationMatrix.from_element(
        None,
        pfid,
        PFIDDataModel(elements=[pfid], activation=create_activation()),
        global_axis,
        model_axis,
    )
    kinetic_matrix = OptimizationMatrix.from_element(
        None, kinetic, kinetic_model, global_axis, model_axis
    )

    combined = OptimizationMatrix.combine([kinetic_matrix, pfid_matrix])

    assert combined.clp_axis == ["s1", "s2", "osc_cos", "osc_sin"]
    assert np.array_equal(
        combined.array[..., combined.clp_axis.index("osc_cos")],
        pfid_matrix.array[..., pfid_matrix.clp_axis.index("osc_cos")],
    )


def test_pfid_simulation_and_optimization():
    element = create_element(frequency="osc.frequency", rate="osc.rate")
    library = {"pfid": element}
    wanted = Parameters.from_dict({"osc": [["frequency", 1500.0], ["rate", -2.0]]})
    initial = Parameters.from_dict({"osc": [["frequency", 1501.0], ["rate", -2.1]]})
    model = PFIDDataModel(
        elements=["pfid"],
        activation=create_activation(),
    )
    spectral = np.arange(1485.0, 1516.0, 1.0)
    time = np.linspace(-0.2, 1.5, 100)
    clp = xr.DataArray(
        np.column_stack(
            (
                np.exp(-(((spectral - 1498) / 5) ** 2)),
                0.5 * np.exp(-(((spectral - 1504) / 7) ** 2)),
            )
        ),
        coords={"spectral": spectral, "clp_label": ["osc_cos", "osc_sin"]},
    )
    model.data = simulate(
        model,
        library,
        wanted,
        {"time": time, "spectral": spectral},
        clp=clp,
    )

    optimized, results, info = Optimization(
        models=[ExperimentModel(datasets={"dataset": model})],
        parameters=initial,
        library=library,
        raise_exception=True,
        maximum_number_function_evaluations=30,
    ).run()

    assert info.success
    assert np.isclose(optimized.get("osc.frequency").value, 1500, rtol=1e-3)
    assert np.isclose(optimized.get("osc.rate").value, -2, rtol=1e-2)
    assert "pfid" in results["dataset"].elements
    assert np.allclose(
        results["dataset"].input_data - results["dataset"].residuals,
        model.data.data,
        atol=1e-5,
    )
