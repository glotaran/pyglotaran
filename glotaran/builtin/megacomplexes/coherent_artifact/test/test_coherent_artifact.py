from typing import Any

import numpy as np
import pytest
import xarray as xr

from glotaran.builtin.megacomplexes.coherent_artifact import CoherentArtifactMegacomplex
from glotaran.builtin.megacomplexes.decay import DecayMegacomplex
from glotaran.model import Model
from glotaran.model import fill_item
from glotaran.optimization.matrix_provider import MatrixProvider
from glotaran.optimization.optimize import optimize
from glotaran.parameter import Parameters
from glotaran.project import Scheme
from glotaran.simulation import simulate


@pytest.mark.parametrize(
    "spectral_dependence",
    ("none", "dispersed", "shifted"),
)
def test_coherent_artifact(spectral_dependence: str):
    model_dict: dict[str, dict[str, Any]] = {
        "initial_concentration": {
            "j1": {"compartments": ["s1"], "parameters": ["irf_center"]},
        },
        "megacomplex": {
            "mc1": {"type": "decay", "k_matrix": ["k1"]},
            "mc2": {"type": "coherent-artifact", "order": 3},
        },
        "k_matrix": {
            "k1": {
                "matrix": {
                    ("s1", "s1"): "rate",
                }
            }
        },
        "irf": {
            "irf1": {
                "type": "multi-gaussian",
                "center": ["irf_center"],
                "width": ["irf_width"],
            },
        },
        "dataset": {
            "dataset1": {
                "initial_concentration": "j1",
                "megacomplex": ["mc1", "mc2"],
                "irf": "irf1",
            },
        },
    }

    parameter_list = [
        ["rate", 101e-4],
        ["irf_center", 10, {"vary": False, "non-negative": False}],
        ["irf_width", 20, {"vary": False, "non-negative": False}],
    ]

    irf_spec = model_dict["irf"]["irf1"]

    if spectral_dependence == "dispersed":
        irf_spec["type"] = "spectral-multi-gaussian"
        irf_spec["dispersion_center"] = "irf_dispc"
        irf_spec["center_dispersion_coefficients"] = ["irf_disp1", "irf_disp2"]

        parameter_list += [
            ["irf_dispc", 300, {"vary": False, "non-negative": False}],
            ["irf_disp1", 0.01, {"vary": False, "non-negative": False}],
            ["irf_disp2", 0.001, {"vary": False, "non-negative": False}],
        ]
    elif spectral_dependence == "shifted":
        irf_spec["shift"] = ["irf_shift1", "irf_shift2", "irf_shift3"]
        parameter_list += [
            ["irf_shift1", -2],
            ["irf_shift2", 0],
            ["irf_shift3", 2],
        ]

    model = Model.create_class_from_megacomplexes([DecayMegacomplex, CoherentArtifactMegacomplex])(
        **model_dict
    )

    parameters = Parameters.from_list(parameter_list)  # type: ignore[arg-type]

    time = np.arange(0, 50, 1.5)
    spectral = np.asarray([200, 300, 400])

    dataset_model = fill_item(model.dataset["dataset1"], model, parameters)
    matrix = MatrixProvider.calculate_dataset_matrix(dataset_model, spectral, time)
    compartments = matrix.clp_labels

    print(compartments)
    assert len(compartments) == 4
    for i in range(1, 4):
        assert compartments[i] == f"coherent_artifact_{i}_mc2"

    if spectral_dependence == "none":
        assert matrix.matrix.shape == (time.size, 4)
    else:
        assert matrix.matrix.shape == (spectral.size, time.size, 4)

    clp = xr.DataArray(
        np.ones((3, 4)),
        coords=[
            ("spectral", spectral),
            (
                "clp_label",
                [
                    "s1",
                    "coherent_artifact_1_mc2",
                    "coherent_artifact_2_mc2",
                    "coherent_artifact_3_mc2",
                ],
            ),
        ],
    )
    axis = {"time": time, "spectral": clp.spectral.data}
    data = simulate(model, "dataset1", parameters, axis, clp)

    dataset = {"dataset1": data}
    scheme = Scheme(
        model=model, parameters=parameters, data=dataset, maximum_number_function_evaluations=20
    )
    result = optimize(scheme)
    print(result.optimized_parameters)

    for param in result.optimized_parameters.all():
        assert np.allclose(param.value, parameters.get(param.label).value, rtol=1e-1)

    resultdata = result.data["dataset1"]
    assert np.array_equal(data.time, resultdata.time)
    assert np.array_equal(data.spectral, resultdata.spectral)
    assert data.data.shape == resultdata.data.shape
    assert data.data.shape == resultdata.fitted_data.shape
    assert np.allclose(data.data, resultdata.fitted_data)

    assert "coherent_artifact_response" in resultdata
    if spectral_dependence == "none":
        assert resultdata["coherent_artifact_response"].shape == (time.size, 3)
    else:
        assert resultdata["coherent_artifact_response"].shape == (spectral.size, time.size, 3)

    assert "coherent_artifact_associated_spectra" in resultdata
    assert resultdata["coherent_artifact_associated_spectra"].shape == (3, 3)


def test_two_coherent_artifacts():
    """Test 2 coherent artifacts with 2 different pure solvents."""
    model_dict: dict[str, dict[str, Any]] = {
        "megacomplex": {
            "ac1": {"type": "coherent-artifact", "order": 3},
            "ac2": {"type": "coherent-artifact", "order": 3},
        },
        "irf": {
            "irf1": {
                "type": "gaussian",
                "center": "irf_center_1",
                "width": "irf_width",
            },
            "irf2": {
                "type": "gaussian",
                "center": "irf_center_2",
                "width": "irf_width",
            },
        },
        "dataset": {
            "dataset1": {
                "megacomplex": ["ac1"],
                "irf": "irf1",
            },
            "dataset2": {
                "megacomplex": ["ac2"],
                "irf": "irf2",
            },
        },
    }

    parameter_list = [
        ["irf_center_1", 10, {"vary": False, "non-negative": False}],
        ["irf_center_2", 20, {"vary": False, "non-negative": False}],
        ["irf_width", 30, {"vary": False, "non-negative": False}],
    ]

    model = Model.create_class_from_megacomplexes([CoherentArtifactMegacomplex])(**model_dict)

    parameters = Parameters.from_list(parameter_list)  # type: ignore[arg-type]

    time = np.arange(0, 50, 1.5)
    spectral = np.asarray([200, 300, 400])

    dataset_model1 = fill_item(model.dataset["dataset1"], model, parameters)
    dataset_matrix1 = MatrixProvider.calculate_dataset_matrix(dataset_model1, spectral, time)

    assert dataset_matrix1.matrix.shape == (time.size, 3)

    dataset_model2 = fill_item(model.dataset["dataset2"], model, parameters)
    dataset_matrix2 = MatrixProvider.calculate_dataset_matrix(dataset_model2, spectral, time)

    assert len(dataset_matrix2.clp_labels) == 3

    assert dataset_matrix2.matrix.shape == (time.size, 3)

    assert dataset_matrix1.clp_labels != dataset_matrix2.clp_labels
    assert not np.array_equal(dataset_matrix1.matrix, dataset_matrix2.matrix)

    for i in range(0, 3):
        assert dataset_matrix1.clp_labels[i] == f"coherent_artifact_{i+1}_ac1"

    for i in range(0, 3):
        assert dataset_matrix2.clp_labels[i] == f"coherent_artifact_{i+1}_ac2"
