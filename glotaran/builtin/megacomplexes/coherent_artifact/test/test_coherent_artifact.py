import numpy as np
import pytest
import xarray as xr

from glotaran.analysis.optimize import optimize
from glotaran.analysis.simulation import simulate
from glotaran.analysis.util import calculate_matrix
from glotaran.builtin.megacomplexes.coherent_artifact import CoherentArtifactMegacomplex
from glotaran.builtin.megacomplexes.decay import DecayMegacomplex
from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.project import Scheme


@pytest.mark.parametrize(
    "is_index_dependent",
    (True, False),
)
def test_coherent_artifact(is_index_dependent: bool):
    model_dict = {
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
                "type": "spectral-multi-gaussian",
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

    if is_index_dependent:
        irf_spec = model_dict["irf"]["irf1"]
        irf_spec["dispersion_center"] = "irf_dispc"
        irf_spec["center_dispersion"] = ["irf_disp1", "irf_disp2"]

        parameter_list += [
            ["irf_dispc", 300, {"vary": False, "non-negative": False}],
            ["irf_disp1", 0.01, {"vary": False, "non-negative": False}],
            ["irf_disp2", 0.001, {"vary": False, "non-negative": False}],
        ]

    model = Model.from_dict(
        model_dict.copy(),
        megacomplex_types={
            "decay": DecayMegacomplex,
            "coherent-artifact": CoherentArtifactMegacomplex,
        },
    )

    parameters = ParameterGroup.from_list(parameter_list)

    time = np.arange(0, 50, 1.5)
    spectral = np.asarray([200, 300, 400])
    coords = {"time": time, "spectral": spectral}

    dataset_model = model.dataset["dataset1"].fill(model, parameters)
    dataset_model.overwrite_global_dimension("spectral")
    dataset_model.set_coordinates(coords)
    matrix = calculate_matrix(dataset_model, {"spectral": [0, 1, 2]})
    compartments = matrix.clp_labels

    print(compartments)
    assert len(compartments) == 4
    for i in range(1, 4):
        assert compartments[i] == f"coherent_artifact_{i}"

    assert matrix.matrix.shape == (time.size, 4)

    clp = xr.DataArray(
        np.ones((3, 4)),
        coords=[
            ("spectral", spectral),
            (
                "clp_label",
                [
                    "s1",
                    "coherent_artifact_1",
                    "coherent_artifact_2",
                    "coherent_artifact_3",
                ],
            ),
        ],
    )
    axis = {"time": time, "spectral": clp.spectral}
    data = simulate(model, "dataset1", parameters, axis, clp)

    dataset = {"dataset1": data}
    scheme = Scheme(
        model=model, parameters=parameters, data=dataset, maximum_number_function_evaluations=20
    )
    result = optimize(scheme)
    print(result.optimized_parameters)

    for label, param in result.optimized_parameters.all():
        assert np.allclose(param.value, parameters.get(label).value, rtol=1e-8)

    resultdata = result.data["dataset1"]
    assert np.array_equal(data.time, resultdata.time)
    assert np.array_equal(data.spectral, resultdata.spectral)
    assert data.data.shape == resultdata.data.shape
    assert data.data.shape == resultdata.fitted_data.shape
    assert np.allclose(data.data, resultdata.fitted_data)

    assert "coherent_artifact_response" in resultdata
    if is_index_dependent:
        assert resultdata["coherent_artifact_response"].shape == (spectral.size, time.size, 3)
    else:
        assert resultdata["coherent_artifact_response"].shape == (time.size, 3)

    assert "coherent_artifact_associated_spectra" in resultdata
    assert resultdata["coherent_artifact_associated_spectra"].shape == (3, 3)
