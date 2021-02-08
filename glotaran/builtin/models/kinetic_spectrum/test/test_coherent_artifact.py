import numpy as np
import xarray as xr

from glotaran.analysis.optimize import optimize
from glotaran.analysis.scheme import Scheme
from glotaran.builtin.models.kinetic_spectrum import KineticSpectrumModel
from glotaran.builtin.models.kinetic_spectrum.kinetic_spectrum_matrix import (
    kinetic_spectrum_matrix,
)
from glotaran.parameter import ParameterGroup


def test_coherent_artifact():
    model_dict = {
        "initial_concentration": {
            "j1": {"compartments": ["s1"], "parameters": ["2"]},
        },
        "megacomplex": {
            "mc1": {"k_matrix": ["k1"]},
        },
        "k_matrix": {
            "k1": {
                "matrix": {
                    ("s1", "s1"): "1",
                }
            }
        },
        "irf": {
            "irf1": {
                "type": "gaussian-coherent-artifact",
                "center": "2",
                "width": "3",
                "coherent_artifact_order": 3,
            },
        },
        "dataset": {
            "dataset1": {
                "initial_concentration": "j1",
                "megacomplex": ["mc1"],
                "irf": "irf1",
            },
        },
    }
    model = KineticSpectrumModel.from_dict(model_dict.copy())

    parameters = ParameterGroup.from_list(
        [
            101e-4,
            [10, {"vary": False, "non-negative": False}],
            [20, {"vary": False, "non-negative": False}],
            [30, {"vary": False, "non-negative": False}],
        ]
    )

    time = np.asarray(np.arange(0, 50, 1.5))

    irf = model.irf["irf1"].fill(model, parameters)
    irf_same_width = irf.calculate_coherent_artifact(time)

    model_dict["irf"]["irf1"]["coherent_artifact_width"] = "4"
    model = KineticSpectrumModel.from_dict(model_dict)

    irf = model.irf["irf1"].fill(model, parameters)
    irf_diff_width = irf.calculate_coherent_artifact(time)

    assert np.array_equal(irf_same_width[0], irf_diff_width[0])  # labels the same
    assert not np.array_equal(irf_same_width[1], irf_diff_width[1])  # but content is not

    data = model.dataset["dataset1"].fill(model, parameters)
    compartments, matrix = kinetic_spectrum_matrix(data, time, 0)

    assert len(compartments) == 4
    for i in range(1, 4):
        assert compartments[i] == f"coherent_artifact_{i}"

    assert matrix.shape == (time.size, 4)

    clp = xr.DataArray(
        [[1, 1, 1, 1]],
        coords=[
            ("spectral", [0]),
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
    data = model.simulate("dataset1", parameters, axis, clp)

    dataset = {"dataset1": data}
    scheme = Scheme(
        model=model, parameters=parameters, data=dataset, maximum_number_function_evaluations=20
    )
    result = optimize(scheme)
    print(result.optimized_parameters)

    for label, param in result.optimized_parameters.all():
        assert np.allclose(param.value, parameters.get(label).value, rtol=1e-1)

    resultdata = result.data["dataset1"]
    assert np.array_equal(data.time, resultdata.time)
    assert np.array_equal(data.spectral, resultdata.spectral)
    assert data.data.shape == resultdata.data.shape
    assert data.data.shape == resultdata.fitted_data.shape
    assert np.allclose(data.data, resultdata.fitted_data, rtol=1e-2)

    assert "coherent_artifact_concentration" in resultdata
    assert resultdata["coherent_artifact_concentration"].shape == (time.size, 3)

    assert "coherent_artifact_associated_spectra" in resultdata
    assert resultdata["coherent_artifact_associated_spectra"].shape == (1, 3)
