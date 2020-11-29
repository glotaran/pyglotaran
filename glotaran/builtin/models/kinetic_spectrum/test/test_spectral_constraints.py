import numpy as np
import xarray as xr

from glotaran.analysis.optimize import optimize
from glotaran.analysis.scheme import Scheme
from glotaran.builtin.models.kinetic_image.kinetic_image_matrix import kinetic_image_matrix
from glotaran.builtin.models.kinetic_spectrum import KineticSpectrumModel
from glotaran.builtin.models.kinetic_spectrum.spectral_constraints import (
    apply_spectral_constraints,
)
from glotaran.parameter import ParameterGroup


def test_spectral_constraint():
    model = KineticSpectrumModel.from_dict(
        {
            "initial_concentration": {
                "j1": {
                    "compartments": ["s1", "s2"],
                    "parameters": ["i.1", "i.2"],
                },
            },
            "megacomplex": {
                "mc1": {"k_matrix": ["k1"]},
            },
            "k_matrix": {
                "k1": {
                    "matrix": {
                        ("s2", "s1"): "kinetic.1",
                        ("s2", "s2"): "kinetic.2",
                    }
                }
            },
            "spectral_constraints": [
                {"type": "zero", "compartment": "s2", "interval": (float("-inf"), float("inf"))},
            ],
            "dataset": {
                "dataset1": {
                    "initial_concentration": "j1",
                    "megacomplex": ["mc1"],
                },
            },
        }
    )
    print(model)

    parameter = ParameterGroup.from_dict(
        {
            "kinetic": [1e-4, 1e-5],
            "i": [1, 2],
        }
    )

    time = np.asarray(np.arange(0, 50, 1.5))
    dataset = model.dataset["dataset1"].fill(model, parameter)
    compartments, matrix = kinetic_image_matrix(dataset, time, 0)

    assert len(compartments) == 2
    assert matrix.shape == (time.size, 2)

    reduced_compartments, reduced_matrix = apply_spectral_constraints(
        model, compartments, matrix, 1
    )

    print(reduced_matrix)
    assert len(reduced_compartments) == 1
    assert reduced_matrix.shape == (time.size, 1)

    reduced_compartments, reduced_matrix = model.constrain_matrix_function(
        parameter, compartments, matrix, 1
    )

    assert reduced_matrix.shape == (time.size, 1)

    clp = xr.DataArray(
        [[1.0, 10.0, 20.0, 1]], coords=(("spectral", [1]), ("clp_label", ["s1", "s2", "s3", "s4"]))
    )

    data = model.simulate(
        "dataset1", parameter, clp=clp, axes={"time": time, "spectral": np.array([1])}
    )

    dataset = {"dataset1": data}
    scheme = Scheme(model=model, parameter=parameter, data=dataset, nfev=20)
    result = optimize(scheme)

    result_data = result.data["dataset1"]
    print(result_data.clp_label)
    print(result_data.clp)
    #  TODO: save reduced clp
    #  assert result_data.clp.shape == (1, 1)

    print(result_data.species_associated_spectra)
    assert result_data.species_associated_spectra.shape == (1, 2)
    assert result_data.species_associated_spectra[0, 1] == 0
