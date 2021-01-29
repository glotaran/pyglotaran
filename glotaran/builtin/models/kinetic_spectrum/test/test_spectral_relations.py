import numpy as np
import xarray as xr

from glotaran.analysis.optimize import optimize
from glotaran.analysis.scheme import Scheme
from glotaran.builtin.models.kinetic_image.kinetic_image_matrix import kinetic_image_matrix
from glotaran.builtin.models.kinetic_spectrum import KineticSpectrumModel
from glotaran.builtin.models.kinetic_spectrum.spectral_relations import (
    create_spectral_relation_matrix,
)
from glotaran.parameter import ParameterGroup


def test_spectral_relation():
    model = KineticSpectrumModel.from_dict(
        {
            "initial_concentration": {
                "j1": {
                    "compartments": ["s1", "s2", "s3", "s4"],
                    "parameters": ["i.1", "i.2", "i.3", "i.4"],
                },
            },
            "megacomplex": {
                "mc1": {"k_matrix": ["k1"]},
            },
            "k_matrix": {
                "k1": {
                    "matrix": {
                        ("s1", "s1"): "kinetic.1",
                        ("s2", "s2"): "kinetic.1",
                        ("s3", "s3"): "kinetic.1",
                        ("s4", "s4"): "kinetic.1",
                    }
                }
            },
            "spectral_relations": [
                {
                    "compartment": "s1",
                    "target": "s2",
                    "parameter": "rel.1",
                    "interval": [(0, 2)],
                },
                {
                    "compartment": "s1",
                    "target": "s3",
                    "parameter": "rel.2",
                    "interval": [(0, 2)],
                },
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

    rel1, rel2 = 10, 20
    parameters = ParameterGroup.from_dict(
        {
            "kinetic": [1e-4],
            "i": [1, 2, 3, 4],
            "rel": [rel1, rel2],
        }
    )

    time = np.asarray(np.arange(0, 50, 1.5))
    dataset = model.dataset["dataset1"].fill(model, parameters)
    compartments, matrix = kinetic_image_matrix(dataset, time, 0)

    assert len(compartments) == 4
    assert matrix.shape == (time.size, 4)

    reduced_compartments, relation_matrix = create_spectral_relation_matrix(
        model, "dataset1", parameters, compartments, matrix, 1
    )

    print(relation_matrix)
    assert len(reduced_compartments) == 2
    assert relation_matrix.shape == (4, 2)
    assert np.array_equal(
        relation_matrix,
        [
            [1.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [0.0, 1.0],
        ],
    )

    reduced_compartments, reduced_matrix = model.constrain_matrix_function(
        "dataset1", parameters, compartments, matrix, 1
    )

    assert reduced_matrix.shape == (time.size, 2)

    print(reduced_matrix[0, 0], matrix[0, 0], matrix[0, 1], matrix[0, 2])
    assert np.allclose(
        reduced_matrix[:, 0], matrix[:, 0] + rel1 * matrix[:, 1] + rel2 * matrix[:, 2]
    )

    clp = xr.DataArray(
        [[1.0, 10.0, 20.0, 1]], coords=(("spectral", [1]), ("clp_label", ["s1", "s2", "s3", "s4"]))
    )

    data = model.simulate(
        "dataset1", parameters, clp=clp, axes={"time": time, "spectral": np.array([1])}
    )

    dataset = {"dataset1": data}
    scheme = Scheme(
        model=model, parameters=parameters, data=dataset, maximum_number_function_evaluations=20
    )
    result = optimize(scheme)

    for label, param in result.optimized_parameters.all():
        if param.vary:
            assert np.allclose(param.value, parameters.get(label).value, rtol=1e-1)

    result_data = result.data["dataset1"]
    print(result_data.species_associated_spectra)
    assert result_data.species_associated_spectra.shape == (1, 4)
    assert (
        result_data.species_associated_spectra[0, 1]
        == rel1 * result_data.species_associated_spectra[0, 0]
    )
    assert np.allclose(
        result_data.species_associated_spectra[0, 2].values,
        rel2 * result_data.species_associated_spectra[0, 0].values,
    )


if __name__ == "__main__":
    test_spectral_relation()
