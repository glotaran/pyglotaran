import numpy as np
import pytest
import xarray as xr

from glotaran.builtin.elements.spectral import SpectralDataModel
from glotaran.builtin.elements.spectral import SpectralElement
from glotaran.model import ExperimentModel
from glotaran.optimization import Optimization
from glotaran.parameter import Parameters
from glotaran.simulation import simulate

test_library = {
    "gaussian": SpectralElement(
        **{
            "label": "gaussian",
            "type": "spectral",
            "shapes": {
                "s1": {
                    "type": "gaussian",
                    "amplitude": "shape.amplitude",
                    "location": "shape.location",
                    "width": "shape.width",
                }
            },
        }
    ),
    "skewed_gaussian_neg": SpectralElement(
        **{
            "label": "skewed_gaussian_neg",
            "type": "spectral",
            "shapes": {
                "s1": {
                    "type": "skewed-gaussian",
                    "location": "shape.location",
                    "width": "shape.width",
                    "skewness": -1,
                }
            },
        }
    ),
    "skewed_gaussian_pos": SpectralElement(
        **{
            "label": "skewed_gaussian_pos",
            "type": "spectral",
            "shapes": {
                "s1": {
                    "type": "skewed-gaussian",
                    "location": "shape.location",
                    "width": "shape.width",
                    "skewness": 1,
                }
            },
        }
    ),
    "skewed_gaussian_zero": SpectralElement(
        **{
            "label": "skewed_gaussian_zero",
            "type": "spectral",
            "shapes": {
                "s1": {
                    "type": "skewed-gaussian",
                    "location": "shape.location",
                    "width": "shape.width",
                    "skewness": 0,
                }
            },
        }
    ),
}


test_parameters_simulation = Parameters.from_dict(
    {
        "shape": [
            ["amplitude", 50],
            ["location", 50],
            ["width", 20],
        ],
    }
)
test_parameters = Parameters.from_dict(
    {
        "shape": [
            ["amplitude", 50, {"vary": False}],
            ["location", 51],
            ["width", 21],
        ],
    }
)

test_global_axis = np.array([0])
test_model_axis = np.arange(0, 100, 1)
test_axies = {"time": test_global_axis, "spectral": test_model_axis}
test_clp = xr.DataArray(
    [[1]],
    coords=[
        ("clp_label", ["s1"]),
        ("time", test_global_axis.data),
    ],
).T


@pytest.mark.parametrize(
    "shape",
    (
        "gaussian",
        "skewed_gaussian_neg",
        "skewed_gaussian_pos",
        "skewed_gaussian_zero",
    ),
)
def test_spectral(shape: str):
    data_model = SpectralDataModel(elements=[shape])
    data_model.data = simulate(
        data_model, test_library, test_parameters_simulation, test_axies, clp=test_clp
    )
    experiments = [
        ExperimentModel(
            datasets={"spectral": data_model},
        )
    ]
    optimization = Optimization(
        experiments,
        test_parameters,
        test_library,
        raise_exception=True,
        maximum_number_function_evaluations=25,
    )
    optimized_parameters, optimized_data, result = optimization.run()
    assert result.success
    print(test_parameters_simulation)
    print(optimized_parameters)
    assert optimized_parameters.close_or_equal(test_parameters_simulation)

    assert "spectral" in optimized_data
    assert "spectrum" in optimized_data["spectral"]
    assert "spectrum_associated_estimation" in optimized_data["spectral"]
    assert "spectra" in optimized_data["spectral"]
