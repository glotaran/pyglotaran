from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from glotaran.builtin.items.activation import GaussianActivation
from glotaran.builtin.items.activation import MultiGaussianActivation
from glotaran.builtin.items.activation.data_model import ActivationDataModel
from glotaran.model.errors import GlotaranUserError


def test_gaussian_activation():
    activation = GaussianActivation(
        type="gaussian",
        center=1,
        width=10,
        backsweep=2,
        compartments={},
    )
    parameter = activation.parameters(np.array([0]))
    assert len(parameter) == 1
    assert parameter[0].backsweep_period == 2
    assert parameter[0].center == 1
    assert parameter[0].width == 10


def test_multi_gaussian_activation():
    activation = MultiGaussianActivation(
        type="multi-gaussian",
        center=[1, 2],
        width=[10, 20],
        compartments={},
    )
    parameter = activation.parameters(np.array([0]))
    assert len(parameter) == 2
    assert parameter[0].backsweep_period == 0
    assert parameter[0].center == 1
    assert parameter[0].width == 10
    assert parameter[1].center == 2
    assert parameter[1].width == 20


def test_gaussian_activation_shift():
    activation = GaussianActivation(
        type="gaussian",
        center=1,
        width=10,
        shift=[0.5, 1],
        compartments={},
    )
    parameter = activation.parameters(np.array([0, 1]))
    assert len(parameter) == 2
    assert parameter[0][0].center == 0.5
    assert parameter[1][0].center == 0
    with pytest.raises(GlotaranUserError):
        activation.parameters(np.array([0, 1, 2]))


def test_gaussian_activation_dispersion():
    activation = GaussianActivation(
        type="gaussian",
        center=1,
        width=10,
        dispersion_center=100,
        center_dispersion_coefficients=[2],
        width_dispersion_coefficients=[2],
        compartments={},
    )
    parameter = activation.parameters(np.array([0, 100]))
    assert len(parameter) == 2
    assert parameter[0][0].center == -1
    assert parameter[0][0].width == 8
    assert parameter[1][0].center == 1
    assert parameter[1][0].width == 10


def test_gaussian_activation_dispersion_reciproke():
    activation = GaussianActivation(
        type="gaussian",
        center=1,
        width=10,
        dispersion_center=200,
        center_dispersion_coefficients=[2],
        width_dispersion_coefficients=[2],
        reciproke_global_axis=True,
        compartments={},
    )
    parameter = activation.parameters(np.array([100, 200]))
    assert len(parameter) == 2
    assert parameter[0][0].center == 11
    assert parameter[0][0].width == 20
    assert parameter[1][0].center == 1
    assert parameter[1][0].width == 10


def test_create_result_gaussian_activation():
    """Test create_result with GaussianActivation produces expected dimensions."""
    model = ActivationDataModel(
        elements=[],
        activations={
            "irf": GaussianActivation(
                type="gaussian",
                center=0.1,
                width=0.1,
                compartments={},
            )
        },
    )

    global_axis = np.linspace(400, 600, 5)
    model_axis = np.linspace(-1, 1, 20)

    amplitudes = xr.DataArray(
        np.zeros((5, 3)),
        coords={"wavelength": global_axis, "species": ["s1", "s2", "s3"]},
        dims=["wavelength", "species"],
    )

    concentrations = xr.DataArray(
        np.zeros((20, 3)),
        coords={"time": model_axis, "species": ["s1", "s2", "s3"]},
        dims=["time", "species"],
    )

    result = model.create_result(
        model=model,
        global_dimension="wavelength",
        model_dimension="time",
        amplitudes=amplitudes,
        concentrations=concentrations,
    )

    assert "irf" in result
    assert "trace" in result["irf"]
    assert "shift" in result["irf"]
    assert "center" in result["irf"]
    assert result["irf"]["trace"].dims == ("time",)
    assert result["irf"]["trace"].shape == (model_axis.size,)
    assert result["irf"]["shift"].dims == ("wavelength",)
    assert result["irf"]["shift"].shape == (global_axis.size,)
    assert result["irf"]["center"].dims == ("component_index", "wavelength")
    assert result["irf"]["center"].shape == (1, global_axis.size)
    assert np.allclose(result["irf"]["center"][0], 0.1)


def test_create_result_multi_gaussian_activation():
    """Test create_result with MultiGaussianActivation produces expected dimensions."""
    model = ActivationDataModel(
        elements=[],
        activations={
            "irf": MultiGaussianActivation(
                type="multi-gaussian",
                center=[0.1, 0.5],
                width=[0.1, 0.2],
                compartments={},
            )
        },
    )

    global_axis = np.linspace(400, 600, 5)
    model_axis = np.linspace(-1, 1, 20)

    amplitudes = xr.DataArray(
        np.zeros((5, 3)),
        coords={"wavelength": global_axis, "species": ["s1", "s2", "s3"]},
        dims=["wavelength", "species"],
    )

    concentrations = xr.DataArray(
        np.zeros((20, 3)),
        coords={"time": model_axis, "species": ["s1", "s2", "s3"]},
        dims=["time", "species"],
    )

    result = model.create_result(
        model=model,
        global_dimension="wavelength",
        model_dimension="time",
        amplitudes=amplitudes,
        concentrations=concentrations,
    )

    assert "irf" in result
    assert "trace" in result["irf"]
    assert "shift" in result["irf"]
    assert "center" in result["irf"]
    assert result["irf"]["trace"].dims == ("time",)
    assert result["irf"]["trace"].shape == (model_axis.size,)
    assert result["irf"]["shift"].dims == ("wavelength",)
    assert result["irf"]["shift"].shape == (global_axis.size,)
    assert result["irf"]["center"].dims == ("component_index", "wavelength")
    assert result["irf"]["center"].shape == (2, global_axis.size)
    assert np.allclose(result["irf"]["center"][0], 0.1)
    assert np.allclose(result["irf"]["center"][1], 0.5)
