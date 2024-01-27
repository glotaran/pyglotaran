from __future__ import annotations

import numpy as np
import pytest

from glotaran.builtin.items.activation import GaussianActivation
from glotaran.builtin.items.activation import MultiGaussianActivation
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
