from __future__ import annotations

import numpy as np
import xarray as xr

from glotaran.builtin.items.activation.activation import Activation
from glotaran.builtin.items.activation.gaussian import MultiGaussianActivation
from glotaran.builtin.items.activation.instant import InstantActivation  # noqa: F401
from glotaran.model import Attribute
from glotaran.model import DataModel
from glotaran.model import ItemIssue
from glotaran.parameter import Parameters


class NoActivationIssue(ItemIssue):
    def to_string(self) -> str:
        """Get the issue as string.

        Returns
        -------
        str
        """
        return "ActivationError: No Activation defined in dataset."


def validate_activations(
    value: list[Activation],
    activation: Activation,
    parameters: Parameters | None,
) -> list[ItemIssue]:
    issues = []
    if len(value) == 0:
        issues.append(NoActivationIssue())
    return issues


class ActivationDataModel(DataModel):
    activation: list[Activation.get_annotated_type()] = Attribute(
        validator=validate_activations,
        description="The activation(s) of the dataset.",
    )


def add_activation_to_result_data(model: ActivationDataModel, data: xr.Dataset):
    gaussian_activations = [a for a in model.activation if isinstance(a, MultiGaussianActivation)]
    if "gaussian_activation" in data or not len(gaussian_activations):
        return
    data.coords["gaussian_activation"] = np.arange(1, len(gaussian_activations) + 1)
    global_dimension = data.attrs["global_dimension"]
    global_axis = data.coords[global_dimension]
    model_dimension = data.attrs["model_dimension"]
    model_axis = data.coords[model_dimension]

    activations = []
    activation_parameters = []
    activation_shifts = []
    has_shifts = any(a.shift is not None for a in gaussian_activations)
    activation_dispersions = []
    has_dispersions = any(a.dispersion_center is not None for a in gaussian_activations)
    for activation in gaussian_activations:
        activations.append(activation.calculate_function(model_axis))
        activation_parameters.append(activation.parameters())
        if has_shifts:
            activation_shifts.append(
                activation.shift if activation.shift is not None else [0] * global_axis.size
            )
        if has_dispersions:
            activation_dispersions.append(
                activation.calculate_dispersion(global_axis)
                if activation.dispersion_center is not None
                else activation.center * global_axis.size
            )

    data["gaussian_activation_function"] = (
        ("gaussian_activation", model_dimension),
        activations,
    )
    data["gaussian_activation_center"] = (
        ("gaussian_activation", "gaussian_activation_part"),
        [[p.center for p in ps] for ps in activation_parameters],
    )
    data["gaussian_activation_width"] = (
        ("gaussian_activation", "gaussian_activation_part"),
        [[p.width for p in ps] for ps in activation_parameters],
    )
    data["gaussian_activation_scale"] = (
        ("gaussian_activation", "gaussian_activation_part"),
        [[p.scale for p in ps] for ps in activation_parameters],
    )
    if has_shifts:
        data["gaussian_activation_shift"] = (
            ("gaussian_activation", global_dimension),
            activation_shifts,
        )
    if has_dispersions:
        data["gaussian_activation_dispersion"] = (
            ("gaussian_activation", "gaussian_activation_part", global_dimension),
            activation_dispersions,
        )
