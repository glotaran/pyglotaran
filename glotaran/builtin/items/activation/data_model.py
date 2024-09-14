from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import numpy as np
import xarray as xr

from glotaran.builtin.items.activation.activation import Activation  # noqa: TCH001
from glotaran.builtin.items.activation.gaussian import GaussianActivationParameters
from glotaran.builtin.items.activation.gaussian import MultiGaussianActivation
from glotaran.builtin.items.activation.instant import InstantActivation  # noqa: F401
from glotaran.model.data_model import DataModel
from glotaran.model.errors import ItemIssue
from glotaran.model.item import Attribute

if TYPE_CHECKING:
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
    issues: list[ItemIssue] = []
    if len(value) == 0:
        issues.append(NoActivationIssue())
    return issues


class ActivationDataModel(DataModel):
    activation: list[Activation.get_annotated_type()] = Attribute(  # type:ignore[valid-type]
        validator=validate_activations,
        description="The activation(s) of the dataset.",
    )

    @staticmethod
    def create_result(
        model: ActivationDataModel,  # type:ignore[override]
        global_dimension: str,
        model_dimension: str,
        amplitudes: xr.DataArray,
        concentrations: xr.DataArray,
    ) -> dict[str, xr.DataArray]:
        gaussian_activations = [
            a for a in model.activation if isinstance(a, MultiGaussianActivation)
        ]
        if not len(gaussian_activations):
            return {}

        global_axis = amplitudes.coords[global_dimension]
        model_axis = concentrations.coords[model_dimension]

        activations = []
        activation_parameters: list[list[GaussianActivationParameters]] = []
        activation_shifts = []
        activation_dispersions = []

        has_shifts = any(a.shift is not None for a in gaussian_activations)
        has_dispersions = any(a.dispersion_center is not None for a in gaussian_activations)

        for activation in gaussian_activations:
            activations.append(activation.calculate_function(model_axis))
            activation_parameters.append(
                cast(list[GaussianActivationParameters], activation.parameters())
            )
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

        result = {}

        activation_coords = {"gaussian_activation": np.arange(1, len(gaussian_activations) + 1)}
        result["gaussian_activation_function"] = xr.DataArray(
            activations,
            coords=activation_coords | {model_dimension: model_axis},
            dims=("gaussian_activation", model_dimension),
        )

        if has_shifts:
            result["activation_shift"] = xr.DataArray(
                activation_shifts,
                coords=activation_coords | {global_dimension: global_axis},
                dims=("gaussian_activation", global_dimension),
            )

        activation_coords = activation_coords | {
            "gaussian_activation_part": np.arange(max([len(ps) for ps in activation_parameters]))
        }

        result["activation_center"] = xr.DataArray(
            [[p.center for p in ps] for ps in activation_parameters],
            coords=activation_coords,
            dims=("gaussian_activation", "gaussian_activation_part"),
        )
        result["activation_width"] = xr.DataArray(
            [[p.width for p in ps] for ps in activation_parameters],
            coords=activation_coords,
            dims=("gaussian_activation", "gaussian_activation_part"),
        )
        result["activation_scale"] = xr.DataArray(
            [[p.scale for p in ps] for ps in activation_parameters],
            coords=activation_coords,
            dims=("gaussian_activation", "gaussian_activation_part"),
        )

        if has_dispersions:
            result["activation_dispersion"] = xr.DataArray(
                activation_dispersions,
                coords=activation_coords | {global_dimension: global_axis},
                dims=("gaussian_activation", "gaussian_activation_part", global_dimension),
            )

        return result
