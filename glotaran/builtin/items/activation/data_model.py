from __future__ import annotations

from dataclasses import asdict
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
    value: dict[str, Activation],
    activation: Activation,
    parameters: Parameters | None,
) -> list[ItemIssue]:
    issues: list[ItemIssue] = []
    if len(value) == 0:
        issues.append(NoActivationIssue())
    return issues


class ActivationDataModel(DataModel):
    activations: dict[str, Activation.get_annotated_type()] = Attribute(  # type:ignore[valid-type]
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
    ) -> dict[str, xr.Dataset]:
        gaussian_activations = {
            key: a
            for key, a in model.activations.items()
            if isinstance(a, MultiGaussianActivation)
        }
        if not len(gaussian_activations):
            return {}

        global_axis = amplitudes.coords[global_dimension]
        model_axis = concentrations.coords[model_dimension]

        result: dict[str, xr.Dataset] = {}

        for key, activation in gaussian_activations.items():
            trace = activation.calculate_function(model_axis)
            shift = activation.shift if activation.shift is not None else [0] * global_axis.size
            center = (
                np.sum(activation.calculate_dispersion(global_axis), axis=0)
                if activation.dispersion_center is not None
                else activation.center * global_axis.size
            )
            # Since we don't pass the ``global_axis`` the type ambiguity is resolved
            props = [
                asdict(p)
                for p in cast(list[GaussianActivationParameters], activation.parameters())
            ]
            result[key] = xr.Dataset(
                {
                    "trace": xr.DataArray(
                        trace, coords={model_dimension: model_axis}, dims=(model_dimension,)
                    ),
                    "shift": xr.DataArray(
                        shift, coords={global_dimension: global_axis}, dims=(global_dimension,)
                    ),
                    "center": xr.DataArray(
                        center, coords={global_dimension: global_axis}, dims=(global_dimension,)
                    ),
                },
                attrs={"activation": props},
            )

        return result
