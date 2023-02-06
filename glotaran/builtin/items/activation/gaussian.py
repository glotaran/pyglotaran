from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from glotaran.builtin.items.activation.activation import Activation
from glotaran.model import Attribute
from glotaran.model import GlotaranUserError
from glotaran.model import ItemIssue
from glotaran.model import Library
from glotaran.model import ParameterType
from glotaran.parameter import Parameters


class DispersionIssue(ItemIssue):
    def to_string(self) -> str:
        """Get the issue as string.

        Returns
        -------
        str
        """
        return "ActivationError: No dispersion coefficients defined."


class MultiGaussianIssue(ItemIssue):
    def __init__(self, centers: int, widths: int):
        self._centers = centers
        self._widths = widths

    def to_string(self) -> str:
        """Get the issue as string.

        Returns
        -------
        str
        """
        return (
            f"ActivationError: The size of the centers({self._centers}) "
            f"does not match the size of the width({self._widths})"
        )


def validate_multi_gaussian(
    value: list[ParameterType],
    activation: MultiGaussianActivation,
    library: Library,
    parameters: Parameters | None,
) -> list[ItemIssue]:
    issues = []
    if not isinstance(value, list):
        value = [value]
    len_centers = len(value)
    len_widths = len(activation.width) if isinstance(activation.width, list) else 1
    if len_centers - len_widths != 0 and len_centers != 1 and len_widths != 1:
        issues.append(MultiGaussianIssue(len_centers, len_widths))
    return issues


def validate_dispersion(
    value: ParameterType | None,
    activation: MultiGaussianActivation,
    library: Library,
    parameters: Parameters | None,
) -> list[ItemIssue]:
    issues = []
    if value is not None:
        len_centers = len(activation.center_dispersion_coefficients)
        len_widths = len(activation.width_dispersion_coefficients)
        if len_centers + len_widths == 0:
            issues.append(DispersionIssue())
    return issues


@dataclass
class GaussianActivationParameters:

    center: float
    width: float
    scale: float
    backsweep: bool
    backsweep_period: float

    def shift(self, value: float):
        self.center -= value

    def disperse(
        self,
        index: float,
        center: float,
        center_coefficients: list[float],
        width_coefficients: list[float],
        reciproke_global_axis: bool,
    ):
        distance = (
            (1e3 / index - 1e3 / center) if reciproke_global_axis else (index - center) / 100
        )
        for i, coefficient in enumerate(center_coefficients):
            self.center += coefficient * np.power(distance, i + 1)
        for i, coefficient in enumerate(width_coefficients):
            self.width += coefficient * np.power(distance, i + 1)


class MultiGaussianActivation(Activation):
    type: Literal["multi-gaussian"]

    center: list[ParameterType] = Attribute(
        validator=validate_multi_gaussian,  # type:ignore[arg-type]
        description="The center of the gaussian",
    )

    width: list[ParameterType] = Attribute(description="The width of the gaussian.")
    scale: list[ParameterType] | None = Attribute(
        default=None, description="The scales of the gaussians."
    )
    shift: list[ParameterType] | None = Attribute(
        default=None,
        description=(
            "A list parameters which gets subtracted from the centers along the global axis."
        ),
    )

    normalize: bool = Attribute(default=True, description="Whether to normalize the gaussians.")

    backsweep: ParameterType | None = Attribute(
        default=None, description="The period of the backsweep in a streak experiment."
    )
    dispersion_center: ParameterType | None = Attribute(
        default=None, validator=validate_dispersion, description="The center of the dispersion."
    )
    center_dispersion_coefficients: list[ParameterType] = Attribute(
        factory=list, description="The center coefficients of the dispersion."
    )
    width_dispersion_coefficients: list[ParameterType] = Attribute(
        factory=list, description="The width coefficients of the dispersion."
    )
    reciproke_global_axis: bool = Attribute(
        default=False,
        description="Set `True` if the global axis is reciproke (e.g. for wavennumbers),",
    )

    def is_index_dependent(self) -> bool:
        return self.shift is not None or self.dispersion_center is not None

    def parameters(
        self, global_axis: ArrayLike
    ) -> list[GaussianActivationParameters | list[GaussianActivationParameters]]:
        centers = self.center if isinstance(self.center, list) else [self.center]
        widths = self.width if isinstance(self.width, list) else [self.width]

        len_centers = len(centers)
        len_widths = len(widths)
        nr_gaussians = max(len_centers, len_widths)
        if len_centers != len_widths:
            if len_centers == 1:
                centers = centers * nr_gaussians
            else:
                widths = widths * nr_gaussians

        scales = self.scale or [1.0] * nr_gaussians
        backsweep = self.backsweep is not None
        backsweep_period = self.backsweep if backsweep else 0

        parameters = [
            GaussianActivationParameters(
                float(center), float(width), float(scale), backsweep, backsweep_period
            )
            for center, width, scale in zip(centers, widths, scales)
        ]

        if self.shift is None and self.dispersion_center is None:
            return parameters

        parameters = [[replace(p) for p in parameters] for _ in global_axis]

        if self.shift is not None:
            if global_axis.size != len(self.shift):
                raise GlotaranUserError(
                    f"The number of shifts({len(self.shift)}) does not match "
                    f"the size of the global axis({global_axis.size})."
                )
            for ps, shift in zip(parameters, self.shift):
                for p in ps:
                    p.shift(shift)

        if self.dispersion_center is not None:
            for ps, index in zip(parameters, global_axis):
                for p in ps:
                    p.disperse(
                        index,
                        self.dispersion_center,
                        self.center_dispersion_coefficients,
                        self.width_dispersion_coefficients,
                        self.reciproke_global_axis,
                    )

        return parameters


class GaussianActivation(MultiGaussianActivation):
    type: Literal["gaussian"]
    center: ParameterType
    width: ParameterType
