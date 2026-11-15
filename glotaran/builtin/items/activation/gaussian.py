from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import Literal
from typing import overload

import numpy as np

from glotaran.builtin.items.activation.activation import Activation
from glotaran.model.errors import GlotaranUserError
from glotaran.model.errors import ItemIssue
from glotaran.model.item import Attribute
from glotaran.model.item import ParameterType

if TYPE_CHECKING:
    from glotaran.parameter import Parameters
    from glotaran.typing.types import ArrayLike


class DispersionIssue(ItemIssue):
    def to_string(self) -> str:
        """Get the issue as string.

        Returns
        -------
        str
        """
        return "ActivationError: No dispersion coefficients defined."


class MultiGaussianIssue(ItemIssue):
    def __init__(self, centers: int, widths: int) -> None:
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
    parameters: Parameters | None,  # noqa: ARG001
) -> list[ItemIssue]:
    issues: list[ItemIssue] = []
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
    parameters: Parameters | None,  # noqa: ARG001
) -> list[ItemIssue]:
    issues: list[ItemIssue] = []
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
    backsweep_period: float

    def shift(self, value: float) -> None:
        self.center -= value

    def disperse(
        self,
        index: float,
        center: float,
        center_coefficients: list[float],
        width_coefficients: list[float],
        *,
        reciproke_global_axis: bool,
    ) -> None:
        distance = (
            (1e3 / index - 1e3 / center) if reciproke_global_axis else (index - center) / 100
        )
        for i, coefficient in enumerate(center_coefficients):
            self.center += coefficient * np.power(distance, i + 1)
        for i, coefficient in enumerate(width_coefficients):
            self.width += coefficient * np.power(distance, i + 1)


class MultiGaussianActivation(Activation):
    """Excitation through an instrument response function that is a sum of Gaussians."""

    type: Literal["multi-gaussian"]  # type:ignore[assignment]

    center: list[ParameterType] = Attribute(
        validator=validate_multi_gaussian,
        description="The centers of the Gaussians on the model dimension.",
    )

    width: list[ParameterType] = Attribute(
        description="The widths of the Gaussians, as standard deviations."
    )
    scale: list[ParameterType] | None = Attribute(
        default=None,
        description="The scales of the Gaussians. Without it every Gaussian has the scale 1.",
    )
    shift: list[ParameterType] | None = Attribute(
        default=None,
        description=(
            "One value for each point of the global dimension, subtracted from the centers."
        ),
    )

    normalize: bool = Attribute(default=True, description="Whether to normalize the Gaussians.")

    backsweep: ParameterType | None = Attribute(
        default=None, description="The period of the backsweep in a streak experiment."
    )
    dispersion_center: ParameterType | None = Attribute(
        default=None,
        validator=validate_dispersion,
        description=(
            "The value of the global dimension at which the dispersion is zero. "
            "Setting it turns the dispersion on."
        ),
    )
    center_dispersion_coefficients: list[ParameterType] = Attribute(
        factory=list,
        description=(
            "The coefficients of the polynomial that moves the centers along the global "
            "dimension. Coefficient i, counted from 1, multiplies "
            "((value - dispersion_center) / 100) ** i."
        ),
    )
    width_dispersion_coefficients: list[ParameterType] = Attribute(
        factory=list,
        description=(
            "The coefficients of the polynomial that changes the widths along the global "
            "dimension, used like `center_dispersion_coefficients`."
        ),
    )
    reciproke_global_axis: bool = Attribute(
        default=False,
        description=(
            "Calculate the dispersion with 1000 / value - 1000 / dispersion_center, "
            "for example for a global dimension in wavenumbers."
        ),
    )

    def calculate_dispersion(self, axis: ArrayLike) -> ArrayLike:
        centers = self.center if isinstance(self.center, list) else [self.center]
        widths = self.width if isinstance(self.width, list) else [self.width]
        nr_gaussians = max(len(centers), len(widths))
        if len(centers) != nr_gaussians:
            centers = centers * nr_gaussians
        center = np.array([float(c) for c in centers], dtype=np.float64)
        if self.dispersion_center is None:
            return np.array([np.full(axis.size, c) for c in center])

        axis_values = np.asarray(axis, dtype=np.float64)
        dispersion_center = float(self.dispersion_center)
        distance = (
            (1e3 / axis_values - 1e3 / dispersion_center)
            if self.reciproke_global_axis
            else (axis_values - dispersion_center) / 100
        )
        center_columns = center[:, np.newaxis] * np.ones((nr_gaussians, axis_values.size))
        for i, coefficient in enumerate(self.center_dispersion_coefficients):
            center_columns += float(coefficient) * np.power(distance, i + 1)
        return center_columns

    def is_index_dependent(self) -> bool:
        return self.shift is not None or self.dispersion_center is not None

    @overload
    def parameters(
        self, global_axis: None = None
    ) -> list[GaussianActivationParameters] | list[list[GaussianActivationParameters]]: ...
    @overload
    def parameters(self, global_axis: ArrayLike) -> list[list[GaussianActivationParameters]]: ...

    def parameters(  # noqa: C901
        self, global_axis: ArrayLike | None = None
    ) -> list[GaussianActivationParameters] | list[list[GaussianActivationParameters]]:
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
        backsweep_period = float(self.backsweep) if self.backsweep is not None else 0

        parameters: list[GaussianActivationParameters] = [
            GaussianActivationParameters(
                float(center),
                float(width),
                float(scale),
                backsweep_period,
            )
            for center, width, scale in zip(centers, widths, scales, strict=False)
        ]

        if global_axis is None or not self.is_index_dependent():
            return parameters

        global_parameters: list[list[GaussianActivationParameters]] = [
            [replace(p) for p in parameters] for _ in global_axis
        ]

        if self.shift is not None:
            if global_axis.size != len(self.shift):
                msg = (
                    f"The number of shifts({len(self.shift)}) does not match "
                    f"the size of the global axis({global_axis.size})."
                )
                raise GlotaranUserError(msg)
            for ps, shift in zip(global_parameters, self.shift, strict=True):
                for p in ps:
                    p.shift(shift)  # type:ignore[arg-type]

        if self.dispersion_center is not None:
            for ps, index in zip(global_parameters, global_axis, strict=True):
                for p in ps:
                    p.disperse(
                        index,
                        self.dispersion_center,  # type:ignore[arg-type]
                        self.center_dispersion_coefficients,  # type:ignore[arg-type]
                        self.width_dispersion_coefficients,  # type:ignore[arg-type]
                        reciproke_global_axis=self.reciproke_global_axis,
                    )

        return global_parameters

    def calculate_function(self, axis: ArrayLike) -> ArrayLike:
        return np.sum(
            [
                p.scale * np.exp(-1 * (axis - p.center) ** 2 / (2 * p.width**2))  # type:ignore[union-attr]
                for p in self.parameters()
            ],
            axis=0,
        )


class GaussianActivation(MultiGaussianActivation):
    """Excitation through a Gaussian instrument response function."""

    type: Literal["gaussian"]  # type:ignore[assignment]
    center: ParameterType = Attribute(  # type:ignore[assignment]
        description="The center of the Gaussian on the model dimension."
    )
    width: ParameterType = Attribute(  # type:ignore[assignment]
        description="The width of the Gaussian, as standard deviation."
    )
