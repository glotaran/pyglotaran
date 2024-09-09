from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Literal

import numpy as np
import xarray as xr

from glotaran.builtin.elements.damped_oscillation.matrix import (
    calculate_damped_oscillation_matrix_gaussian_activation,
)
from glotaran.builtin.elements.damped_oscillation.matrix import (
    calculate_damped_oscillation_matrix_gaussian_activation_on_index,
)
from glotaran.builtin.elements.damped_oscillation.matrix import (
    calculate_damped_oscillation_matrix_instant_activation,
)
from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.builtin.items.activation import MultiGaussianActivation
from glotaran.model.data_model import DataModel  # noqa: TCH001
from glotaran.model.element import Element
from glotaran.model.element import ElementResult
from glotaran.model.item import Item
from glotaran.model.item import ParameterType

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


class Oscillation(Item):
    frequency: ParameterType
    rate: ParameterType


class DampedOscillationElement(Element):
    type: Literal["damped-oscillation"]  # type:ignore[assignment]
    register_as: ClassVar[str] = "damped-oscillation"
    dimension: str = "time"
    data_model_type: ClassVar[type[DataModel]] = ActivationDataModel  # type:ignore[valid-type]
    oscillations: dict[str, Oscillation]

    def calculate_matrix(  # type:ignore[override]
        self,
        model: ActivationDataModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
    ):
        delta = np.abs(model_axis[1:] - model_axis[:-1])
        delta_min = delta[np.argmin(delta)]
        # c multiply by 0.03 to convert wavenumber (cm-1) to frequency (THz)
        # where 0.03 is the product of speed of light 3*10**10 cm/s and time-unit ps (10^-12)
        frequency_max = 1 / (2 * 0.03 * delta_min)

        clp_labels = []
        matrices = []
        for activation in model.activation:
            oscillation_labels = [
                label for label in self.oscillations if label in activation.compartments
            ]
            if not len(oscillation_labels):
                continue

            clp_label = [f"{label}_cos" for label in oscillation_labels] + [
                f"{label}_sin" for label in oscillation_labels
            ]

            inputs = np.array([activation.compartments[label] for label in oscillation_labels])
            frequencies = (
                np.array([self.oscillations[label].frequency for label in oscillation_labels])
                * 0.03
                * 2
                * np.pi
            )
            frequencies[frequencies >= frequency_max] = np.mod(
                frequencies[frequencies >= frequency_max], frequency_max
            )
            rates = np.array([self.oscillations[label].rate for label in oscillation_labels])
            if isinstance(activation, MultiGaussianActivation):
                parameters = activation.parameters(global_axis)
                matrix_shape = (
                    (global_axis.size, model_axis.size, len(clp_label))
                    if activation.is_index_dependent()
                    else (model_axis.size, len(clp_label))
                )
                matrix = np.zeros(matrix_shape, dtype=np.float64)

                if activation.is_index_dependent():
                    calculate_damped_oscillation_matrix_gaussian_activation(
                        matrix,
                        inputs,
                        frequencies,
                        rates,
                        parameters,  # type:ignore[arg-type]
                        model_axis,
                    )
                else:
                    calculate_damped_oscillation_matrix_gaussian_activation_on_index(
                        matrix,
                        inputs,
                        frequencies,
                        rates,
                        parameters,  # type:ignore[arg-type]
                        model_axis,
                    )
            else:
                matrix = np.zeros((model_axis.size, len(clp_label)), dtype=np.float64)
                calculate_damped_oscillation_matrix_instant_activation(
                    matrix, inputs, frequencies, rates, model_axis
                )

            clp_labels.append(clp_label)
            matrices.append(matrix)

        if len(matrices) == 1:
            return clp_labels[0], matrices[0]

        clp_axis = [f"{label}_cos" for label in self.oscillations] + [
            f"{label}_sin" for label in self.oscillations
        ]
        index_dependent = any(len(m.shape) > 2 for m in matrices)
        matrix_shape = (
            (global_axis.size, model_axis.size, len(clp_labels))
            if index_dependent
            else (model_axis.size, len(clp_axis))
        )
        matrix = np.zeros(matrix_shape, dtype=np.float64)
        for clp_label, m in zip(clp_labels, matrices):
            need_new_axis = len(matrix.shape) > len(m.shape)
            matrix[..., [clp_axis.index(label) for label in clp_label]] += (
                m[np.newaxis, :, :] if need_new_axis else m
            )

        return clp_axis, matrix

    def create_result(
        self,
        model: ActivationDataModel,
        global_dimension: str,
        model_dimension: str,
        amplitudes: xr.Dataset,
        concentrations: xr.Dataset,
    ) -> ElementResult:
        oscillations = list(self.oscillations)
        frequencies = [self.oscillations[label].frequency for label in oscillations]
        rates = [self.oscillations[label].rate for label in oscillations]

        oscillation_coords = {
            "damped_oscillation": oscillations,
            "damped_oscillation_frequency": ("damped_oscillation", frequencies),
            "damped_oscillation_rate": ("damped_oscillation", rates),
        }

        sin_label = [f"{label}_sin" for label in oscillations]
        cos_label = [f"{label}_cos" for label in oscillations]

        sin_amplitudes = (
            amplitudes.sel(amplitude_label=sin_label)
            .rename(amplitude_label="damped_oscillation")
            .assign_coords(oscillation_coords)
        )
        cos_amplitudes = (
            amplitudes.sel(amplitude_label=cos_label)
            .rename(amplitude_label="damped_oscillation")
            .assign_coords(oscillation_coords)
        )
        doas_amplitudes = np.sqrt(sin_amplitudes**2 + cos_amplitudes**2)
        phase_amplitudes = xr.DataArray(
            np.unwrap(np.arctan2(sin_amplitudes, cos_amplitudes)),
            coords=doas_amplitudes.coords,
        )

        sin_concentrations = (
            concentrations.sel(amplitude_label=sin_label)
            .rename(amplitude_label="damped_oscillation")
            .assign_coords(oscillation_coords)
        )
        cos_concentrations = (
            concentrations.sel(amplitude_label=cos_label)
            .rename(amplitude_label="damped_oscillation")
            .assign_coords(oscillation_coords)
        )
        doas_concentrations = np.sqrt(sin_concentrations**2 + cos_concentrations**2)
        phase_concentrations = xr.DataArray(
            np.unwrap(np.arctan2(sin_concentrations, cos_concentrations)),
            coords=doas_concentrations.coords,
        )

        return ElementResult(
            amplitudes={
                "damped_oscillation": doas_amplitudes,
                "damped_oscillation_phase": phase_amplitudes,
                "damped_oscillation_sin": sin_amplitudes,
                "damped_oscillation_cos": cos_amplitudes,
            },
            concentrations={
                "damped_oscillation": doas_concentrations,
                "damped_oscillation_phase": phase_concentrations,
                "damped_oscillation_sin": sin_concentrations,
                "damped_oscillation_cos": cos_concentrations,
            },
        )
