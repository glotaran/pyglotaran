from __future__ import annotations

from typing import TYPE_CHECKING
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
from glotaran.model import Element
from glotaran.model import Item
from glotaran.model import ParameterType

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


class Oscillation(Item):
    frequency: ParameterType
    rate: ParameterType


class DampedOscillationElement(Element):
    type: Literal["damped-oscillation"]
    register_as = "damped-oscillation"
    dimension = "time"
    data_model = ActivationDataModel
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
                        matrix, inputs, frequencies, rates, parameters, model_axis
                    )
                else:
                    calculate_damped_oscillation_matrix_gaussian_activation_on_index(
                        matrix, inputs, frequencies, rates, parameters, model_axis
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
            (global_axis.size, model_axis.size, len(clp_label))
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

    def add_to_result_data(  # type:ignore[override]
        self,
        model: ActivationDataModel,
        data: xr.Dataset,
        as_global: bool = False,
    ):
        prefix = "damped_oscillation"
        if as_global or prefix in data:
            # not implemented
            return

        elements = [m for m in model.elements if isinstance(m, DampedOscillationElement)]
        oscillations = [label for m in elements for label in m.oscillations]
        frequencies = [o.frequency for m in elements for o in m.oscillations.values()]
        rates = [o.rate for m in elements for o in m.oscillations.values()]

        data.coords[f"{prefix}"] = oscillations
        data.coords[f"{prefix}_frequency"] = (prefix, frequencies)
        data.coords[f"{prefix}_rate"] = (prefix, rates)

        model_dimension = data.attrs["model_dimension"]
        global_dimension = data.attrs["global_dimension"]
        dim1 = data.coords[global_dimension].size
        dim2 = len(oscillations)
        doas = np.zeros((dim1, dim2), dtype=np.float64)
        phase = np.zeros((dim1, dim2), dtype=np.float64)
        for i, label in enumerate(oscillations):
            sin = data.clp.sel(clp_label=f"{label}_sin")
            cos = data.clp.sel(clp_label=f"{label}_cos")
            doas[:, i] = np.sqrt(sin * sin + cos * cos)
            phase[:, i] = np.unwrap(np.arctan2(sin, cos))

        data[f"{prefix}_associated_estimation"] = ((global_dimension, prefix), doas)

        data[f"{prefix}_phase"] = ((global_dimension, prefix), phase)

        if len(data.matrix.shape) > 2:
            data[f"{prefix}_sin"] = (
                (
                    global_dimension,
                    model_dimension,
                    prefix,
                ),
                data.matrix.sel(clp_label=[f"{label}_sin" for label in oscillations]).values,
            )

            data[f"{prefix}_cos"] = (
                (
                    global_dimension,
                    model_dimension,
                    prefix,
                ),
                data.matrix.sel(clp_label=[f"{label}_cos" for label in oscillations]).values,
            )
        else:
            data[f"{prefix}_sin"] = (
                (model_dimension, prefix),
                data.matrix.sel(clp_label=[f"{label}_sin" for label in oscillations]).values,
            )

            data[f"{prefix}_cos"] = (
                (model_dimension, prefix),
                data.matrix.sel(clp_label=[f"{label}_cos" for label in oscillations]).values,
            )
