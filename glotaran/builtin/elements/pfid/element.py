from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Literal

import numpy as np
import xarray as xr

from glotaran.builtin.elements.pfid.matrix import calculate_pfid_matrix
from glotaran.builtin.items.activation import MultiGaussianActivation  # noqa: TC001
from glotaran.model.data_model import DataModel
from glotaran.model.element import Element
from glotaran.model.item import Item
from glotaran.model.item import ParameterType

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


class PFIDOscillation(Item):
    """Parameters describing one perturbed free-induction-decay oscillation."""

    frequency: ParameterType
    rate: ParameterType


class PFIDDataModel(DataModel):
    """Dataset fields required by a PFID element."""

    activation: MultiGaussianActivation
    spectral_axis_inverted: bool = False
    spectral_axis_scale: float = 1


class PFIDElement(Element):
    """Element for perturbed free-induction-decay fitting."""

    type: Literal["pfid"]  # type:ignore[assignment]
    register_as: ClassVar[str] = "pfid"
    dimension: str = "time"
    data_model_type: ClassVar[type[DataModel]] = PFIDDataModel  # type:ignore[valid-type]
    oscillations: dict[str, PFIDOscillation]

    def calculate_matrix(  # type:ignore[override]
        self,
        model: PFIDDataModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
    ) -> tuple[list[str], ArrayLike]:
        labels = list(self.oscillations)
        clp_labels = [f"{label}_cos" for label in labels] + [f"{label}_sin" for label in labels]
        frequencies = np.array([float(self.oscillations[label].frequency) for label in labels])
        rates = np.array([float(self.oscillations[label].rate) for label in labels])

        if model.spectral_axis_inverted:
            frequencies = model.spectral_axis_scale / frequencies
        elif model.spectral_axis_scale != 1:
            frequencies *= model.spectral_axis_scale

        parameters = model.activation.parameters(global_axis)
        if not any(isinstance(parameter, list) for parameter in parameters):
            parameters = [parameters for _ in global_axis]

        matrix = np.zeros((global_axis.size, model_axis.size, len(clp_labels)), dtype=np.float64)
        calculate_pfid_matrix(
            matrix,
            frequencies,
            rates,
            parameters,  # type:ignore[arg-type]
            global_axis,
            model_axis,
        )
        return clp_labels, matrix

    def create_result(
        self,
        model: PFIDDataModel,
        global_dimension: str,
        model_dimension: str,
        amplitudes: xr.DataArray,
        concentrations: xr.DataArray,
    ) -> xr.Dataset:
        labels = list(self.oscillations)
        coordinates = {
            "oscillation": labels,
            "oscillation_frequency": (
                "oscillation",
                [self.oscillations[label].frequency for label in labels],
            ),
            "oscillation_rate": (
                "oscillation",
                [self.oscillations[label].rate for label in labels],
            ),
        }
        sin_labels = [f"{label}_sin" for label in labels]
        cos_labels = [f"{label}_cos" for label in labels]

        sin_amplitudes = (
            amplitudes.sel(amplitude_label=sin_labels)
            .rename(amplitude_label="oscillation")
            .assign_coords(coordinates)
        )
        cos_amplitudes = (
            amplitudes.sel(amplitude_label=cos_labels)
            .rename(amplitude_label="oscillation")
            .assign_coords(coordinates)
        )
        associated_spectra = np.sqrt(sin_amplitudes**2 + cos_amplitudes**2)
        phase = xr.DataArray(
            np.unwrap(np.arctan2(sin_amplitudes, cos_amplitudes), axis=0),
            coords=associated_spectra.coords,
        )

        sin_concentrations = (
            concentrations.sel(amplitude_label=sin_labels)
            .rename(amplitude_label="oscillation")
            .assign_coords(coordinates)
        )
        cos_concentrations = (
            concentrations.sel(amplitude_label=cos_labels)
            .rename(amplitude_label="oscillation")
            .assign_coords(coordinates)
        )
        associated_concentrations = np.sqrt(sin_concentrations**2 + cos_concentrations**2)
        phase_concentrations = xr.DataArray(
            np.unwrap(np.arctan2(sin_concentrations, cos_concentrations), axis=0),
            coords=associated_concentrations.coords,
        )

        return xr.Dataset(
            {
                "amplitudes": associated_spectra,
                "phase": phase,
                "sin_amplitudes": sin_amplitudes,
                "cos_amplitudes": cos_amplitudes,
                "concentrations": associated_concentrations,
                "phase_concentrations": phase_concentrations,
                "sin_concentrations": sin_concentrations,
                "cos_concentrations": cos_concentrations,
            }
        )
