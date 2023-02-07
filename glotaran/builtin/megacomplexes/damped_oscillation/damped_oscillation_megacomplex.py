from __future__ import annotations

from typing import TYPE_CHECKING

import numba as nb
import numpy as np
import xarray as xr
from scipy.special import erf

from glotaran.builtin.megacomplexes.decay.decay_parallel_megacomplex import DecayDatasetModel
from glotaran.builtin.megacomplexes.decay.irf import IrfMultiGaussian
from glotaran.builtin.megacomplexes.decay.util import index_dependent
from glotaran.model import DatasetModel
from glotaran.model import ItemIssue
from glotaran.model import Megacomplex
from glotaran.model import Model
from glotaran.model import ParameterType
from glotaran.model import attribute
from glotaran.model import megacomplex
from glotaran.parameter import Parameters

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


class OscillationParameterIssue(ItemIssue):
    def __init__(self, label: str, len_labels: int, len_frequencies: int, len_rates: int):
        self._label = label
        self._len_labels = len_labels
        self._len_frequencies = len_frequencies
        self._len_rates = len_rates

    def to_string(self) -> str:
        return (
            f"Size of labels ({self.len_labels}), frequencies ({self.len_frequencies}) "
            f"and rates ({self.len_rates}) does not match for damped oscillation "
            f"megacomplex '{self.label}'."
        )


def validate_oscillation_parameter(
    labels: list[str],
    damped_oscillation: DampedOscillationMegacomplex,
    model: Model,
    parameters: Parameters | None,
) -> list[ItemIssue]:
    issues = []

    len_labels, len_frequencies, len_rates = (
        len(damped_oscillation.labels),
        len(damped_oscillation.frequencies),
        len(damped_oscillation.rates),
    )

    if len({len_labels, len_frequencies, len_rates}) > 1:
        issues.append(
            OscillationParameterIssue(
                damped_oscillation.label, len_labels, len_frequencies, len_rates
            )
        )

    return issues


@megacomplex(dataset_model_type=DecayDatasetModel)
class DampedOscillationMegacomplex(Megacomplex):
    dimension: str = "time"
    type: str = "damped-oscillation"
    labels: list[str] = attribute(validator=validate_oscillation_parameter)
    frequencies: list[ParameterType]
    rates: list[ParameterType]

    def calculate_matrix(
        self,
        dataset_model: DatasetModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        **kwargs,
    ):
        clp_label = [f"{label}_cos" for label in self.labels] + [
            f"{label}_sin" for label in self.labels
        ]

        delta = np.abs(model_axis[1:] - model_axis[:-1])
        delta_min = delta[np.argmin(delta)]
        # c multiply by 0.03 to convert wavenumber (cm-1) to frequency (THz)
        # where 0.03 is the product of speed of light 3*10**10 cm/s and time-unit ps (10^-12)
        frequency_max = 1 / (2 * 0.03 * delta_min)
        frequencies = np.array(self.frequencies) * 0.03 * 2 * np.pi
        frequencies[frequencies >= frequency_max] = np.mod(
            frequencies[frequencies >= frequency_max], frequency_max
        )
        rates = np.array(self.rates)

        irf = dataset_model.irf
        matrix_shape = (
            (global_axis.size, model_axis.size, len(clp_label))
            if index_dependent(dataset_model)
            else (model_axis.size, len(clp_label))
        )
        matrix = np.ones(matrix_shape, dtype=np.float64)

        if irf is None:
            calculate_damped_oscillation_matrix_no_irf(matrix, frequencies, rates, model_axis)
        elif isinstance(irf, IrfMultiGaussian):
            if index_dependent(dataset_model):
                for i in range(global_axis.size):
                    calculate_damped_oscillation_matrix_gaussian_irf_on_index(
                        matrix[i], frequencies, rates, irf, i, global_axis, model_axis
                    )
            else:
                calculate_damped_oscillation_matrix_gaussian_irf_on_index(
                    matrix, frequencies, rates, irf, None, global_axis, model_axis
                )

        return clp_label, matrix

    def finalize_data(
        self,
        dataset_model: DatasetModel,
        dataset: xr.Dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        if is_full_model:
            return

        megacomplexes = (
            dataset_model.global_megacomplex if is_full_model else dataset_model.megacomplex
        )
        unique = len([m for m in megacomplexes if isinstance(m, DampedOscillationMegacomplex)]) < 2

        prefix = "damped_oscillation" if unique else f"{self.label}_damped_oscillation"

        dataset.coords[f"{prefix}"] = self.labels
        dataset.coords[f"{prefix}_frequency"] = (prefix, self.frequencies)
        dataset.coords[f"{prefix}_rate"] = (prefix, self.rates)

        model_dimension = dataset.attrs["model_dimension"]
        global_dimension = dataset.attrs["global_dimension"]
        dim1 = dataset.coords[global_dimension].size
        dim2 = len(self.labels)
        doas = np.zeros((dim1, dim2), dtype=np.float64)
        phase = np.zeros((dim1, dim2), dtype=np.float64)
        for i, label in enumerate(self.labels):
            sin = dataset.clp.sel(clp_label=f"{label}_sin")
            cos = dataset.clp.sel(clp_label=f"{label}_cos")
            doas[:, i] = np.sqrt(sin * sin + cos * cos)
            phase[:, i] = np.unwrap(np.arctan2(sin, cos))

        dataset[f"{prefix}_associated_spectra"] = (
            (global_dimension, prefix),
            doas,
        )

        dataset[f"{prefix}_phase"] = (
            (global_dimension, prefix),
            phase,
        )

        if index_dependent(dataset_model):
            dataset[f"{prefix}_sin"] = (
                (
                    global_dimension,
                    model_dimension,
                    prefix,
                ),
                dataset.matrix.sel(clp_label=[f"{label}_sin" for label in self.labels]).values,
            )

            dataset[f"{prefix}_cos"] = (
                (
                    global_dimension,
                    model_dimension,
                    prefix,
                ),
                dataset.matrix.sel(clp_label=[f"{label}_cos" for label in self.labels]).values,
            )
        else:
            dataset[f"{prefix}_sin"] = (
                (model_dimension, prefix),
                dataset.matrix.sel(clp_label=[f"{label}_sin" for label in self.labels]).values,
            )

            dataset[f"{prefix}_cos"] = (
                (model_dimension, prefix),
                dataset.matrix.sel(clp_label=[f"{label}_cos" for label in self.labels]).values,
            )


@nb.jit(nopython=True, parallel=True)
def calculate_damped_oscillation_matrix_no_irf(matrix, frequencies, rates, axis):
    idx = 0
    for frequency, rate in zip(frequencies, rates):
        osc = np.exp(-rate * axis - 1j * frequency * axis)
        matrix[:, idx] = osc.real
        matrix[:, idx + 1] = osc.imag
        idx += 2


def calculate_damped_oscillation_matrix_gaussian_irf_on_index(
    matrix: ArrayLike,
    frequencies: ArrayLike,
    rates: ArrayLike,
    irf: IrfMultiGaussian,
    global_index: int | None,
    global_axis: ArrayLike,
    model_axis: ArrayLike,
):
    centers, widths, scales, shift, _, _ = irf.parameter(global_index, global_axis)
    for center, width, scale in zip(centers, widths, scales):
        matrix += calculate_damped_oscillation_matrix_gaussian_irf(
            frequencies,
            rates,
            model_axis,
            center,
            width,
            shift,
            scale,
        )
    matrix /= np.sum(scales)


def calculate_damped_oscillation_matrix_gaussian_irf(
    frequencies: np.ndarray,
    rates: np.ndarray,
    model_axis: np.ndarray,
    center: float,
    width: float,
    shift: float,
    scale: float,
):
    """Calculate the damped oscillation matrix taking into account a gaussian irf

    Parameters
    ----------
    frequencies : np.ndarray
        an array of frequencies in THz, one per oscillation
    rates : np.ndarray
        an array of rates, one per oscillation
    model_axis : np.ndarray
        the model axis (time)
    center : float
        the center of the gaussian IRF
    width : float
        the width (σ) parameter of the the IRF
    shift : float
        a shift parameter per item on the global axis
    scale : float
        the scale parameter to scale the matrix by

    Returns
    -------
    np.ndarray
        An array of the real and imaginary part of the oscillation matrix,
        the shape being (len(model_axis), 2*len(frequencies)), with the first
        half of the second dimension representing the real part,
        and the other the imagine part of the oscillation
    """
    shifted_axis = model_axis - center - shift
    # For calculations using the negative rates we use the time axis
    # from the beginning up to 5 σ from the irf center
    left_shifted_axis_indices = np.where(shifted_axis < 5 * width)[0]
    left_shifted_axis = shifted_axis[left_shifted_axis_indices]
    neg_idx = np.where(rates < 0)[0]
    # For calculations using the positive rates axis we use the time axis
    # from 5 σ before the irf center until the end
    right_shifted_axis_indices = np.where(shifted_axis > -5 * width)[0]
    right_shifted_axis = shifted_axis[right_shifted_axis_indices]
    pos_idx = np.where(rates >= 0)[0]

    d = width**2
    k = rates + 1j * frequencies
    dk = k * d
    sqwidth = np.sqrt(2) * width

    a = np.zeros((len(model_axis), len(rates)), dtype=np.complex128)
    a[np.ix_(right_shifted_axis_indices, pos_idx)] = np.exp(
        (-1 * right_shifted_axis[:, None] + 0.5 * dk[pos_idx]) * k[pos_idx]
    )

    a[np.ix_(left_shifted_axis_indices, neg_idx)] = np.exp(
        (-1 * left_shifted_axis[:, None] + 0.5 * dk[neg_idx]) * k[neg_idx]
    )

    b = np.zeros((len(model_axis), len(rates)), dtype=np.complex128)
    b[np.ix_(right_shifted_axis_indices, pos_idx)] = 1 + erf(
        (right_shifted_axis[:, None] - dk[pos_idx]) / sqwidth
    )
    # For negative rates we flip the sign of the `erf` by using `-sqwidth` in lieu of `sqwidth`
    b[np.ix_(left_shifted_axis_indices, neg_idx)] = 1 + erf(
        (left_shifted_axis[:, None] - dk[neg_idx]) / -sqwidth
    )

    osc = a * b * scale

    return np.concatenate((osc.real, osc.imag), axis=1)
