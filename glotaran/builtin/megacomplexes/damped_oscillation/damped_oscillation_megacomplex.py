from __future__ import annotations

from typing import List

import numba as nb
import numpy as np
import xarray as xr
from scipy.special import erf

from glotaran.builtin.megacomplexes.decay.irf import Irf
from glotaran.builtin.megacomplexes.decay.irf import IrfMultiGaussian
from glotaran.model import DatasetModel
from glotaran.model import Megacomplex
from glotaran.model import Model
from glotaran.model import megacomplex
from glotaran.model.item import model_item_validator
from glotaran.parameter import Parameter


@megacomplex(
    dimension="time",
    dataset_model_items={
        "irf": {"type": Irf, "allow_none": True},
    },
    properties={
        "labels": List[str],
        "frequencies": List[Parameter],
        "rates": List[Parameter],
    },
    register_as="damped-oscillation",
)
class DampedOscillationMegacomplex(Megacomplex):
    @model_item_validator(False)
    def ensure_oscillation_parameter(self, model: Model) -> list[str]:

        problems = []

        if len(self.labels) != len(self.frequencies) or len(self.labels) != len(self.rates):
            problems.append(
                f"Size of labels ({len(self.labels)}), frequencies ({len(self.frequencies)}) "
                f"and rates ({len(self.rates)}) does not match for damped oscillation "
                f"megacomplex '{self.label}'."
            )

        return problems

    def calculate_matrix(
        self,
        dataset_model: DatasetModel,
        indices: dict[str, int],
        **kwargs,
    ):

        clp_label = [f"{label}_cos" for label in self.labels] + [
            f"{label}_sin" for label in self.labels
        ]

        model_axis = dataset_model.get_model_axis()
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

        matrix = np.ones((model_axis.size, len(clp_label)), dtype=np.float64)

        if dataset_model.irf is None:
            calculate_damped_oscillation_matrix_no_irf(matrix, frequencies, rates, model_axis)
        elif isinstance(dataset_model.irf, IrfMultiGaussian):
            global_dimension = dataset_model.get_global_dimension()
            global_axis = dataset_model.get_global_axis()
            global_index = indices.get(global_dimension)
            centers, widths, scales, shift, _, _ = dataset_model.irf.parameter(
                global_index, global_axis
            )
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

        return clp_label, matrix

    def index_dependent(self, dataset_model: DatasetModel) -> bool:
        return (
            isinstance(dataset_model.irf, IrfMultiGaussian)
            and dataset_model.irf.is_index_dependent()
        )

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

        dim1 = dataset_model.get_global_axis().size
        dim2 = len(self.labels)
        doas = np.zeros((dim1, dim2), dtype=np.float64)
        phase = np.zeros((dim1, dim2), dtype=np.float64)
        for i, label in enumerate(self.labels):
            sin = dataset.clp.sel(clp_label=f"{label}_sin")
            cos = dataset.clp.sel(clp_label=f"{label}_cos")
            doas[:, i] = np.sqrt(sin * sin + cos * cos)
            phase[:, i] = np.unwrap(np.arctan2(sin, cos))

        dataset[f"{prefix}_associated_spectra"] = (
            (dataset_model.get_global_dimension(), prefix),
            doas,
        )

        dataset[f"{prefix}_phase"] = (
            (dataset_model.get_global_dimension(), prefix),
            phase,
        )

        if self.index_dependent(dataset_model):
            dataset[f"{prefix}_sin"] = (
                (
                    dataset_model.get_global_dimension(),
                    dataset_model.get_model_dimension(),
                    prefix,
                ),
                dataset.matrix.sel(clp_label=[f"{label}_sin" for label in self.labels]).values,
            )

            dataset[f"{prefix}_cos"] = (
                (
                    dataset_model.get_global_dimension(),
                    dataset_model.get_model_dimension(),
                    prefix,
                ),
                dataset.matrix.sel(clp_label=[f"{label}_cos" for label in self.labels]).values,
            )
        else:
            dataset[f"{prefix}_sin"] = (
                (dataset_model.get_model_dimension(), prefix),
                dataset.matrix.sel(clp_label=[f"{label}_sin" for label in self.labels]).values,
            )

            dataset[f"{prefix}_cos"] = (
                (dataset_model.get_model_dimension(), prefix),
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

    d = width ** 2
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
