from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from scipy.special import erf

from glotaran.builtin.megacomplexes.decay.irf import Irf
from glotaran.builtin.megacomplexes.decay.irf import IrfMultiGaussian
from glotaran.model import DatasetModel
from glotaran.model import ItemIssue
from glotaran.model import Megacomplex
from glotaran.model import Model
from glotaran.model import ModelItemType
from glotaran.model import ParameterType
from glotaran.model import attribute
from glotaran.model import item
from glotaran.model import megacomplex

if TYPE_CHECKING:
    from glotaran.parameter import Parameters
    from glotaran.typing.types import ArrayLike


class OscillationParameterIssue(ItemIssue):
    def __init__(self, label: str, len_labels: int, len_frequencies: int, len_rates: int):
        self.label = label
        self.len_labels = len_labels
        self.len_frequencies = len_frequencies
        self.len_rates = len_rates

    def to_string(self) -> str:
        return (
            f"The size of labels ({self.len_labels}), frequencies ({self.len_frequencies}), "
            f"and rates ({self.len_rates}) does not match for pfid "
            f"megacomplex '{self.label}'."
        )


def validate_pfid_parameter(
    labels: list[str],
    pfid: PFIDMegacomplex,
    model: Model,
    parameters: Parameters | None,
) -> list[ItemIssue]:
    issues = []

    len_labels, len_frequencies, len_rates = (
        len(pfid.labels),
        len(pfid.frequencies),
        len(pfid.rates),
    )

    if len({len_labels, len_frequencies, len_rates}) > 1:
        issues.append(
            OscillationParameterIssue(pfid.label, len_labels, len_frequencies, len_rates)
        )

    return issues


@item
class PFIDDatasetModel(DatasetModel):
    spectral_axis_inverted: bool = False
    spectral_axis_scale: float = 1
    irf: ModelItemType[Irf] | None = None


@megacomplex(dataset_model_type=PFIDDatasetModel)
class PFIDMegacomplex(Megacomplex):
    dimension: str = "time"
    type: str = "pfid"
    labels: list[str] = attribute(validator=validate_pfid_parameter)
    frequencies: list[ParameterType]  # omega_a
    rates: list[ParameterType]  # 1/T2

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

        frequencies = np.array(self.frequencies)
        rates = np.array(self.rates)

        if dataset_model.spectral_axis_inverted:
            frequencies = dataset_model.spectral_axis_scale / frequencies
        elif dataset_model.spectral_axis_scale != 1:
            frequencies = frequencies * dataset_model.spectral_axis_scale

        irf = dataset_model.irf
        matrix_shape = (global_axis.size, model_axis.size, len(clp_label))
        matrix = np.zeros(matrix_shape, dtype=np.float64)

        if irf is None:
            msg = "IRF is required for PFID megacomplex"
            raise ValueError(msg)
        if isinstance(irf, IrfMultiGaussian):
            for i in range(global_axis.size):
                calculate_pfid_matrix_gaussian_irf_on_index(
                    matrix[i],
                    frequencies,
                    rates,
                    irf,
                    i,
                    global_axis,
                    model_axis,
                )
        else:
            msg = "IRF should be instance of IrfMultiGaussian"
            raise ValueError(msg)
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
        unique = len([m for m in megacomplexes if isinstance(m, PFIDMegacomplex)]) < 2

        prefix = "pfid" if unique else f"{self.label}_pfid"

        dataset.coords[f"{prefix}"] = self.labels
        dataset.coords[f"{prefix}_frequency"] = (prefix, self.frequencies)
        dataset.coords[f"{prefix}_rate"] = (prefix, self.rates)

        model_dimension = dataset.attrs["model_dimension"]
        global_dimension = dataset.attrs["global_dimension"]
        dim1 = dataset.coords[global_dimension].size
        dim2 = len(self.labels)
        pfid = np.zeros((dim1, dim2), dtype=np.float64)
        phase = np.zeros((dim1, dim2), dtype=np.float64)

        for i, label in enumerate(self.labels):
            sin = dataset.clp.sel(clp_label=f"{label}_sin")
            cos = dataset.clp.sel(clp_label=f"{label}_cos")
            pfid[:, i] = np.sqrt(sin * sin + cos * cos)
            phase[:, i] = np.unwrap(np.arctan2(sin, cos))

        dataset[f"{prefix}_associated_spectra"] = (
            (global_dimension, prefix),
            pfid,
        )

        dataset[f"{prefix}_phase"] = (
            (global_dimension, prefix),
            phase,
        )

        dataset[f"{prefix}_sin"] = (
            (
                global_dimension,
                model_dimension,
                prefix,
            ),
            dataset.matrix.sel(clp_label=[f"{label}_sin" for label in self.labels]).to_numpy(),
        )

        dataset[f"{prefix}_cos"] = (
            (
                global_dimension,
                model_dimension,
                prefix,
            ),
            dataset.matrix.sel(clp_label=[f"{label}_cos" for label in self.labels]).to_numpy(),
        )


def calculate_pfid_matrix_gaussian_irf_on_index(
    matrix: ArrayLike,
    frequencies: ArrayLike,
    rates: ArrayLike,
    irf: IrfMultiGaussian,
    global_index: int | None,
    global_axis: ArrayLike,
    model_axis: ArrayLike,
):
    centers, widths, scales, shift, _, _ = irf.parameter(global_index, global_axis)
    for center, width, scale in zip(centers, widths, scales, strict=True):
        matrix += calculate_pfid_matrix_gaussian_irf(
            frequencies,
            rates,
            model_axis,
            center,
            width,
            shift,
            scale,
            global_axis[global_index],
        )
    matrix /= np.sum(scales)


def calculate_pfid_matrix_gaussian_irf(
    frequencies: np.ndarray,
    rates: np.ndarray,
    model_axis: np.ndarray,
    center: float,
    width: float,
    shift: float,
    scale: float,
    global_axis_value: float,
):
    """Calculate the damped oscillation matrix taking into account a gaussian irf.

    Parameters
    ----------
    frequencies : np.ndarray
        an array of frequencies in THz, one per oscillation
    rates : np.ndarray
        an array of dephasing rates (negative), one per oscillation
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
        the shape being (len(model_axis), len(frequencies)).
    """
    shifted_axis = model_axis - center - shift
    # For calculations using the negative rates we use the time axis
    # from the beginning up to 5 σ from the irf center
    # this is to guard again overflows
    left_shifted_axis_indices = np.where(shifted_axis < 5 * width)[0]
    left_shifted_axis = shifted_axis[left_shifted_axis_indices]
    neg_idx = np.where(rates < 0)[0]

    # c multiply by 0.03 to convert wavenumber (cm-1) to frequency (THz)
    # where 0.03 is the product of speed of light 3*10**10 cm/s and time-unit ps (10^-12)
    # we postpone the conversion because the global axis is
    # always expected to be in cm-1 for relevant experiments
    frequency_diff = (global_axis_value - frequencies) * 0.03 * 2 * np.pi
    d = width**2
    k = rates + 1j * frequency_diff
    dk = k * d
    sqwidth = np.sqrt(2) * width

    a = np.zeros((len(model_axis), len(rates)), dtype=np.complex128)
    a[np.ix_(left_shifted_axis_indices, neg_idx)] = np.exp(
        (-1 * left_shifted_axis[:, None] + 0.5 * dk[:]) * k[:]
    )

    b = np.zeros((len(model_axis), len(rates)), dtype=np.complex128)
    # For negative rates we flip the sign of the `erf` by using `-sqwidth` in lieu of `sqwidth`
    b[np.ix_(left_shifted_axis_indices, neg_idx)] = 1 + erf(
        (left_shifted_axis[:, None] - dk[:]) / -sqwidth
    )

    osc = -(a * b) * scale

    return np.concatenate((osc.real, osc.imag), axis=1)
