from __future__ import annotations

from typing import TYPE_CHECKING

import numba as nb
import numpy as np
import xarray as xr

from glotaran.builtin.megacomplexes.decay.decay_matrix_gaussian_irf import (
    calculate_decay_matrix_gaussian_irf,
)
from glotaran.builtin.megacomplexes.decay.decay_matrix_gaussian_irf import (
    calculate_decay_matrix_gaussian_irf_on_index,
)
from glotaran.builtin.megacomplexes.decay.irf import IrfMultiGaussian
from glotaran.builtin.megacomplexes.decay.irf import IrfSpectralMultiGaussian
from glotaran.model import DatasetModel
from glotaran.model import Megacomplex
from glotaran.model import get_dataset_model_model_dimension

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


def index_dependent(dataset_model: DatasetModel) -> bool:
    """Determine if a dataset_model is index dependent.

    Parameters
    ----------
    dataset_model : DatasetModel
        A dataset model instance.

    Returns
    -------
    bool
        Returns True if the dataset_model has an IRF that is index dependent (e.g. has dispersion).
    """
    return (
        isinstance(dataset_model.irf, IrfMultiGaussian) and dataset_model.irf.is_index_dependent()
    )


def calculate_matrix(
    megacomplex: Megacomplex,
    dataset_model: DatasetModel,
    global_axis: ArrayLike,
    model_axis: ArrayLike,
    **kwargs,
):
    compartments = megacomplex.get_compartments(dataset_model)
    initial_concentration = megacomplex.get_initial_concentration(dataset_model)
    k_matrix = megacomplex.get_k_matrix()

    rates = k_matrix.rates(compartments, initial_concentration)

    # init the matrix
    matrix_shape = (
        (global_axis.size, model_axis.size, rates.size)
        if index_dependent(dataset_model)
        else (model_axis.size, rates.size)
    )
    matrix = np.zeros(matrix_shape, dtype=np.float64)

    if index_dependent(dataset_model):
        decay_matrix_implementation_index_dependent(
            matrix, rates, global_axis, model_axis, dataset_model
        )
    else:
        decay_matrix_implementation_index_independent(
            matrix, rates, global_axis, model_axis, dataset_model
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"Non-finite concentrations for K-Matrix '{k_matrix.label}':\n"
            f"{k_matrix.matrix_as_markdown(fill_parameters=True)}"
        )

    # apply A matrix
    matrix = matrix @ megacomplex.get_a_matrix(dataset_model)

    # done
    return compartments, matrix


def collect_megacomplexes(dataset_model: DatasetModel, as_global: bool) -> list[Megacomplex]:
    from glotaran.builtin.megacomplexes.decay.decay_megacomplex import DecayMegacomplex
    from glotaran.builtin.megacomplexes.decay.decay_parallel_megacomplex import (
        DecayParallelMegacomplex,
    )
    from glotaran.builtin.megacomplexes.decay.decay_sequential_megacomplex import (
        DecaySequentialMegacomplex,
    )

    return list(
        filter(
            lambda m: isinstance(
                m, (DecayMegacomplex, DecayParallelMegacomplex, DecaySequentialMegacomplex)
            ),
            dataset_model.global_megacomplex if as_global else dataset_model.megacomplex,
        )
    )


def finalize_data(
    dataset_model: DatasetModel,
    dataset: xr.Dataset,
    is_full_model: bool = False,
    as_global: bool = False,
):
    species_dimension = "decay_species" if as_global else "species"
    if species_dimension in dataset.coords:
        # The first decay megacomplescomplex called will finalize the data for all
        # decay megacomplexes.
        return

    decay_megacomplexes = collect_megacomplexes(dataset_model, as_global)
    global_dimension = dataset.attrs["global_dimension"]
    name = "images" if global_dimension == "pixel" else "spectra"

    all_species = []
    for megacomplex in decay_megacomplexes:
        for species in megacomplex.get_compartments(dataset_model):
            if species not in all_species:
                all_species.append(species)
    retrieve_species_associated_data(
        dataset_model,
        dataset,
        all_species,
        species_dimension,
        global_dimension,
        name,
        is_full_model,
        as_global,
    )
    retrieve_initial_concentration(
        dataset_model,
        dataset,
        species_dimension,
    )
    retrieve_irf(dataset_model, dataset, global_dimension)

    if not is_full_model:
        for megacomplex in decay_megacomplexes:
            retrieve_decay_associated_data(
                megacomplex,
                dataset_model,
                dataset,
                global_dimension,
                name,
            )


def decay_matrix_implementation_index_independent(
    matrix: np.ndarray,
    rates: np.ndarray,
    global_axis: np.ndarray,
    model_axis: np.ndarray,
    dataset_model: DatasetModel,
):
    if isinstance(dataset_model.irf, IrfMultiGaussian):
        (
            centers,
            widths,
            irf_scales,
            shift,
            backsweep,
            backsweep_period,
        ) = dataset_model.irf.parameter(None, global_axis)

        calculate_decay_matrix_gaussian_irf_on_index(
            matrix,
            rates,
            model_axis,
            centers - shift,
            widths,
            irf_scales,
            backsweep,
            backsweep_period,
        )
        if dataset_model.irf.normalize:
            matrix /= np.sum(irf_scales)

    else:
        calculate_decay_matrix_no_irf(matrix, rates, model_axis)


def decay_matrix_implementation_index_dependent(
    matrix: np.ndarray,
    rates: np.ndarray,
    global_axis: np.ndarray,
    model_axis: np.ndarray,
    dataset_model: DatasetModel,
):
    all_centers, all_widths = [], []
    backsweep, backsweep_period = False, None
    irf_scales = []
    for global_index in range(global_axis.size):
        (
            centers,
            widths,
            irf_scales,
            shift,
            backsweep,
            backsweep_period,
        ) = dataset_model.irf.parameter(global_index, global_axis)
        all_centers.append(centers - shift)
        all_widths.append(widths)

    calculate_decay_matrix_gaussian_irf(
        matrix,
        rates,
        model_axis,
        np.array(all_centers),
        np.array(all_widths),
        irf_scales,
        backsweep,
        backsweep_period,
    )
    if dataset_model.irf.normalize:
        matrix /= np.sum(irf_scales)


@nb.jit(nopython=True, parallel=True)
def calculate_decay_matrix_no_irf(matrix, rates, times):
    for n_r in nb.prange(rates.size):
        r_n = rates[n_r]
        for n_t in range(times.size):
            t_n = times[n_t]
            matrix[n_t, n_r] += np.exp(-r_n * t_n)


def retrieve_species_associated_data(
    dataset_model: DatasetModel,
    dataset: xr.Dataset,
    species: list[str],
    species_dimension: str,
    global_dimension: str,
    name: str,
    is_full_model: bool,
    as_global: bool,
):
    model_dimension = get_dataset_model_model_dimension(dataset_model)
    if as_global:
        model_dimension, global_dimension = global_dimension, model_dimension
    dataset.coords[species_dimension] = species

    matrix = dataset.global_matrix if as_global else dataset.matrix
    clp_dim = "global_clp_label" if as_global else "clp_label"

    if len(dataset.matrix.shape) == 3:
        #  index dependent
        dataset["species_concentration"] = (
            (
                global_dimension,
                model_dimension,
                species_dimension,
            ),
            matrix.sel({clp_dim: species}).values,
        )
    else:
        #  index independent
        dataset["species_concentration"] = (
            (
                model_dimension,
                species_dimension,
            ),
            matrix.sel({clp_dim: species}).values,
        )

    if not is_full_model:
        dataset[f"species_associated_{name}"] = (
            (
                global_dimension,
                species_dimension,
            ),
            dataset.clp.sel(clp_label=species).data,
        )


def retrieve_initial_concentration(
    dataset_model: DatasetModel,
    dataset: xr.Dataset,
    species_dimension: str,
):
    if (
        not hasattr(dataset_model, "initial_concentration")
        or dataset_model.initial_concentration is None
    ):
        # For parallel and sequential decay we don't have dataset wide initial concentration
        # unless mixed with general decays
        return

    dataset["initial_concentration"] = (
        (species_dimension,),
        dataset_model.initial_concentration.parameters,
    )


def retrieve_decay_associated_data(
    megacomplex: Megacomplex,
    dataset_model: DatasetModel,
    dataset: xr.Dataset,
    global_dimension: str,
    name: str,
):
    species = megacomplex.get_compartments(dataset_model)
    initial_concentration = megacomplex.get_initial_concentration(dataset_model)
    k_matrix = megacomplex.get_k_matrix()

    matrix = k_matrix.full(species)
    matrix_reduced = k_matrix.reduced(species)
    a_matrix = megacomplex.get_a_matrix(dataset_model)
    rates = k_matrix.rates(species, initial_concentration)
    lifetimes = 1 / rates

    das = dataset[f"species_associated_{name}"].sel(species=species).values @ a_matrix.T

    component_name = f"component_{megacomplex.label}"
    component_coords = {
        component_name: np.arange(1, rates.size + 1),
        f"rate_{megacomplex.label}": (component_name, rates),
        f"lifetime_{megacomplex.label}": (component_name, lifetimes),
    }
    das_coords = component_coords.copy()
    das_coords[global_dimension] = dataset.coords[global_dimension]
    das_name = f"decay_associated_{name}_{megacomplex.label}"
    das = xr.DataArray(das, dims=(global_dimension, component_name), coords=das_coords)

    initial_concentration = megacomplex.get_initial_concentration(dataset_model, normalized=False)
    species_name = f"species_{megacomplex.label}"
    a_matrix_coords = component_coords.copy()
    a_matrix_coords[species_name] = species
    a_matrix_coords[f"initial_concentration_{megacomplex.label}"] = (
        species_name,
        initial_concentration,
    )
    a_matrix_name = f"a_matrix_{megacomplex.label}"
    a_matrix = xr.DataArray(a_matrix, coords=a_matrix_coords, dims=(component_name, species_name))

    to_species_name = f"to_species_{megacomplex.label}"
    from_species_name = f"from_species_{megacomplex.label}"
    k_matrix_name = f"k_matrix_{megacomplex.label}"
    k_matrix = xr.DataArray(
        matrix, coords=[(to_species_name, species), (from_species_name, species)]
    )

    k_matrix_reduced_name = f"k_matrix_reduced_{megacomplex.label}"
    k_matrix_reduced = xr.DataArray(
        matrix_reduced, coords=[(to_species_name, species), (from_species_name, species)]
    )

    dataset[das_name] = das
    dataset[a_matrix_name] = a_matrix
    dataset[k_matrix_name] = k_matrix
    dataset[k_matrix_reduced_name] = k_matrix_reduced


def retrieve_irf(dataset_model: DatasetModel, dataset: xr.Dataset, global_dimension: str):
    if not isinstance(dataset_model.irf, IrfMultiGaussian) or "irf" in dataset:
        return

    irf = dataset_model.irf
    model_dimension = get_dataset_model_model_dimension(dataset_model)

    dataset["irf"] = (
        (model_dimension),
        irf.calculate(
            index=0,
            global_axis=dataset.coords[global_dimension].values,
            model_axis=dataset.coords[model_dimension].values,
        ).data,
    )

    center = irf.center if isinstance(irf.center, list) else [irf.center]
    width = irf.width if isinstance(irf.width, list) else [irf.width]
    dataset["irf_center"] = ("irf_nr", center) if len(center) > 1 else center[0]
    dataset["irf_width"] = ("irf_nr", width) if len(width) > 1 else width[0]

    if irf.shift is not None:
        dataset["irf_shift"] = (global_dimension, [center[0] - p.value for p in irf.shift])

    if isinstance(irf, IrfSpectralMultiGaussian) and irf.dispersion_center:
        dataset["irf_center_location"] = (
            ("irf_nr", global_dimension),
            irf.calculate_dispersion(dataset.coords["spectral"].values),
        )
        # center_dispersion_1 for backwards compatibility (0.3-0.4.1)
        dataset["center_dispersion_1"] = dataset["irf_center_location"].sel(irf_nr=0)
