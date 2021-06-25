from __future__ import annotations

import xarray as xr

from glotaran.analysis.problem import Problem
from glotaran.builtin.models.kinetic_image.irf import IrfMultiGaussian
from glotaran.builtin.models.kinetic_image.kinetic_baseline_megacomplex import (
    KineticBaselineMegacomplex,
)
from glotaran.builtin.models.kinetic_image.kinetic_decay_megacomplex import KineticDecayMegacomplex


def finalize_kinetic_image_result(model, problem: Problem, data: dict[str, xr.Dataset]):

    for label, dataset in data.items():

        dataset_model = problem.filled_dataset_descriptors[label]

        retrieve_species_associated_data(problem.model, dataset, dataset_model, "images")
        retrieve_decay_associated_data(problem.model, dataset, dataset_model, "images")

        if any(
            isinstance(megacomplex, KineticBaselineMegacomplex)
            for megacomplex in dataset_model.megacomplex
        ):
            dataset["baseline"] = dataset.clp.sel(clp_label=f"{dataset_model.label}_baseline")

        retrieve_irf(problem.model, dataset, dataset_model, "images")


def retrieve_species_associated_data(model, dataset, dataset_model, name):
    compartments = dataset_model.initial_concentration.compartments
    global_dimension = dataset_model.get_global_dimension()
    model_dimension = dataset_model.get_model_dimension()

    dataset.coords["species"] = compartments
    dataset[f"species_associated_{name}"] = (
        (
            global_dimension,
            "species",
        ),
        dataset.clp.sel(clp_label=compartments).data,
    )

    if len(dataset.matrix.shape) == 3:
        #  index dependent
        dataset["species_concentration"] = (
            (
                global_dimension,
                model_dimension,
                "species",
            ),
            dataset.matrix.sel(clp_label=compartments).values,
        )
    else:
        #  index independent
        dataset["species_concentration"] = (
            (
                model_dimension,
                "species",
            ),
            dataset.matrix.sel(clp_label=compartments).values,
        )


def retrieve_decay_associated_data(model, dataset, dataset_model, name):
    # get_das
    all_das = []
    all_a_matrix = []
    all_k_matrix = []
    all_k_matrix_reduced = []
    all_das_labels = []

    global_dimension = dataset_model.get_global_dimension()

    for megacomplex in dataset_model.megacomplex:

        if isinstance(megacomplex, KineticDecayMegacomplex):
            k_matrix = megacomplex.full_k_matrix()

            compartments = dataset_model.initial_concentration.compartments
            compartments = [c for c in compartments if c in k_matrix.involved_compartments()]

            matrix = k_matrix.full(compartments)
            matrix_reduced = k_matrix.reduced(compartments)
            a_matrix = k_matrix.a_matrix(dataset_model.initial_concentration)
            rates = k_matrix.rates(dataset_model.initial_concentration)
            lifetimes = 1 / rates

            das = (
                dataset[f"species_associated_{name}"].sel(species=compartments).values @ a_matrix.T
            )

            component_coords = {"rate": ("component", rates), "lifetime": ("component", lifetimes)}

            das_coords = component_coords.copy()
            das_coords[global_dimension] = dataset.coords[global_dimension]
            all_das_labels.append(megacomplex.label)
            all_das.append(
                xr.DataArray(das, dims=(global_dimension, "component"), coords=das_coords)
            )
            a_matrix_coords = component_coords.copy()
            a_matrix_coords["species"] = compartments
            all_a_matrix.append(
                xr.DataArray(a_matrix, coords=a_matrix_coords, dims=("component", "species"))
            )
            all_k_matrix.append(
                xr.DataArray(
                    matrix, coords=[("to_species", compartments), ("from_species", compartments)]
                )
            )

            all_k_matrix_reduced.append(
                xr.DataArray(
                    matrix_reduced,
                    coords=[("to_species", compartments), ("from_species", compartments)],
                )
            )

    if all_das:
        if len(all_das) == 1:
            dataset[f"decay_associated_{name}"] = all_das[0]
            dataset["a_matrix"] = all_a_matrix[0]
            dataset["k_matrix"] = all_k_matrix[0]
            dataset["k_matrix_reduced"] = all_k_matrix_reduced[0]

        else:
            for i, das_label in enumerate(all_das_labels):
                dataset[f"decay_associated_{name}_{das_label}"] = all_das[i].rename(
                    component=f"component_{das_label}"
                )
                dataset[f"a_matrix_{das_label}"] = all_a_matrix[i].rename(
                    component=f"component_{das_label}"
                )
                dataset[f"k_matrix_{das_label}"] = all_k_matrix[i]
                dataset[f"k_matrix_reduced_{das_label}"] = all_k_matrix_reduced[i]


def retrieve_irf(model, dataset, dataset_model, name):

    irf = dataset_model.irf
    global_dimension = dataset_model.get_global_dimension()
    model_dimension = dataset_model.get_model_dimension()

    if isinstance(irf, IrfMultiGaussian):
        dataset["irf"] = (
            (model_dimension),
            irf.calculate(
                index=0,
                global_axis=dataset.coords[global_dimension],
                model_axis=dataset.coords[model_dimension],
            ).data,
        )
