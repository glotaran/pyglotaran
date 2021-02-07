from __future__ import annotations

import xarray as xr

from glotaran.analysis.problem import Problem

from .irf import IrfMultiGaussian


def finalize_kinetic_image_result(model, problem: Problem, data: dict[str, xr.Dataset]):

    for label, dataset in data.items():

        dataset_descriptor = problem.filled_dataset_descriptors[label]
        if not dataset_descriptor.get_k_matrices():
            continue

        retrieve_species_assocatiated_data(problem.model, dataset, dataset_descriptor, "images")
        retrieve_decay_assocatiated_data(problem.model, dataset, dataset_descriptor, "images")

        if dataset_descriptor.baseline:
            dataset["baseline"] = dataset.clp.sel(clp_label=f"{dataset_descriptor.label}_baseline")

        retrieve_irf(problem.model, dataset, dataset_descriptor, "images")


def retrieve_species_assocatiated_data(model, dataset, dataset_descriptor, name):
    compartments = dataset_descriptor.initial_concentration.compartments

    compartments = [c for c in compartments if c in dataset_descriptor.compartments()]

    dataset.coords["species"] = compartments
    dataset[f"species_associated_{name}"] = (
        (
            model.global_dimension,
            "species",
        ),
        dataset.clp.sel(clp_label=compartments),
    )

    if len(dataset.matrix.shape) == 3:
        #  index dependent
        dataset["species_concentration"] = (
            (
                model.global_dimension,
                model.model_dimension,
                "species",
            ),
            dataset.matrix.sel(clp_label=compartments).values,
        )
    else:
        #  index independent
        dataset["species_concentration"] = (
            (
                model.model_dimension,
                "species",
            ),
            dataset.matrix.sel(clp_label=compartments).values,
        )


def retrieve_decay_assocatiated_data(model, dataset, dataset_descriptor, name):
    # get_das
    all_das = []
    all_a_matrix = []
    all_k_matrix = []
    all_k_matrix_reduced = []
    all_das_labels = []
    for megacomplex in dataset_descriptor.megacomplex:

        k_matrix = megacomplex.full_k_matrix()
        if k_matrix is None:
            continue

        compartments = dataset_descriptor.initial_concentration.compartments
        compartments = [c for c in compartments if c in k_matrix.involved_compartments()]

        matrix = k_matrix.full(compartments)
        matrix_reduced = k_matrix.reduced(compartments)
        a_matrix = k_matrix.a_matrix(dataset_descriptor.initial_concentration)
        rates = k_matrix.rates(dataset_descriptor.initial_concentration)
        lifetimes = 1 / rates

        das = dataset[f"species_associated_{name}"].sel(species=compartments).values @ a_matrix.T

        component_coords = {"rate": ("component", rates), "lifetime": ("component", lifetimes)}

        das_coords = component_coords.copy()
        das_coords[model.global_dimension] = dataset.coords[model.global_dimension]
        all_das_labels.append(megacomplex.label)
        all_das.append(
            xr.DataArray(das, dims=(model.global_dimension, "component"), coords=das_coords)
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


def retrieve_irf(model, dataset, dataset_descriptor, name):

    irf = dataset_descriptor.irf

    if isinstance(irf, IrfMultiGaussian):
        index = dataset.coords[model.global_dimension][0].values
        dataset["irf"] = ((model.model_dimension), irf.calculate(index, dataset.coords["time"]))
