from __future__ import annotations

from typing import Any
from typing import Callable

from yaml import dump


def generate_parallel_model(nr_species: int = 1):
    species = [f"species_{i+1}" for i in range(nr_species)]
    initial_concentration_parameters = [
        f"intitial_concentration.species_{i+1}" for i in range(nr_species)
    ]
    k_matrix = {
        f"(species_{i+1}, species_{i+1})": f"decay.species_{i+1}" for i in range(nr_species)
    }
    return {
        "initial_concentration": {
            "initial_concentration_dataset_1": {
                "compartments": species,
                "parameters": initial_concentration_parameters,
            },
        },
        "k_matrix": {"k_matrix_parallel": {"matrix": k_matrix}},
        "megacomplex": {
            "megacomplex_parallel_decay": {
                "type": "decay",
                "k_matrix": ["k_matrix_parallel"],
            },
        },
        "dataset": {
            "dataset_1": {
                "initial_concentration": "initial_concentration_dataset_1",
                "megacomplex": ["megacomplex_parallel_decay"],
            }
        },
    }


def generate_sequential_model(nr_species: int = 1, irf: bool = False) -> dict:
    species = [f"species_{i+1}" for i in range(nr_species)]
    initial_concentration_parameters = ["initial_concentration.1"] + [
        "initial_concentration.0" for i in range(1, nr_species)
    ]
    k_matrix = {
        f"(species_{i+2}, species_{i+1})": f"decay.species_{i+1}" for i in range(nr_species - 1)
    }
    k_matrix[f"(species_{nr_species}, species_{nr_species})"] = f"decay.species_{nr_species}"

    model = {
        "initial_concentration": {
            "initial_concentration_dataset_1": {
                "compartments": species,
                "parameters": initial_concentration_parameters,
            },
        },
        "k_matrix": {"k_matrix_sequential": {"matrix": k_matrix}},
        "megacomplex": {
            "megacomplex_parallel_decay": {
                "type": "decay",
                "k_matrix": ["k_matrix_sequential"],
            },
        },
        "dataset": {
            "dataset_1": {
                "initial_concentration": "initial_concentration_dataset_1",
                "megacomplex": ["megacomplex_parallel_decay"],
                "irf": "gaussian_irf" if irf else None,
            }
        },
    }
    if irf:
        model["irf"] = {
            "gaussian_irf": {"type": "gaussian", "center": "irf.center", "width": "irf.width"},
        }
    return model


generators: dict[str, Callable] = {
    "decay-parallel": generate_parallel_model,
    "decay-sequential": generate_sequential_model,
}

available_generators: list[str] = list(generators.keys())


def generate_model_yml(generator: str, **generator_arguments: dict[str, Any]) -> str:
    if generator not in generators:
        raise ValueError(
            f"Unknown model generator '{generator}'. "
            f"Known generators are: {list(generators.keys())}"
        )
    model = generators[generator](**generator_arguments)
    return dump(model)
