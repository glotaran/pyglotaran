from __future__ import annotations


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


def generate_sequential_model(nr_species: int = 1):
    species = [f"species_{i}" for i in range(nr_species)]
    initial_concentration_parameters = [
        f"intitial_concentration.species_{i}" for i in range(nr_species)
    ]
    k_matrix = {
        f"(species_{i+2}, species_{i+1})": f"decay.species_{i+1}" for i in range(nr_species - 1)
    }
    k_matrix[f"(species_{nr_species}, species_{nr_species})"] = f"decay.species_{nr_species}"

    return {
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
            }
        },
    }


generators = {"decay-parallel": generate_parallel_model}
generators = {"decay-sequential": generate_parallel_model}
