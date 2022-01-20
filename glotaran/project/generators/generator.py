"""The glotaran generator module."""
from __future__ import annotations

from typing import Callable

from glotaran.builtin.io.yml.yml import write_dict
from glotaran.model import Model


def _generate_decay_model(
    nr_compartments: int, irf: bool, spectral: bool, decay_type: str
) -> dict:
    """Generate a decay model dictionary.

    Parameters
    ----------
    nr_compartments : int
        The number of compartments.
    irf : bool
        Whether to add a gaussian irf.
    spectral : bool
        Whether to add a spectral model.
    decay_type : str
        The dype of the decay

    Returns
    -------
    dict :
        The generated model dictionary.
    """
    compartments = [f"species_{i+1}" for i in range(nr_compartments)]
    rates = [f"rates.species_{i+1}" for i in range(nr_compartments)]
    model = {
        "megacomplex": {
            f"megacomplex_{decay_type}_decay": {
                "type": f"decay-{decay_type}",
                "compartments": compartments,
                "rates": rates,
            },
        },
        "dataset": {"dataset_1": {"megacomplex": [f"megacomplex_{decay_type}_decay"]}},
    }
    if spectral:
        model["megacomplex"]["megacomplex_spectral"] = {  # type:ignore[index]
            "type": "spectral",
            "shape": {
                compartment: f"shape_species_{i+1}" for i, compartment in enumerate(compartments)
            },
        }
        model["shape"] = {
            f"shape_species_{i+1}": {
                "type": "gaussian",
                "amplitude": f"shapes.species_{i+1}.amplitude",
                "location": f"shapes.species_{i+1}.location",
                "width": f"shapes.species_{i+1}.width",
            }
            for i in range(nr_compartments)
        }
        model["dataset"]["dataset_1"]["global_megacomplex"] = [  # type:ignore[index]
            "megacomplex_spectral"
        ]
    if irf:
        model["dataset"]["dataset_1"]["irf"] = "gaussian_irf"  # type:ignore[index]
        model["irf"] = {
            "gaussian_irf": {"type": "gaussian", "center": "irf.center", "width": "irf.width"},
        }
    return model


def generate_parallel_decay_model(nr_compartments: int = 1, irf: bool = False) -> dict:
    """Generate a parallel decay model dictionary.

    Parameters
    ----------
    nr_compartments : int
        The number of compartments.
    irf : bool
        Whether to add a gaussian irf.

    Returns
    -------
    dict :
        The generated model dictionary.
    """
    return _generate_decay_model(nr_compartments, irf, False, "parallel")


def generate_parallel_spectral_decay_model(nr_compartments: int = 1, irf: bool = False) -> dict:
    """Generate a parallel spectral decay model dictionary.

    Parameters
    ----------
    nr_compartments : int
        The number of compartments.
    irf : bool
        Whether to add a gaussian irf.

    Returns
    -------
    dict :
        The generated model dictionary.
    """
    return _generate_decay_model(nr_compartments, irf, True, "parallel")


def generate_sequential_decay_model(nr_compartments: int = 1, irf: bool = False) -> dict:
    """Generate a sequential decay model dictionary.

    Parameters
    ----------
    nr_compartments : int
        The number of compartments.
    irf : bool
        Whether to add a gaussian irf.

    Returns
    -------
    dict :
        The generated model dictionary.
    """
    return _generate_decay_model(nr_compartments, irf, False, "sequential")


def generate_sequential_spectral_decay_model(nr_compartments: int = 1, irf: bool = False) -> dict:
    """Generate a sequential spectral decay model dictionary.

    Parameters
    ----------
    nr_compartments : int
        The number of compartments.
    irf : bool
        Whether to add a gaussian irf.

    Returns
    -------
    dict :
        The generated model dictionary.
    """
    return _generate_decay_model(nr_compartments, irf, True, "sequential")


generators: dict[str, Callable] = {
    "decay_parallel": generate_parallel_decay_model,
    "spectral_decay_parallel": generate_parallel_spectral_decay_model,
    "decay_sequential": generate_sequential_decay_model,
    "spectral_decay_sequential": generate_sequential_spectral_decay_model,
}

available_generators: list[str] = list(generators.keys())


def generate_model(generator: str, **generator_arguments: dict) -> Model:
    """Generate a model.

    Parameters
    ----------
    generator : str
        The generator to use.
    generator_arguments : dict
        Arguments for the generator.

    Returns
    -------
    Model
        The generated model

    Raises
    ------
    ValueError
        Raised when an unknown generator is specified.
    """
    if generator not in generators:
        raise ValueError(
            f"Unknown model generator '{generator}'. "
            f"Known generators are: {list(generators.keys())}"
        )
    model = generators[generator](**generator_arguments)
    return Model.from_dict(model)


def generate_model_yml(generator: str, **generator_arguments: dict) -> str:
    """Generate a model as yml string.

    Parameters
    ----------
    generator : str
        The generator to use.
    generator_arguments : dict
        Arguments for the generator.

    Returns
    -------
    str
        The generated model yml string.

    Raises
    ------
    ValueError
        Raised when an unknown generator is specified.
    """
    if generator not in generators:
        raise ValueError(
            f"Unknown model generator '{generator}'. "
            f"Known generators are: {list(generators.keys())}"
        )
    model = generators[generator](**generator_arguments)
    yml: str = write_dict(model)  # type:ignore[assignment]
    return yml
