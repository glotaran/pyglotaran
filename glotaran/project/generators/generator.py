"""The glotaran generator module."""
from __future__ import annotations

from typing import Callable

from yaml import dump

from glotaran.model import Model


def _generate_decay_model(nr_compartments: int, irf: bool, decay_type: str) -> dict:
    """Generate a decay model dictionary.

    Parameters
    ----------
    nr_compartments : int
        The number of compartments.
    irf : bool
        Whether to add a gaussian irf.
    decay_type : str
        The dype of the decay

    Returns
    -------
    dict :
        The generated model dictionary.
    """
    compartments = [f"species_{i+1}" for i in range(nr_compartments)]
    rates = [f"decay.species_{i+1}" for i in range(nr_compartments)]
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
    if irf:
        model["dataset"]["dataset_1"]["irf"] = "gaussian_irf"  # type:ignore[index]
        model["irf"] = {
            "gaussian_irf": {"type": "gaussian", "center": "irf.center", "width": "irf.width"},
        }
    return model


def generate_parallel_model(nr_compartments: int = 1, irf: bool = False) -> dict:
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
    return _generate_decay_model(nr_compartments, irf, "parallel")


def generate_sequential_model(nr_compartments: int = 1, irf: bool = False) -> dict:
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
    return _generate_decay_model(nr_compartments, irf, "sequential")


generators: dict[str, Callable] = {
    "decay-parallel": generate_parallel_model,
    "decay-sequential": generate_sequential_model,
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
    return dump(model)
