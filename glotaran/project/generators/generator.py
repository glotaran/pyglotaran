"""The glotaran generator module."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any
from typing import TypedDict
from typing import cast

from glotaran.builtin.io.yml.utils import write_dict
from glotaran.builtin.megacomplexes.decay import DecayParallelMegacomplex
from glotaran.builtin.megacomplexes.decay import DecaySequentialMegacomplex
from glotaran.builtin.megacomplexes.spectral import SpectralMegacomplex
from glotaran.model import Model


def _generate_decay_model(
    *, nr_compartments: int, irf: bool, spectral: bool, decay_type: str
) -> dict[str, Any]:
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
    dict[str, Any] :
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


def generate_parallel_decay_model(
    *, nr_compartments: int = 1, irf: bool = False
) -> dict[str, Any]:
    """Generate a parallel decay model dictionary.

    Parameters
    ----------
    nr_compartments : int
        The number of compartments.
    irf : bool
        Whether to add a gaussian irf.

    Returns
    -------
    dict[str, Any] :
        The generated model dictionary.
    """
    return _generate_decay_model(
        nr_compartments=nr_compartments, irf=irf, spectral=False, decay_type="parallel"
    )


def generate_parallel_spectral_decay_model(
    *, nr_compartments: int = 1, irf: bool = False
) -> dict[str, Any]:
    """Generate a parallel spectral decay model dictionary.

    Parameters
    ----------
    nr_compartments : int
        The number of compartments.
    irf : bool
        Whether to add a gaussian irf.

    Returns
    -------
    dict[str, Any] :
        The generated model dictionary.
    """
    return _generate_decay_model(
        nr_compartments=nr_compartments, irf=irf, spectral=True, decay_type="parallel"
    )


def generate_sequential_decay_model(nr_compartments: int = 1, irf: bool = False) -> dict[str, Any]:
    """Generate a sequential decay model dictionary.

    Parameters
    ----------
    nr_compartments : int
        The number of compartments.
    irf : bool
        Whether to add a gaussian irf.

    Returns
    -------
    dict[str, Any] :
        The generated model dictionary.
    """
    return _generate_decay_model(
        nr_compartments=nr_compartments, irf=irf, spectral=False, decay_type="sequential"
    )


def generate_sequential_spectral_decay_model(
    *, nr_compartments: int = 1, irf: bool = False
) -> dict[str, Any]:
    """Generate a sequential spectral decay model dictionary.

    Parameters
    ----------
    nr_compartments : int
        The number of compartments.
    irf : bool
        Whether to add a gaussian irf.

    Returns
    -------
    dict[str, Any] :
        The generated model dictionary.
    """
    return _generate_decay_model(
        nr_compartments=nr_compartments, irf=irf, spectral=True, decay_type="sequential"
    )


generators: dict[str, Callable] = {
    "decay_parallel": generate_parallel_decay_model,
    "spectral_decay_parallel": generate_parallel_spectral_decay_model,
    "decay_sequential": generate_sequential_decay_model,
    "spectral_decay_sequential": generate_sequential_spectral_decay_model,
}

available_generators: list[str] = list(generators.keys())


class GeneratorArguments(TypedDict, total=False):
    """Arguments used by ``generate_model`` and ``generate_model``.

    Parameters
    ----------
    nr_compartments : int
        The number of compartments.
    irf : bool
        Whether to add a gaussian irf.

    See Also
    --------
    generate_model
    generate_model_yml
    """

    nr_compartments: int
    irf: bool


def generate_model(*, generator_name: str, generator_arguments: GeneratorArguments) -> Model:
    """Generate a model.

    Parameters
    ----------
    generator_name : str
        The generator to use.
    generator_arguments : GeneratorArguments
        Arguments for the generator.

    Returns
    -------
    Model
        The generated model

    See Also
    --------
    generate_parallel_decay_model
    generate_parallel_spectral_decay_model
    generate_sequential_decay_model
    generate_sequential_spectral_decay_model

    Raises
    ------
    ValueError
        Raised when an unknown generator is specified.
    """
    if generator_name not in generators:
        raise ValueError(
            f"Unknown model generator '{generator_name}'. "
            f"Known generators are: {list(generators.keys())}"
        )
    model = generators[generator_name](**generator_arguments)
    return Model.create_class_from_megacomplexes(
        [DecayParallelMegacomplex, DecaySequentialMegacomplex, SpectralMegacomplex]
    )(**model)


def generate_model_yml(*, generator_name: str, generator_arguments: GeneratorArguments) -> str:
    """Generate a model as yml string.

    Parameters
    ----------
    generator_name : str
        The generator to use.
    generator_arguments : GeneratorArguments
        Arguments for the generator.

    Returns
    -------
    str
        The generated model yml string.

    See Also
    --------
    generate_parallel_decay_model
    generate_parallel_spectral_decay_model
    generate_sequential_decay_model
    generate_sequential_spectral_decay_model

    Raises
    ------
    ValueError
        Raised when an unknown generator is specified.
    """
    if generator_name not in generators:
        raise ValueError(
            f"Unknown model generator '{generator_name}'. "
            f"Known generators are: {list(generators.keys())}"
        )
    model = generators[generator_name](**generator_arguments)
    return cast(str, write_dict(model))
