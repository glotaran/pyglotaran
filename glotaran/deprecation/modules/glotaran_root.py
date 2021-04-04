"""Deprecated attributes from ``glotaran.__init__`` which are removed."""
from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.deprecation import deprecate
from glotaran.io import load_model
from glotaran.io import load_parameters

if TYPE_CHECKING:
    from glotaran.model import Model
    from glotaran.parameter import ParameterGroup


@deprecate(
    deprecated_qual_name_usage="glotaran.read_model_from_yaml(model_yml_str)",
    new_qual_name_usage='glotaran.io.load_model(model_yml_str, format_name="yml_str")',
    to_be_removed_in_version="0.6.0",
)
def read_model_from_yaml(model_yml_str: str) -> Model:
    """Parse ``yaml`` string to :class:`Model`.

    Warning
    -------
    Deprecated use ``glotaran.io.load_model(model_yml_str, format_name="yml_str")``
    instead.

    Parameters
    ----------
    model_yml_str : str
        Model spec description in yaml.

    Returns
    -------
    Model
        Model described in ``model_yml_str``.
    """
    return load_model(model_yml_str, format_name="yml_str")


@deprecate(
    deprecated_qual_name_usage="glotaran.read_model_from_yaml_file(model_file)",
    new_qual_name_usage="glotaran.io.load_model(model_file)",
    to_be_removed_in_version="0.6.0",
)
def read_model_from_yaml_file(model_file: str) -> Model:
    """Parse ``model.yaml`` file to :class:`Model`.

    Warning
    -------
    Deprecated use ``glotaran.io.load_model(model_file)`` instead.

    Parameters
    ----------
    model_file : str
        File with model spec description as yaml.

    Returns
    -------
    Model
        Model described in ``model_file``.
    """
    return load_model(model_file)


@deprecate(
    deprecated_qual_name_usage="glotaran.read_parameters_from_csv_file(parameters_file)",
    new_qual_name_usage="glotaran.io.load_parameters(parameters_file)",
    to_be_removed_in_version="0.6.0",
)
def read_parameters_from_csv_file(parameters_file: str) -> ParameterGroup:
    """Parse ``parameters_file`` to :class:`ParameterGroup`.

    Warning
    -------
    Deprecated use ``glotaran.io.load_parameters(parameters_file)`` instead.

    Parameters
    ----------
    parameters_file : str
        File with parameters in csv.

    Returns
    -------
    ParameterGroup
        ParameterGroup described in ``parameters_file``.
    """
    return load_parameters(parameters_file)


@deprecate(
    deprecated_qual_name_usage="glotaran.read_parameters_from_yaml(parameters_yml_str)",
    new_qual_name_usage="glotaran.io.load_model(parameters_yml_str)",
    to_be_removed_in_version="0.6.0",
)
def read_parameters_from_yaml(parameters_yml_str: str) -> ParameterGroup:
    """Parse ``yaml`` string to :class:`ParameterGroup`.

    Warning
    -------
    Deprecated use ``glotaran.io.load_parameters(parameters_yml_str, format_name="yml_str")``
    instead.

    Parameters
    ----------
    parameters_yml_str : str
        PArameter spec description in yaml.

    Returns
    -------
    ParameterGroup
        ParameterGroup described in ``parameters_yml_str``.
    """
    return load_parameters(parameters_yml_str, format_name="yml_str")


@deprecate(
    deprecated_qual_name_usage="glotaran.read_parameters_from_yaml_file(parameters_file)",
    new_qual_name_usage="glotaran.io.load_parameters(parameters_file)",
    to_be_removed_in_version="0.6.0",
)
def read_parameters_from_yaml_file(parameters_file: str) -> ParameterGroup:
    """Parse ``parameters_file`` to :class:`ParameterGroup`.

    Warning
    -------
    Deprecated use ``glotaran.io.load_parameters(parameters_file)`` instead.

    Parameters
    ----------
    parameters_file : str
        File with parameters in yaml.

    Returns
    -------
    ParameterGroup
        ParameterGroup described in ``parameters_file``.
    """
    return load_parameters(parameters_file)
