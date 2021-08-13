from __future__ import annotations

import sys
from typing import Iterable

import click
from click import echo
from click import prompt

from glotaran.io import ProjectIoInterface
from glotaran.io import load_dataset
from glotaran.io import load_model
from glotaran.io import load_parameters
from glotaran.io import load_scheme
from glotaran.plugin_system.base_registry import methods_differ_from_baseclass_table
from glotaran.plugin_system.project_io_registration import get_project_io
from glotaran.plugin_system.project_io_registration import known_project_formats


def signature_analysis(cmd):
    cmd = click.option(
        "--model_file",
        "-m",
        default=None,
        type=click.Path(exists=True, dir_okay=False),
        help="Path to model file.",
    )(cmd)
    cmd = click.option(
        "--parameters_file",
        "-p",
        default=None,
        type=click.Path(exists=True, dir_okay=False),
        help="(optional) Path to parameter file.",
    )(cmd)
    cmd = click.argument(
        "scheme_file", type=click.Path(exists=True, dir_okay=False), required=False
    )(cmd)
    return cmd


def _load_file(filename, loader, name, verbose):
    try:
        if verbose:
            echo(f"Loading {name} file '{filename}'... ", nl=False)
        result = loader(filename)
        if verbose:
            echo("Success")
        return result
    except Exception as e:
        if verbose:
            echo(message="Error", err=True)
        else:
            echo(message=f"Error parsing {name} file: \n", err=True)
        echo(message=e, err=True)
        sys.exit(1)


def load_scheme_file(filename, verbose=False):
    return _load_file(
        filename, lambda file: load_scheme(file, format_name="yml"), "scheme", verbose
    )


def load_model_file(filename, verbose=False):
    return _load_file(filename, lambda file: load_model(file, format_name="yml"), "model", verbose)


def load_parameter_file(filename, fmt=None, verbose=False):
    def loader(filename):
        return load_parameters(filename, format_name=fmt)

    return _load_file(filename, loader, "parameter", verbose)


def load_dataset_file(filename, fmt=None, verbose=False):
    def loader(filename):
        return load_dataset(filename, format_name=fmt)

    return _load_file(filename, loader, "parameter", verbose)


def select_name(filename, dataset):

    names = list(dataset)
    echo(f"\nDataset names in in '{filename}':\n")
    for i, n in enumerate(names):
        echo(f"* [{i}] {n}")

    n = prompt(
        "\n Please select a name to export",
        type=click.IntRange(min=0, max=len(names) - 1),
        show_choices=True,
    )
    return names[n]


def select_data(data, dim, selection):
    try:
        [float(i) for i in data.coords[dim].values]
        numeric = True
    except ValueError:
        numeric = False
    method = None
    if numeric:
        try:
            if isinstance(selection, tuple):
                selection = tuple(float(c) for c in selection)
            elif isinstance(selection, list):
                selection = [float(c) for c in selection]
            else:
                selection = float(selection)
        except ValueError:
            raise ValueError(f"Error: Selection '{selection}' is not numeric")
        method = "nearest"
    if isinstance(selection, tuple):
        min = selection[0]
        max = selection[1]
        selection = data.coords[dim]
        selection = selection.where(selection >= min, drop=True)
        selection = selection.where(selection <= max, drop=True)
    return data.sel({dim: selection}, method=method)


def write_data(data, out):
    df = data.to_dataframe()
    if len(data.dims) == 2:
        df = df.reset_index().pivot(index=data.dims[0], columns=data.dims[1], values=data.name)
    df.to_csv(out)


def project_io_list_supporting_plugins(
    method_name: str, block_list: Iterable[str] | None = None
) -> Iterable[str]:
    """List all project-io plugin that implement ``method_name``.

    Parameters
    ----------
    method_name: str
        Name of the method which should be supported.
    block_list: Iterable[str]
        Iterable of plugin names which should be omitted.
    """
    if block_list is None:
        block_list = []
    support_table = methods_differ_from_baseclass_table(
        method_names=method_name,
        plugin_registry_keys=known_project_formats(full_names=False),
        get_plugin_function=get_project_io,
        base_class=ProjectIoInterface,
    )
    support_table = filter(lambda entry: entry[1], support_table)
    supporting_list: Iterable[str] = (entry[0].replace("`", "") for entry in support_table)
    return list(filter(lambda entry: entry not in block_list, supporting_list))


class ValOrRangeOrList(click.ParamType):
    name = "number or range or list"

    def convert(self, value, param, ctx):
        if value[0] == "(" and value[-1] == ")":
            split = value[1:-1].split(",")
            if len(split) != 2:
                self.fail(f"Malformed range: '{value}'")
            return (split[0].strip(), split[1].strip())
        if value[0] == "[" and value[-1] == "]":
            split = value[1:-1].split(",")
            return [s.strip() for s in split]
        return value


VALORRANGEORLIST = ValOrRangeOrList()
