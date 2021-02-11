import sys

import click
from click import echo
from click import prompt

import glotaran as gta


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
    return _load_file(filename, gta.analysis.scheme.Scheme.from_yaml_file, "scheme", verbose)


def load_model_file(filename, verbose=False):
    return _load_file(filename, gta.read_model_from_yaml_file, "model", verbose)


def load_parameter_file(filename, fmt=None, verbose=False):
    def loader(filename):
        return gta.parameter.ParameterGroup.from_file(filename, fmt=fmt)

    return _load_file(filename, loader, "parameter", verbose)


def load_dataset_file(filename, fmt=None, verbose=False):
    def loader(filename):
        return gta.io.read_data_file(filename, fmt=fmt)

    return _load_file(filename, loader, "parameter", verbose)


def select_name(filename, dataset):

    names = [n for n in dataset]
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
