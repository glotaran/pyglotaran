import sys
from pathlib import Path
import xarray as xr

import click
from click import echo, prompt

import glotaran as gta


def load_model_file(filename, verbose=False):
    try:
        model = gta.read_model_from_yml_file(filename)
        if verbose:
            echo("Model parsing successfull.")
        return model
    except Exception as e:
        echo(message=f"Error parsing model file: \n\n{e}", err=True)
        sys.exit(1)


def load_parameter_file(filename, verbose=False):
    try:
        parameter = gta.read_parameter_from_yml_file(filename)
        if verbose:
            echo("Parameter parsing successfull.")
        return parameter
    except Exception as e:
        echo(message=f"Error parsing parameter file: \n\n{e}", err=True)
        sys.exit(1)


file_readers = {
    'ascii': gta.io.read_ascii_time_trace,
    'sdt': gta.io.read_sdt_data,
    'nc': xr.open_dataset,
}


def load_dataset(path, dtype=None):
    path = Path(path)
    if dtype is None:
        dtype = path.suffix[1:]
    if dtype not in file_readers:
        echo(f"Unknown file type '{dtype}'."
             f"Supported file types are {list(file_readers.keys())}.", err=True)
        sys.exit(1)

    try:
        dataset = file_readers[dtype](path)
        echo("Dataset loading successfull.")
        return dataset
    except Exception as e:
        echo(message=f"Error loading dataset file: \n\n{e}", err=True)
        sys.exit(1)


def select_name(filename, dataset):

    names = [n for n in dataset]
    echo(f"\nDataset names in in '{filename}':\n")
    for i, n in enumerate(names):
        echo(f"* [{i}] {n}")

    n = prompt("\n Please select a name to export",
               type=click.IntRange(min=0, max=len(names)-1),
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
        method = 'nearest'
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
    name = 'number or range or list'

    def convert(self, value, param, ctx):
        if value[0] == '(' and value[-1] == ')':
            split = value[1:-1].split(',')
            if not len(split) == 2:
                self.fail(f"Malformed range: '{value}'")
            return (split[0].strip(), split[1].strip())
        if value[0] == '[' and value[-1] == ']':
            split = value[1:-1].split(',')
            return [s.strip() for s in split]
        return value


VALORRANGEORLIST = ValOrRangeOrList()
