import typing
import sys

import click
from click import confirm, echo, pause, prompt

import glotaran as gta

from . import util


@click.group()
@click.version_option(version=gta.__version__)
def glotaran():
    pass


@click.command(short_help='Validates a model file.')
@click.option('--parameter', '-p', default=None, type=click.Path(exists=True, dir_okay=False),
              help='(optional) Path to parameter file.')
@click.argument("model", type=click.Path(exists=True, dir_okay=False))
def validate(parameter: str, model: str):
    """Validates a model file and optionally a parameter file."""
    echo(f"Validating model in file: '{model}'")

    model = util.load_model_file(model, verbose=True)

    if parameter is not None:
        echo(f"Validating parameter in file: '{parameter}'")
        parameter = util.load_parameter_file(parameter, verbose=True)

    echo(model.validate(parameter=parameter))


@click.command(name='print', short_help="Prints a model as markdown.")
@click.option('--parameter', '-p', default=None, type=click.Path(exists=True, dir_okay=False),
              help='(optional) Path to parameter file.')
@click.argument("model", type=click.Path(exists=True, dir_okay=False))
def print_model(parameter: str, model: str):
    """Parses a model file and prints the result as a Markdown formatted string. A parameter file
    can be included optionally."""

    model = util.load_model_file(model)

    if parameter is not None:
        parameter = util.load_parameter_file(parameter)

    echo(model.markdown(parameter=parameter))


@click.command(
    name='optimize',
    short_help="Optimizes a model.",
)
@click.option('--parameter', '-p', required=True, type=click.Path(exists=True, dir_okay=False),
              help='Path to parameter file.')
@click.option('--datatype', '-dtype', default=None,
              type=click.Choice([k for k in util.file_readers]),
              help='The input format of the data. Will be infered from extension if not set.')
@click.option('--data', '-d', required=True, multiple=True,
              type=(str, click.Path(exists=True, dir_okay=False)),
              help="Path to a dataset in the form '--data DATASET_LABEL PATH_TO_DATA'")
@click.option('--out', '-o', default=None, type=click.Path(file_okay=False),
              help='Path to an output directory.', show_default=True)
@click.option('--nfev', '-n', default=None, type=click.IntRange(min=1),
              help='Maximum number of function evaluations.', show_default=True)
@click.option('--nnls', is_flag=True,
              help='Use non-negative least squares.')
@click.option('--yes', '-y', is_flag=True,
              help="Don't ask for confirtmation.")
@click.argument("model", type=click.Path(exists=True, dir_okay=False))
def optimize(parameter: str, datatype: str, data: typing.List[str],
             out: str, nfev: int, nnls: bool, yes: bool, model: str):
    """Optimizes a model."""
    echo(f"Optimizing model in file: '{model}'")

    model = util.load_model_file(model, verbose=True)

    if parameter is not None:
        echo(f"Loading parameter from file: '{parameter}'")
    parameter = util.load_parameter_file(parameter, verbose=True)

    echo(model.validate(parameter=parameter))
    if not model.valid():
        sys.exit(1)

    dataset_files = {arg[0]: arg[1] for arg in data}
    datasets = {}
    for label in model.dataset:
        if label not in dataset_files:
            echo(f"Missing dataset for '{label}'", err=True)
            sys.exit(1)
        path = dataset_files[label]
        echo(f"Loading dataset '{label}' from file '{path}'")
        datasets[label] = util.load_dataset(path, dtype=datatype)

    echo('Starting optimization.')
    echo(f"Use NNLS: {nnls}")
    echo(f"Max Nr Function Evaluations: {nfev}")
    if yes or click.confirm('Do you want to start optimization?', abort=True, default=True):
        try:
            result = model.optimize(parameter, datasets, max_nfev=nfev, nnls=nnls)
            echo('Optimization successfull.')
            echo(result.markdown(with_model=False))
            echo('Optimized Parameter:')
            echo(parameter.markdown())
        except Exception as e:
            echo(f"An error occured during optimization: \n\n{e}", err=True)
            sys.exit(1)

        if out is not None:
            try:
                echo(f"Saving directory is '{out}'")
                if yes or click.confirm('Do you want to save the data?', default=True):
                    paths = result.save(out)
                    echo(f"File saving successfull, the follwing files have been written:\n")
                    for p in paths:
                        echo(f"* {p}")
            except Exception as e:
                echo(f"An error occured during optimization: \n\n{e}", err=True)
                sys.exit(1)

        echo('All done, have a nice day!')


@click.command(name='export', short_help="Exports data from netCDF4 to ASCII.")
@click.option('--name', '-n', default=None, type=str, show_default=True,
              help='Name of the datagroup to export.')
@click.option('--select', '-s', default=None, type=(str, util.VALORRANGEORLIST), show_default=True,
              multiple=True, help="Selection of data. Example --select DIMENSION_NAME VALUE "
              "where VALUE can be a scalar, a list like '[V1,V2,V3]' or a tuple like '(MIN,MAX)'"
              )
@click.option('--out', '-o', default=None, type=click.Path(dir_okay=False), show_default=True,
              help='Path to the output file.')
@click.argument("filename", type=click.Path(exists=True))
def export(filename: str, select, out: str, name: str):
    """Exports data from netCDF4 to ascii."""

    echo(f"Opening dataset at {filename}")

    dataset = util.load_dataset(filename, dtype='nc')

    if name is None:
        name = util.select_name(filename, dataset)

    stop = False
    data = dataset[name]

    for sel in select:
        data = util.select_data(data, sel[0], sel[1])

    if out is not None:
        if len(data.shape) > 2:
            pause('Cannot export data with more than 2 dimensions to ASCII')
        else:
            util.write_data(data, out)
            stop = True

    while not stop:

        echo(f"Selected dataset '{name}'.")
        echo(f"\nDataset Content\n\n{data}\n")

        choice = prompt('How to proceed', default='exit',
                        type=click.Choice(["exit", "save", "select", "change", "help"]))
        if choice == 'exit':
            stop = True
        elif choice == 'change':
            name = util.select_name(filename, dataset)
            data = dataset[name]
        elif choice == 'select':
            cont = True
            while cont:
                dims = [d for d in data.dims]
                choice = prompt('Please select a dimension or action', default='back',
                                type=click.Choice(dims + ['reset', 'back']))
                if choice == 'back':
                    cont = False
                elif choice == 'reset':
                    data = dataset[name]
                else:
                    dim = choice
                    choice = prompt(
                        "Please select a value. Type 2 values sperated by ',' to select a range.",
                        default='back', type=util.VALORRANGEORLIST
                    )
                    if not choice == 'back':
                        try:
                            data = util.select_data(data, dim, choice)
                        except ValueError as e:
                            echo(e, err=True)
                            continue

        elif choice == 'save':
            if len(data.shape) > 2:
                pause('Cannot export data with more than 2 dimensions to ASCII')
            else:
                cont = True
                change = False
                while cont:
                    if out is None or change:
                        out = prompt('Please type in a name for the output file.',
                                     default='out.csv', type=click.Path(dir_okay=False))
                        change = False
                    choice = prompt(f"The filepath of the ouput file is '{out}'. Save the data?",
                                    default='yes', type=click.Choice(['yes', 'no', 'change']))
                    if choice == 'change':
                        change = True
                    else:
                        cont = False
                        if choice == 'yes':
                            util.write(data, out)
    echo('Good-bye, have a nice day!')


glotaran.add_command(validate)
glotaran.add_command(print_model)
glotaran.add_command(optimize)
glotaran.add_command(export)

if __name__ == '__main__':
    glotaran()
