import sys

import click
from click import echo

import glotaran as gta

from . import util


@click.group()
@click.version_option(version=gta.__version__)
def glotaran():
    pass


@click.command(short_help='Validates a model file.')
@click.option('--parameter', '-p', default=None, help='(optional) Path to parameter file.')
@click.argument("model")
def validate(parameter: str, model: str):
    """Validates a model file and optionally a parameter file."""
    echo(f"Validating model in file: '{model}'")

    model = util.load_model_file(model, verbose=True)

    if parameter is not None:
        echo(f"Validating parameter in file: '{parameter}'")
        parameter = util.load_parameter_file(parameter, verbose=True)

    echo(model.validate(parameter=parameter))


@click.command(name='print', short_help="Prints a model as markdown.")
@click.option('--parameter', '-p', default=None, help='(optional) Path to parameter file.')
@click.argument("model")
def print_model(parameter: str, model: str):
    """Parses a model file and prints the result as a Markdown formatted string. A parameter file
    can be included optionally."""

    model = util.load_model_file(model)

    if parameter is not None:
        parameter = util.load_parameter_file(parameter)

    echo(model.markdown(parameter=parameter))


@click.command(
    name='optimize',
    short_help="Prints a model as markdown.",
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.option('--parameter', '-p', help='Path to parameter file.')
@click.option('--out', '-o', default=None, help='(optional) Path to an output directory.')
@click.option('--nfev', '-n', default=None,
              help='(optional) Maximum number of function evaluations.', show_default=True)
@click.option('--nnls', is_flag=True,
              help='Use non-negative least squares.', show_default=True)
@click.argument("model")
@click.pass_context
def optimize(ctx, parameter: str, out: str, nfev: int, nnls: bool, model: str):
    """Parses a model file and prints the result as a Markdown formatted string. A parameter file
    can be included optionally."""
    echo(f"Optimizing model in file: '{model}'")

    model = util.load_model_file(model, verbose=True)

    if parameter is not None:
        echo(f"Loading parameter from file: '{parameter}'")
    parameter = util.load_parameter_file(parameter, verbose=True)

    echo(model.validate(parameter=parameter))
    if not model.valid():
        sys.exit(1)

    dataset_files = {arg.split(':')[0]: arg.split(':')[1] for arg in ctx.args}
    datasets = {}

    for label in model.dataset:
        if label not in dataset_files:
            echo(f"Missing dataset for '{label}'", err=True)
            sys.exit(1)
        path = dataset_files[label]
        echo(f"Loading dataset '{label}' from file '{path}'")
        datasets[label] = util.load_dataset(path)

    echo('Starting optimization.')
    echo(f"Use NNLS: {nnls}")
    echo(f"Max Nr Function Evaluations: {nfev}")
    try:
        result = model.optimize(parameter, datasets, max_nfev=int(nfev), nnls=nnls)
        echo('Optimization successfull.')
        echo(result.markdown(with_model=False))
        echo('Optimized Parameter:')
        echo(parameter.markdown())
    except Exception as e:
        echo(f"An error occured during optimization: \n\n{e}", err=True)
        sys.exit(1)

    if out is not None:
        try:
            echo(f"Saving result to '{out}'")
            paths = result.save(out)
            echo(f"File saving successfull, the follwing files have been written:\n")
            for p in paths:
                echo(f"* {p}")
        except Exception as e:
            echo(f"An error occured during optimization: \n\n{e}", err=True)
            sys.exit(1)

    echo('All done, have a nice day!')


glotaran.add_command(validate)
glotaran.add_command(print_model)
glotaran.add_command(optimize)

if __name__ == '__main__':
    glotaran()
