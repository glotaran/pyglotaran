import sys
import typing

import click

import glotaran as gta
from glotaran.analysis.optimize import optimize

from . import util


@click.option(
    "--dataformat",
    "-dfmt",
    default=None,
    type=click.Choice(gta.io.reader.known_reading_formats.keys()),
    help="The input format of the data. Will be inferred from extension if not set.",
)
@click.option(
    "--data",
    "-d",
    multiple=True,
    type=(str, click.Path(exists=True, dir_okay=False)),
    help="Path to a dataset in the form '--data DATASET_LABEL PATH_TO_DATA'",
)
@click.option(
    "--out",
    "-o",
    default=None,
    type=click.Path(file_okay=False),
    help="Path to an output directory.",
    show_default=True,
)
@click.option(
    "--nfev",
    "-n",
    default=None,
    type=click.IntRange(min=1),
    help="Maximum number of function evaluations.",
    show_default=True,
)
@click.option("--nnls", is_flag=True, help="Use non-negative least squares.")
@click.option("--yes", "-y", is_flag=True, help="Don't ask for confirmation.")
@util.signature_analysis
def optimize_cmd(
    dataformat: str,
    data: typing.List[str],
    out: str,
    nfev: int,
    nnls: bool,
    yes: bool,
    parameters_file: str,
    model_file: str,
    scheme_file: str,
):
    """Optimizes a model.
    e.g.:
    glotaran optimize --

    """
    if scheme_file is not None:
        scheme = util.load_scheme_file(scheme_file, verbose=True)
        if nfev is not None:
            scheme.nfev = nfev
    else:
        if model_file is None:
            click.echo("Error: Neither scheme nor model specified", err=True)
            sys.exit(1)
        model = util.load_model_file(model_file, verbose=True)

        if parameters_file is None:
            click.echo("Error: Neither scheme nor parameter specified", err=True)
            sys.exit(1)
        parameters = util.load_parameter_file(parameters_file, verbose=True)

        if len(data) == 0:
            click.echo("Error: Neither scheme nor data specified", err=True)
            sys.exit(1)
        dataset_files = {arg[0]: arg[1] for arg in data}
        datasets = {}
        for label in model.dataset:
            if label not in dataset_files:
                click.echo(f"Missing dataset for '{label}'", err=True)
                sys.exit(1)
            path = dataset_files[label]
            datasets[label] = util.load_dataset_file(path, fmt=dataformat, verbose=True)

        scheme = gta.analysis.scheme.Scheme(
            model=model,
            parameters=parameters,
            data=datasets,
            non_negative_least_squares=nnls,
            maximum_number_function_evaluations=nfev,
        )

    click.echo(scheme.validate())
    click.echo(f"Use NNLS: {scheme.non_negative_least_squares}")
    click.echo(f"Max Nr Function Evaluations: {scheme.maximum_number_function_evaluations}")
    click.echo(f"Saving directory: is '{out if out is not None else 'None'}'")

    if yes or click.confirm("Do you want to start optimization?", abort=True, default=True):
        #  try:
        #      click.echo('Preparing optimization...', nl=False)
        #      optimizer = gta.analysis.optimizer.Optimizer(scheme)
        #      click.echo(' Success')
        #  except Exception as e:
        #      click.echo(" Error")
        #      click.echo(e, err=True)
        #      sys.exit(1)
        try:
            click.echo("Optimizing...")
            result = optimize(scheme)
            click.echo("Optimization done.")
            click.echo(result.markdown(with_model=False))
            click.echo("Optimized Parameter:")
            click.echo(result.optimized_parameters.markdown())
        except Exception as e:
            click.echo(f"An error occurred during optimization: \n\n{e}", err=True)
            sys.exit(1)

        if out is not None:
            try:
                click.echo(f"Saving directory is '{out}'")
                if yes or click.confirm("Do you want to save the data?", default=True):
                    paths = result.save(out)
                    click.echo("File saving successful, the following files have been written:\n")
                    for p in paths:
                        click.echo(f"* {p}")
            except Exception as e:
                click.echo(f"An error occurred during optimization: \n\n{e}", err=True)
                sys.exit(1)

        click.echo("All done, have a nice day!")
