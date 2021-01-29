import click

from . import util


@util.signature_analysis
def print_cmd(parameters_file: str, model_file: str, scheme_file: str):
    """Parses scheme, a model or a parameter file and prints the result as a Markdown formatted
    string."""
    model = None
    parameters = None
    if scheme_file is not None:
        scheme = util.load_scheme_file(scheme_file, verbose=False)
        click.echo(scheme.markdown())
        return
    if model_file is not None:
        model = util.load_model_file(model_file, verbose=False)

    if parameters_file is not None:
        parameters = util.load_parameters_file(parameters_file, verbose=False)

    if model:
        click.echo(model.markdown(parameters=parameters))
    elif parameters:
        click.echo(parameters.markdown())
    else:
        click.echo("Nothing to print, please type 'glotaran print --help' for more info.")
