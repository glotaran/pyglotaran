import click

from . import util


@util.signature_analysis
def print_cmd(parameter: str, model: str, scheme: str):
    """Parses scheme, a model or a parameter file and prints the result as a Markdown formatted
    string."""

    if scheme is not None:
        scheme = util.load_scheme_file(scheme, verbose=False)
        click.echo(scheme.markdown())
        return
    if model is not None:
        model = util.load_model_file(model, verbose=False)

    if parameter is not None:
        parameter = util.load_parameter_file(parameter, verbose=False)

    if model:
        click.echo(model.markdown(parameter=parameter))
    elif parameter:
        click.echo(parameter.markdown())
    else:
        click.echo("Nothing to print, please type 'glotaran print --help' for more info.")
