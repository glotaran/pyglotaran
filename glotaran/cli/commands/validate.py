import click

from . import util


@util.signature_analysis
def validate_cmd(parameter: str, model: str, scheme: str):
    """Validates a model file and optionally a parameter file."""

    if scheme is not None:
        scheme = util.load_scheme_file(scheme, verbose=True)
        click.echo(scheme.validate())
        return

    if model is not None:

        model = util.load_model_file(model, verbose=True)

        if parameter is not None:
            parameter = util.load_parameter_file(parameter, verbose=True)

        click.echo(model.validate(parameter=parameter))
    else:
        click.echo("Neither analysis scheme nor model file specified. "
                   "Type 'glotaran validate --help' for more info.")
