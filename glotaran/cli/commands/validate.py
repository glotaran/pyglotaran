import click

from . import util


@util.signature_analysis
def validate_cmd(parameters_file: str, model_file: str, scheme_file: str):
    """Validates a model file and optionally a parameter file."""

    if scheme_file is not None:
        scheme = util.load_scheme_file(scheme_file, verbose=True)
        click.echo(scheme.validate())
        return

    if model_file is not None:

        model = util.load_model_file(model_file, verbose=True)
        parameters = None
        if parameters_file is not None:
            parameters = util.load_parameter_file(parameters_file, verbose=True)

        click.echo(model.validate(parameters=parameters))
    else:
        click.echo(
            "Neither analysis scheme nor model file specified. "
            "Type 'glotaran validate --help' for more info."
        )
