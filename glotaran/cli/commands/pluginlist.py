import click

from glotaran.io.register import known_project_fmts
from glotaran.model import known_model_names
from glotaran.plugin_system.data_io_registration import known_data_formats


def plugin_list_cmd():
    """Prints a list of installed plugins."""

    output = """
    Installed Glotaran Plugins:

    Models:
    """
    output += "\n"

    for name in known_model_names():
        output += f"    * {name}\n"

    output += "\nData file Formats\n\n"

    for reader_fmt in known_data_formats():
        output += f"    * .{reader_fmt}\n"

    output += "\nProject file Formats\n\n"

    for reader_fmt in known_project_fmts():
        output += f"    * .{reader_fmt}\n"

    click.echo(output)
