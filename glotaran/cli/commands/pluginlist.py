import click

from glotaran.io.register import known_fmts
from glotaran.model import known_model_names


def plugin_list_cmd():
    """Prints a list of installed plugins."""

    output = """
    Installed Glotaran Plugins:

    Models:
    """
    output += "\n"

    for name in known_model_names():
        output += f"    * {name}\n"

    output += "\nFile Formats\n\n"

    for reader_fmt in known_fmts():
        output += f"    * .{reader_fmt}\n"

    click.echo(output)
