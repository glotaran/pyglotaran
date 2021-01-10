import click

from glotaran.io.reader import known_reading_formats
from glotaran.parse.register import known_model_names


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

    for reader in known_reading_formats.values():
        output += f"    * .{reader.extension} : {reader.name}\n"

    click.echo(output)
