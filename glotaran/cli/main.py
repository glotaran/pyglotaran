import click

import glotaran as gta

from .commands.optimize import optimize_cmd
from .commands.pluginlist import plugin_list_cmd
from .commands.print import print_cmd
from .commands.validate import validate_cmd


class Cli(click.Group):
    def __init__(self, *args, **kwargs):
        self.help_priorities = {}
        super().__init__(*args, **kwargs)

    def get_help(self, ctx):
        self.list_commands = self.list_commands_for_help
        return super().get_help(ctx)

    def list_commands_for_help(self, ctx):
        """reorder the list of commands when listing the help"""
        commands = super().list_commands(ctx)
        return (
            c[1]
            for c in sorted(
                (self.help_priorities.get(command, 1), command) for command in commands
            )
        )

    def command(self, *args, **kwargs):
        """Behaves the same as `click.Group.command()` except capture
        a priority for listing command names in help.
        """
        help_priority = kwargs.pop("help_priority", 1)
        help_priorities = self.help_priorities

        def decorator(f):
            cmd = super(Cli, self).command(*args, **kwargs)(f)
            help_priorities[cmd.name] = help_priority
            return cmd

        return decorator


@click.group(cls=Cli)
@click.version_option(version=gta.__version__)
def glotaran():
    pass


glotaran.add_command(
    glotaran.command(
        name="pluginlist", short_help="Prints a list of installed plugins.", help_priority=4
    )(plugin_list_cmd)
)
glotaran.add_command(
    glotaran.command(name="print", short_help="Prints a model as markdown.", help_priority=3)(
        print_cmd
    )
)
glotaran.add_command(
    glotaran.command(name="validate", short_help="Validates a model file.", help_priority=2)(
        validate_cmd
    )
)
glotaran.add_command(
    glotaran.command(name="optimize", short_help="Optimizes a model.", help_priority=1)(
        optimize_cmd
    )
)


if __name__ == "__main__":
    glotaran()
