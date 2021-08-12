import click

from glotaran import __version__ as VERSION
from glotaran.cli.commands.optimize import optimize_cmd
from glotaran.cli.commands.pluginlist import plugin_list_cmd
from glotaran.cli.commands.print import print_cmd
from glotaran.cli.commands.validate import validate_cmd


class Cli(click.Group):
    """The glotaran CLI implementation of :class:`click.group`"""

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
@click.version_option(version=VERSION)
def main(prog_name="glotaran"):
    """The glotaran CLI main function."""
    pass


main.add_command(
    main.command(
        name="pluginlist", short_help="Prints a list of installed plugins.", help_priority=4
    )(plugin_list_cmd)
)
main.add_command(
    main.command(name="print", short_help="Prints a model as markdown.", help_priority=3)(
        print_cmd
    )
)
main.add_command(
    main.command(name="validate", short_help="Validates a model file.", help_priority=2)(
        validate_cmd
    )
)
main.add_command(
    main.command(name="optimize", short_help="Optimizes a model.", help_priority=1)(optimize_cmd)
)


if __name__ == "__main__":
    raise SystemExit(main(prog_name="glotaran"))
