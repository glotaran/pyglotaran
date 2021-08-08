from click.testing import CliRunner

from glotaran import cli


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.glotaran)
    assert result.exit_code == 0
    help_result = runner.invoke(cli.glotaran, ["--help"])
    assert help_result.exit_code == 0
    assert "Usage: glotaran [OPTIONS] COMMAND [ARGS]..." in help_result.output
    plugin_result = runner.invoke(cli.glotaran, ["pluginlist"])
    assert plugin_result.exit_code == 0
    assert "Installed Glotaran Plugins" in plugin_result.output
