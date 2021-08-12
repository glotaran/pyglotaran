from pathlib import Path

from click.testing import CliRunner

from glotaran.cli import main


def test_cli_help():
    """Test the CLI help options."""
    runner = CliRunner()
    result = runner.invoke(main)
    assert result.exit_code == 0
    help_result = runner.invoke(main, ["--help"], prog_name="glotaran")
    assert help_result.exit_code == 0
    assert "Usage: glotaran [OPTIONS] COMMAND [ARGS]..." in help_result.output


def test_cli_pluginlist():
    """Test the CLI pluginlist option."""
    runner = CliRunner()
    result = runner.invoke(main, ["pluginlist"], prog_name="glotaran")
    assert result.exit_code == 0
    assert "Installed Glotaran Plugins" in result.output


def test_cli_validate_parameters_file(tmp_path: Path):
    """Test the CLI pluginlist option."""
    empty_file = tmp_path.joinpath("empty_file.yml")
    empty_file.touch()
    runner = CliRunner()
    result_ok = runner.invoke(
        main, ["validate", "--parameters_file", str(empty_file)], prog_name="glotaran"
    )
    assert result_ok.exit_code == 0
    assert "Type 'glotaran validate --help' for more info." in result_ok.output
    non_existing_file = tmp_path.joinpath("_does_not_exist_.yml")
    result_file_not_exist = runner.invoke(
        main, ["validate", "--parameters_file", str(non_existing_file)], prog_name="glotaran"
    )
    assert result_file_not_exist.exit_code == 2
    assert all(
        substring in result_file_not_exist.output for substring in ("Error", "does not exist")
    )
