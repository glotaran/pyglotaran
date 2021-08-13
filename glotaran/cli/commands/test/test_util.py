from glotaran.cli.commands.util import project_io_list_supporting_plugins


def test_project_io_list_supporting_plugins_save_result():
    """Same as used in ``--outformat`` CLI option."""
    result = project_io_list_supporting_plugins("save_result", ("yml_str"))

    assert "csv" not in result
    assert "yml_str" not in result
    assert "folder" in result
