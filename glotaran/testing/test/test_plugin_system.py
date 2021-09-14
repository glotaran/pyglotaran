import pytest

from glotaran.io import DataIoInterface
from glotaran.io import ProjectIoInterface
from glotaran.model import Megacomplex
from glotaran.model import megacomplex
from glotaran.plugin_system.data_io_registration import known_data_formats
from glotaran.plugin_system.megacomplex_registration import known_megacomplex_names
from glotaran.plugin_system.project_io_registration import known_project_formats
from glotaran.testing.plugin_system import monkeypatch_plugin_registry
from glotaran.testing.plugin_system import monkeypatch_plugin_registry_data_io
from glotaran.testing.plugin_system import monkeypatch_plugin_registry_megacomplex
from glotaran.testing.plugin_system import monkeypatch_plugin_registry_project_io


@megacomplex(dimension="test")
class DummyMegacomplex(Megacomplex):
    pass


class DummyDataIo(DataIoInterface):
    pass


class DummyProjectIo(ProjectIoInterface):
    pass


def test_monkeypatch_megacomplexes():
    """Megacomplex only added to registry while context is entered."""
    with monkeypatch_plugin_registry_megacomplex(test_megacomplex={"test_mc": DummyMegacomplex}):
        assert "test_mc" in known_megacomplex_names()

    assert "test_mc" not in known_megacomplex_names()
    with monkeypatch_plugin_registry(test_megacomplex={"test_full": DummyMegacomplex}):
        assert "test_full" in known_megacomplex_names()

    assert "test_full" not in known_megacomplex_names()


def test_monkeypatch_data_io():
    """DataIoInterface only added to registry while context is entered."""
    with monkeypatch_plugin_registry_data_io(
        test_data_io={"test_dio": DummyDataIo(format_name="test")}
    ):
        assert "test_dio" in known_data_formats()

    assert "test_mc" not in known_data_formats()

    with monkeypatch_plugin_registry(test_data_io={"test_full": DummyDataIo(format_name="test")}):
        assert "test_full" in known_data_formats()

    assert "test_full" not in known_data_formats()


def test_monkeypatch_project_io():
    """ProjectIoInterface only added to registry while context is entered."""
    with monkeypatch_plugin_registry_project_io(
        test_project_io={"test_pio": DummyProjectIo(format_name="test")}
    ):
        assert "test_pio" in known_project_formats()

    assert "test_pio" not in known_megacomplex_names()
    with monkeypatch_plugin_registry(
        test_project_io={"test_full": DummyProjectIo(format_name="test")}
    ):
        assert "test_full" in known_project_formats()

    assert "test_full" not in known_project_formats()


@pytest.mark.parametrize("create_new_registry", (True, False))
def test_monkeypatch_plugin_registry_full(create_new_registry: bool):
    """Create a completely new registry."""

    assert "decay" in known_megacomplex_names()
    assert "yml" in known_project_formats()
    assert "sdt" in known_data_formats()

    with monkeypatch_plugin_registry(
        test_megacomplex={"test_mc": DummyMegacomplex},
        test_project_io={"test_pio": DummyProjectIo(format_name="test")},
        test_data_io={"test_dio": DummyDataIo(format_name="test")},
        create_new_registry=create_new_registry,
    ):
        assert "test_mc" in known_megacomplex_names()
        assert "test_pio" in known_project_formats()
        assert "test_dio" in known_data_formats()
        assert ("decay" not in known_megacomplex_names()) is create_new_registry
        assert ("yml" not in known_project_formats()) is create_new_registry
        assert ("sdt" not in known_data_formats()) is create_new_registry
