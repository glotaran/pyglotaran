from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import glotaran
from glotaran.deprecation.deprecation_utils import OverDueDeprecation
from glotaran.deprecation.deprecation_utils import deprecate
from glotaran.deprecation.deprecation_utils import glotaran_version
from glotaran.deprecation.deprecation_utils import module_attribute
from glotaran.deprecation.deprecation_utils import parse_version
from glotaran.deprecation.deprecation_utils import warn_deprecated

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch
    from _pytest.recwarn import WarningsRecorder


DEP_UTILS_QUALNAME = "glotaran.deprecation.deprecation_utils"

DEPRECATION_QUAL_NAME = f"{DEP_UTILS_QUALNAME}.parse_version(version_str)"
NEW_QUAL_NAME = f"{DEP_UTILS_QUALNAME}.check_qualnames_in_tests(qualnames)"

DEPRECATION_WARN_MESSAGE = (
    "Usage of 'glotaran.deprecation.deprecation_utils.parse_version(version_str)' "
    "was deprecated, use "
    "'glotaran.deprecation.deprecation_utils.check_qualnames_in_tests(qualnames)' "
    "instead.\nThis usage will be an error in version: '0.6.0'."
)


class DummyClass:
    """Dummy class to check check_qualnames_in_tests"""

    foo: dict[str, str] = {}


@pytest.fixture
def glotaran_0_3_0(monkeypatch: MonkeyPatch):
    """Mock glotaran version to always be 0.3.0 for the test."""
    monkeypatch.setattr(
        glotaran.deprecation.deprecation_utils, "glotaran_version", lambda: "0.3.0"
    )
    yield


def test_glotaran_version():
    """Versions are the same."""
    assert glotaran_version() == glotaran.__version__


@pytest.mark.parametrize(
    "version_str, expected",
    (
        ("0.0.1", (0, 0, 1)),
        ("0.0.1.post", (0, 0, 1)),
        ("0.0.1-dev", (0, 0, 1)),
        ("0.0.1-dev.post", (0, 0, 1)),
    ),
)
def test_parse_version(version_str: str, expected: tuple[int, int, int]):
    """Valid version strings."""
    assert parse_version(version_str) == expected


@pytest.mark.parametrize(
    "version_str",
    ("1", "0.1", "a.b.c"),
)
def test_parse_version_errors(version_str: str):
    """Invalid version strings"""
    with pytest.raises(ValueError, match=f"'{version_str}'"):
        parse_version(version_str)


@pytest.mark.usefixtures("glotaran_0_3_0")
def test_warn_deprecated():
    """Warning gets shown when all is in order."""
    with pytest.warns(DeprecationWarning) as record:
        warn_deprecated(
            deprecated_qual_name_usage=DEPRECATION_QUAL_NAME,
            new_qual_name_usage=NEW_QUAL_NAME,
            to_be_removed_in_version="0.6.0",
        )

        assert len(record) == 1
        assert record[0].message.args[0] == DEPRECATION_WARN_MESSAGE
        assert Path(record[0].filename) == Path(__file__)


def test_warn_deprecated_overdue_deprecation(monkeypatch: MonkeyPatch):
    """Current version is equal or bigger than drop_version."""
    monkeypatch.setattr(
        glotaran.deprecation.deprecation_utils, "glotaran_version", lambda: "1.0.0"
    )

    with pytest.raises(OverDueDeprecation) as record:
        warn_deprecated(
            deprecated_qual_name_usage=DEPRECATION_QUAL_NAME,
            new_qual_name_usage=NEW_QUAL_NAME,
            to_be_removed_in_version="0.6.0",
        )

        assert len(record) == 1  # type: ignore [arg-type]
        expected = (
            "Support for 'glotaran.read_model_from_yaml' was supposed "
            "to be dropped in version: '0.6.0'.\n"
            "Current version is: '1.0.0'"
        )

        assert record[0].message.args[0] == expected  # type: ignore [index]
        assert Path(record[0].filename) == Path(__file__)  # type: ignore [index]


@pytest.mark.filterwarnings("ignore:Usage")
@pytest.mark.xfail(strict=True, reason="Dev version aren't checked")
def test_warn_deprecated_no_overdue_deprecation_on_dev(monkeypatch: MonkeyPatch):
    """Current version is equal or bigger than drop_version but it's a dev version."""
    monkeypatch.setattr(
        glotaran.deprecation.deprecation_utils, "glotaran_version", lambda: "0.6.0-dev"
    )

    with pytest.raises(OverDueDeprecation):
        warn_deprecated(
            deprecated_qual_name_usage=DEPRECATION_QUAL_NAME,
            new_qual_name_usage=NEW_QUAL_NAME,
            to_be_removed_in_version="0.6.0",
        )


@pytest.mark.parametrize(
    "deprecated_qual_name_usage,new_qual_name_usage",
    (
        (
            "glotaran.does_not_exists(foo)",
            DEPRECATION_QUAL_NAME,
        ),
        (
            DEPRECATION_QUAL_NAME,
            "glotaran.does_not_exists(foo)",
        ),
    ),
)
@pytest.mark.xfail(strict=True, reason="Should fail if any qualname is wrong.")
@pytest.mark.usefixtures("glotaran_0_3_0")
def test_warn_deprecated_broken_deprecated_qualname(
    deprecated_qual_name_usage: str, new_qual_name_usage: str
):
    """Fail if any qualname is wrong."""
    warn_deprecated(
        deprecated_qual_name_usage=deprecated_qual_name_usage,
        new_qual_name_usage=new_qual_name_usage,
        to_be_removed_in_version="0.6.0",
    )


@pytest.mark.parametrize(
    "deprecated_qual_name_usage,new_qual_name_usage,check_qualnames",
    (
        ("glotaran.does_not_exists(foo)", DEPRECATION_QUAL_NAME, (False, True)),
        (DEPRECATION_QUAL_NAME, "glotaran.does_not_exists(foo)", (True, False)),
    ),
)
@pytest.mark.usefixtures("glotaran_0_3_0")
def test_warn_deprecated_broken_qualname_no_check(
    deprecated_qual_name_usage: str, new_qual_name_usage: str, check_qualnames: tuple[bool, bool]
):
    """Not checking broken imports."""
    with pytest.warns(DeprecationWarning):
        warn_deprecated(
            deprecated_qual_name_usage=deprecated_qual_name_usage,
            new_qual_name_usage=new_qual_name_usage,
            to_be_removed_in_version="0.6.0",
            check_qual_names=check_qualnames,
        )


@pytest.mark.usefixtures("glotaran_0_3_0")
def test_warn_deprecated_sliced_method():
    """Slice away method for importing and check class for attribute"""
    with pytest.warns(DeprecationWarning):
        warn_deprecated(
            deprecated_qual_name_usage=(
                "glotaran.deprecation.test.test_deprecation_utils.DummyClass.foo()"
            ),
            new_qual_name_usage=DEPRECATION_QUAL_NAME,
            to_be_removed_in_version="0.6.0",
            importable_indices=(2, 1),
        )


@pytest.mark.usefixtures("glotaran_0_3_0")
def test_warn_deprecated_sliced_mapping():
    """Slice away mapping for importing and check class for attribute"""
    with pytest.warns(DeprecationWarning):
        warn_deprecated(
            deprecated_qual_name_usage=(
                "glotaran.deprecation.test.test_deprecation_utils.DummyClass.foo['bar']"
            ),
            new_qual_name_usage=DEPRECATION_QUAL_NAME,
            to_be_removed_in_version="0.6.0",
            importable_indices=(2, 1),
        )


@pytest.mark.xfail(
    strict=True,
    reason="Should fail if new qualname is wrong, even if check_deprecated_qualname is False.",
)
@pytest.mark.usefixtures("glotaran_0_3_0")
def test_warn_deprecated_broken_new_qualname_no_check():
    """Fail if new qualname is wrong."""
    warn_deprecated(
        deprecated_qual_name_usage="glotaran.does_not_exists(foo)",
        new_qual_name_usage="glotaran.does_not_exists(foo)",
        to_be_removed_in_version="0.6.0",
        check_qual_names=(False, True),
    )


@pytest.mark.usefixtures("glotaran_0_3_0")
def test_deprecated_decorator_function(recwarn: WarningsRecorder):
    """Deprecate function with decorator."""
    warnings.simplefilter("always")

    @deprecate(
        deprecated_qual_name_usage=DEPRECATION_QUAL_NAME,
        new_qual_name_usage=NEW_QUAL_NAME,
        to_be_removed_in_version="0.6.0",
    )
    def dummy():
        """Dummy docstring for testing."""

    dummy()

    assert dummy.__doc__ == "Dummy docstring for testing."
    assert len(recwarn) == 1
    assert recwarn[0].category == DeprecationWarning
    assert recwarn[0].message.args[0] == DEPRECATION_WARN_MESSAGE  # type: ignore [union-attr]
    assert Path(recwarn[0].filename) == Path(__file__)

    dummy()

    assert len(recwarn) == 2


@pytest.mark.usefixtures("glotaran_0_3_0")
def test_deprecated_decorator_class(recwarn: WarningsRecorder):
    """Deprecate class with decorator."""
    warnings.simplefilter("always")

    @deprecate(
        deprecated_qual_name_usage=DEPRECATION_QUAL_NAME,
        new_qual_name_usage=NEW_QUAL_NAME,
        to_be_removed_in_version="0.6.0",
    )
    class Foo:
        """Foo class docstring for testing."""

        @classmethod
        def from_string(cls, string: str):
            """Just another method to init the class for testing."""
            return cls()

    Foo()

    assert Foo.__doc__ == "Foo class docstring for testing."
    assert len(recwarn) == 1
    assert recwarn[0].category == DeprecationWarning
    assert recwarn[0].message.args[0] == DEPRECATION_WARN_MESSAGE  # type: ignore [union-attr]
    assert Path(recwarn[0].filename) == Path(__file__)

    Foo.from_string("foo")

    assert len(recwarn) == 2


def test_module_attribute():
    """Same code as the original import"""

    result = module_attribute("glotaran.deprecation.deprecation_utils", "parse_version")

    assert result.__code__ == parse_version.__code__


@pytest.mark.usefixtures("glotaran_0_3_0")
def test_deprecate_module_attribute():
    """Same code as the original import and warning"""

    with pytest.warns(DeprecationWarning) as record:

        from glotaran.deprecation.test.dummy_package.deprecated_module_attribute import (
            deprecated_attribute,
        )

        assert deprecated_attribute.__code__ == parse_version.__code__
        assert Path(record[0].filename) == Path(__file__)


@pytest.mark.usefixtures("glotaran_0_3_0")
def test_deprecate_submodule(recwarn: WarningsRecorder):
    """Raise warning when Attribute of fake module is used"""

    from glotaran.deprecation.test.dummy_package import deprecated_module

    assert (
        deprecated_module.parse_version.__code__  # type: ignore [attr-defined]
        == parse_version.__code__
    )

    assert len(recwarn) == 1
    assert recwarn[0].category == DeprecationWarning


@pytest.mark.usefixtures("glotaran_0_3_0")
def test_deprecate_submodule_from_import(recwarn: WarningsRecorder):
    """Raise warning when Attribute of fake module is imported"""

    from glotaran.deprecation.test.dummy_package.deprecated_module import (  # noqa: F401
        parse_version,
    )

    assert len(recwarn) == 1
    assert recwarn[0].category == DeprecationWarning
    assert Path(recwarn[0].filename) == Path(__file__)


@pytest.mark.usefixtures("glotaran_0_3_0")
def test_deprecate_submodule_import_error(recwarn: WarningsRecorder):
    """Raise warning when Attribute of fake module is imported"""

    with pytest.raises(ImportError) as record:

        from glotaran.deprecation.test.dummy_package.deprecated_module import (  # noqa: F401
            does_not_exists,
        )

        assert len(record) == 1  # type:ignore[arg-type]
        assert record[0].message.args[0] == (  # type:ignore[index]
            "ImportError: cannot import name 'does_not_exists' from "
            "'glotaran.deprecation.test.deprecated_module' (unknown location)"
        )
        assert len(recwarn) == 0
        assert Path(record[0].filename) == Path(__file__)  # type:ignore[index]


@pytest.mark.usefixtures("glotaran_0_3_0")
def test_deprecate_submodule_attr__file__(recwarn: WarningsRecorder):
    """Now warning when inspecting __file__ attribute (pytest using inspect)"""
    warnings.simplefilter("always")

    from glotaran.deprecation.test import dummy_package

    dummy_package.__file__

    assert len(recwarn) == 0
