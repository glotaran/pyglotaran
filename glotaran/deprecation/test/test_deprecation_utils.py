from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import glotaran
from glotaran.deprecation.deprecation_utils import GlotaranApiDeprecationWarning
from glotaran.deprecation.deprecation_utils import GlotaranDeprectedApiError
from glotaran.deprecation.deprecation_utils import OverDueDeprecation
from glotaran.deprecation.deprecation_utils import check_overdue
from glotaran.deprecation.deprecation_utils import deprecate
from glotaran.deprecation.deprecation_utils import deprecate_dict_entry
from glotaran.deprecation.deprecation_utils import glotaran_version
from glotaran.deprecation.deprecation_utils import module_attribute
from glotaran.deprecation.deprecation_utils import parse_version
from glotaran.deprecation.deprecation_utils import raise_deprecation_error
from glotaran.deprecation.deprecation_utils import warn_deprecated

if TYPE_CHECKING:
    from typing import Any
    from typing import Hashable
    from typing import Mapping

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
DEPRECATION_ERROR_MESSAGE = (
    "Usage of 'glotaran.deprecation.deprecation_utils.parse_version(version_str)' was deprecated, "
    "use 'glotaran.deprecation.deprecation_utils.check_qualnames_in_tests(qualnames)' instead.\n"
    "It wasn't possible to restore the original behavior of this usage "
    "(mostlikely due to an object hierarchy change)."
    "This usage change message won't be show as of version: '0.6.0'."
)
OVERDUE_ERROR_MESSAGE = (
    "Support for 'glotaran.deprecation.deprecation_utils.parse_version' "
    "was supposed to be dropped in version: '0.6.0'.\n"
    "Current version is: '1.0.0'"
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


@pytest.fixture
def glotaran_1_0_0(monkeypatch: MonkeyPatch):
    """Mock glotaran version to always be 1.0.0 for the test."""
    monkeypatch.setattr(
        glotaran.deprecation.deprecation_utils, "glotaran_version", lambda: "1.0.0"
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
def test_check_overdue_no_raise(monkeypatch: MonkeyPatch):
    """Current version smaller then drop_version."""
    check_overdue(
        deprecated_qual_name_usage=DEPRECATION_QUAL_NAME,
        to_be_removed_in_version="0.6.0",
    )


@pytest.mark.usefixtures("glotaran_1_0_0")
def test_check_overdue_raises(monkeypatch: MonkeyPatch):
    """Current version is equal or bigger than drop_version."""
    with pytest.raises(OverDueDeprecation) as excinfo:
        check_overdue(
            deprecated_qual_name_usage=DEPRECATION_QUAL_NAME,
            to_be_removed_in_version="0.6.0",
        )

    assert str(excinfo.value) == OVERDUE_ERROR_MESSAGE


@pytest.mark.usefixtures("glotaran_0_3_0")
def test_raise_deprecation_error(monkeypatch: MonkeyPatch):
    """Current version smaller then drop_version."""
    with pytest.raises(GlotaranDeprectedApiError) as excinfo:
        raise_deprecation_error(
            deprecated_qual_name_usage=DEPRECATION_QUAL_NAME,
            new_qual_name_usage=NEW_QUAL_NAME,
            to_be_removed_in_version="0.6.0",
        )
    assert str(excinfo.value) == DEPRECATION_ERROR_MESSAGE


@pytest.mark.usefixtures("glotaran_1_0_0")
def test_raise_deprecation_error_overdue(monkeypatch: MonkeyPatch):
    """Current version is equal or bigger than drop_version."""
    with pytest.raises(OverDueDeprecation) as excinfo:
        raise_deprecation_error(
            deprecated_qual_name_usage=DEPRECATION_QUAL_NAME,
            new_qual_name_usage=NEW_QUAL_NAME,
            to_be_removed_in_version="0.6.0",
        )
    assert str(excinfo.value) == OVERDUE_ERROR_MESSAGE


@pytest.mark.usefixtures("glotaran_0_3_0")
def test_warn_deprecated():
    """Warning gets shown when all is in order."""
    with pytest.warns(GlotaranApiDeprecationWarning) as record:
        warn_deprecated(
            deprecated_qual_name_usage=DEPRECATION_QUAL_NAME,
            new_qual_name_usage=NEW_QUAL_NAME,
            to_be_removed_in_version="0.6.0",
        )

        assert len(record) == 1
        assert record[0].message.args[0] == DEPRECATION_WARN_MESSAGE
        assert Path(record[0].filename) == Path(__file__)


@pytest.mark.usefixtures("glotaran_1_0_0")
def test_warn_deprecated_overdue_deprecation(monkeypatch: MonkeyPatch):
    """Current version is equal or bigger than drop_version."""

    with pytest.raises(OverDueDeprecation) as excinfo:
        warn_deprecated(
            deprecated_qual_name_usage=DEPRECATION_QUAL_NAME,
            new_qual_name_usage=NEW_QUAL_NAME,
            to_be_removed_in_version="0.6.0",
        )
    assert str(excinfo.value) == OVERDUE_ERROR_MESSAGE


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
    with pytest.warns(GlotaranApiDeprecationWarning):
        warn_deprecated(
            deprecated_qual_name_usage=deprecated_qual_name_usage,
            new_qual_name_usage=new_qual_name_usage,
            to_be_removed_in_version="0.6.0",
            check_qual_names=check_qualnames,
        )


@pytest.mark.usefixtures("glotaran_0_3_0")
def test_warn_deprecated_sliced_method():
    """Slice away method for importing and check class for attribute"""
    with pytest.warns(GlotaranApiDeprecationWarning):
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
    with pytest.warns(GlotaranApiDeprecationWarning):
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
    assert recwarn[0].category == GlotaranApiDeprecationWarning
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
    assert recwarn[0].category == GlotaranApiDeprecationWarning
    assert recwarn[0].message.args[0] == DEPRECATION_WARN_MESSAGE  # type: ignore [union-attr]
    assert Path(recwarn[0].filename) == Path(__file__)

    Foo.from_string("foo")

    assert len(recwarn) == 2


@pytest.mark.usefixtures("glotaran_0_3_0")
def test_deprecate_dict_key_swap_keys():
    """Replace old with new key while keeping the value."""
    test_dict = {"foo": 123}
    with pytest.warns(
        GlotaranApiDeprecationWarning, match="'foo'.+was deprecated, use 'bar'"
    ) as record:
        deprecate_dict_entry(
            dict_to_check=test_dict,
            deprecated_usage="foo",
            new_usage="bar",
            to_be_removed_in_version="0.6.0",
            swap_keys=("foo", "bar"),
        )

        assert "bar" in test_dict
        assert test_dict["bar"] == 123
        assert "foo" not in test_dict

        assert len(record) == 1
        assert Path(record[0].filename) == Path(__file__)


@pytest.mark.usefixtures("glotaran_0_3_0")
def test_deprecate_dict_key_replace_rules_only_values():
    """Replace old value for key with new value."""
    test_dict = {"foo": 123}
    with pytest.warns(
        GlotaranApiDeprecationWarning, match="'foo: 123'.+was deprecated, use 'foo: 321'"
    ) as record:
        deprecate_dict_entry(
            dict_to_check=test_dict,
            deprecated_usage="foo: 123",
            new_usage="foo: 321",
            to_be_removed_in_version="0.6.0",
            replace_rules=({"foo": 123}, {"foo": 321}),
        )

        assert "foo" in test_dict
        assert test_dict["foo"] == 321

        assert len(record) == 1
        assert Path(record[0].filename) == Path(__file__)


@pytest.mark.usefixtures("glotaran_0_3_0")
def test_deprecate_dict_key_replace_rules_keys_and_values():
    """Replace old with new key AND replace old value for key with new value."""
    test_dict = {"foo": 123}
    with pytest.warns(
        GlotaranApiDeprecationWarning, match="'foo: 123'.+was deprecated, use 'bar: 321'"
    ) as record:
        deprecate_dict_entry(
            dict_to_check=test_dict,
            deprecated_usage="foo: 123",
            new_usage="bar: 321",
            to_be_removed_in_version="0.6.0",
            replace_rules=({"foo": 123}, {"bar": 321}),
        )

        assert "bar" in test_dict
        assert test_dict["bar"] == 321
        assert "foo" not in test_dict

        assert len(record) == 1
        assert Path(record[0].filename) == Path(__file__)


@pytest.mark.xfail(strict=True)
@pytest.mark.usefixtures("glotaran_0_3_0")
def test_deprecate_dict_key_does_not_apply_swap_keys():
    """Don't warn if the dict doesn't change because old_key didn't match"""

    with pytest.warns(
        GlotaranApiDeprecationWarning, match="'foo: 123'.+was deprecated, use 'foo: 321'"
    ):
        deprecate_dict_entry(
            dict_to_check={"foo": 123},
            deprecated_usage="foo: 123",
            new_usage="foo: 321",
            to_be_removed_in_version="0.6.0",
            swap_keys=("bar", "baz"),
        )


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize(
    "replace_rules",
    (
        ({"bar": 123}, {"bar": 321}),
        ({"foo": 111}, {"bar": 321}),
    ),
)
@pytest.mark.usefixtures("glotaran_0_3_0")
def test_deprecate_dict_key_does_not_apply(
    replace_rules: tuple[Mapping[Hashable, Any], Mapping[Hashable, Any]]
):
    """Don't warn if the dict doesn't change because old_key or old_value didn't match"""
    with pytest.warns(
        GlotaranApiDeprecationWarning, match="'foo: 123'.+was deprecated, use 'foo: 321'"
    ):
        deprecate_dict_entry(
            dict_to_check={"foo": 123},
            deprecated_usage="foo: 123",
            new_usage="foo: 321",
            to_be_removed_in_version="0.6.0",
            replace_rules=replace_rules,
        )


@pytest.mark.parametrize(
    "swap_keys, replace_rules",
    (
        (None, None),
        (("bar", "baz"), ({"bar": 1}, {"baz": 2})),
    ),
)
@pytest.mark.usefixtures("glotaran_0_3_0")
def test_deprecate_dict_key_error_no_action(
    swap_keys: tuple[Hashable, Hashable] | None,
    replace_rules: tuple[Mapping[Hashable, Any], Mapping[Hashable, Any]] | None,
):
    """Raise error if none or both `swap_keys` and `replace_rules` were provided."""
    with pytest.raises(
        ValueError,
        match=(
            r"Exactly one of the parameters `swap_keys` or `replace_rules` needs to be provided\."
        ),
    ):
        deprecate_dict_entry(
            dict_to_check={},
            deprecated_usage="",
            new_usage="",
            to_be_removed_in_version="",
            swap_keys=swap_keys,
            replace_rules=replace_rules,
        )


def test_module_attribute():
    """Same code as the original import"""

    result = module_attribute("glotaran.deprecation.deprecation_utils", "parse_version")

    assert result.__code__ == parse_version.__code__


@pytest.mark.usefixtures("glotaran_0_3_0")
def test_deprecate_module_attribute():
    """Same code as the original import and warning"""

    with pytest.warns(GlotaranApiDeprecationWarning) as record:

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
    assert recwarn[0].category == GlotaranApiDeprecationWarning


@pytest.mark.usefixtures("glotaran_0_3_0")
def test_deprecate_submodule_from_import(recwarn: WarningsRecorder):
    """Raise warning when Attribute of fake module is imported"""

    from glotaran.deprecation.test.dummy_package.deprecated_module import (  # noqa: F401
        parse_version,
    )

    assert len(recwarn) == 1
    assert recwarn[0].category == GlotaranApiDeprecationWarning
    assert Path(recwarn[0].filename) == Path(__file__)


@pytest.mark.usefixtures("glotaran_0_3_0")
def test_deprecate_submodule_import_error(recwarn: WarningsRecorder):
    """Raise warning when Attribute of fake module is imported"""

    with pytest.raises(ImportError) as excinfo:

        from glotaran.deprecation.test.dummy_package.deprecated_module import (  # noqa: F401
            does_not_exists,
        )

    assert str(excinfo.value) == (
        "cannot import name 'does_not_exists' from "
        "'glotaran.deprecation.test.dummy_package.deprecated_module' (unknown location)"
    )


@pytest.mark.usefixtures("glotaran_0_3_0")
def test_deprecate_submodule_attr__file__(recwarn: WarningsRecorder):
    """Now warning when inspecting __file__ attribute (pytest using inspect)"""
    warnings.simplefilter("always")

    from glotaran.deprecation.test import dummy_package

    dummy_package.__file__

    assert len(recwarn) == 0
