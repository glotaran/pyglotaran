"""Tests for glotaran.model.dataset_model.DatasetModel"""
from __future__ import annotations

from glotaran.model.item import get_item_issues
from glotaran.model.megacomplex import Megacomplex
from glotaran.model.megacomplex import megacomplex
from glotaran.model.model import Model


@megacomplex()
class MockMegacomplexNonUniqueExclusive(Megacomplex):
    type: str = "test_megacomplex_not_exclusive_unique"


@megacomplex(exclusive=True)
class MockMegacomplexExclusive(Megacomplex):
    type: str = "test_megacomplex_exclusive"


@megacomplex(unique=True)
class MockMegacomplexUnique(Megacomplex):
    type: str = "test_megacomplex_unique"


def test_get_issues_datasetmodel():
    mcls = Model.create_class_from_megacomplexes(
        [
            MockMegacomplexNonUniqueExclusive,
            MockMegacomplexExclusive,
            MockMegacomplexUnique,
        ]
    )
    m = mcls(
        megacomplex={
            "m": {"type": "test_megacomplex_not_exclusive_unique"},
            "m_exclusive": {"type": "test_megacomplex_exclusive"},
            "m_unique": {"type": "test_megacomplex_unique"},
        },
        dataset={
            "ok": {"megacomplex": ["m"]},
            "exclusive": {"megacomplex": ["m", "m_exclusive"]},
            "unique": {"megacomplex": ["m_unique", "m_unique"]},
        },
    )

    assert len(get_item_issues(item=m.dataset["ok"], model=m)) == 0
    assert len(get_item_issues(item=m.dataset["exclusive"], model=m)) == 1
    assert len(get_item_issues(item=m.dataset["unique"], model=m)) == 2

    m = mcls(
        megacomplex={
            "m": {"type": "test_megacomplex_not_exclusive_unique"},
            "m_exclusive": {"type": "test_megacomplex_exclusive"},
            "m_unique": {"type": "test_megacomplex_unique"},
        },
        dataset={
            "ok": {"megacomplex": [], "global_megacomplex": ["m"]},
            "exclusive": {"megacomplex": [], "global_megacomplex": ["m", "m_exclusive"]},
            "unique": {"megacomplex": [], "global_megacomplex": ["m_unique", "m_unique"]},
        },
    )

    assert len(get_item_issues(item=m.dataset["ok"], model=m)) == 0
    assert len(get_item_issues(item=m.dataset["exclusive"], model=m)) == 1
    assert len(get_item_issues(item=m.dataset["unique"], model=m)) == 2
