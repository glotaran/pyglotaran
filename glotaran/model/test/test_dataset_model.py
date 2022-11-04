"""Tests for glotaran.model.dataset_model.DatasetModel"""
from __future__ import annotations

import pytest

from glotaran.model.dataset_model import get_dataset_model_model_dimension
from glotaran.model.item import fill_item
from glotaran.model.item import get_item_issues
from glotaran.model.megacomplex import Megacomplex
from glotaran.model.megacomplex import megacomplex
from glotaran.model.model import Model
from glotaran.parameter import Parameters


@megacomplex()
class MockMegacomplexNonUniqueExclusive(Megacomplex):
    type: str = "test_megacomplex_not_exclusive_unique"


@megacomplex(exclusive=True)
class MockMegacomplexExclusive(Megacomplex):
    type: str = "test_megacomplex_exclusive"


@megacomplex(unique=True)
class MockMegacomplexUnique(Megacomplex):
    type: str = "test_megacomplex_unique"


@megacomplex()
class MockMegacomplexDim1(Megacomplex):
    dimension: str = "dim1"
    type: str = "test_megacomplex_dim1"


@megacomplex()
class MockMegacomplexDim2(Megacomplex):
    dimension: str = "dim2"
    type: str = "test_megacomplex_dim2"


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


def test_get_model_dim():
    mcls = Model.create_class_from_megacomplexes([MockMegacomplexDim1, MockMegacomplexDim2])
    m = mcls(
        megacomplex={
            "m1": {"type": "test_megacomplex_dim1"},
            "m2": {"type": "test_megacomplex_dim2"},
        },
        dataset={
            "ok": {"megacomplex": ["m1"]},
            "error1": {"megacomplex": []},
            "error2": {"megacomplex": ["m1", "m2"]},
        },
    )

    get_dataset_model_model_dimension(
        fill_item(m.dataset["ok"], model=m, parameters=Parameters.from_list([]))
    )
    with pytest.raises(ValueError, match="Dataset model 'ok' was not filled."):
        get_dataset_model_model_dimension(m.dataset["ok"])
    with pytest.raises(ValueError, match="No megacomplex set for dataset model 'error1'."):
        get_dataset_model_model_dimension(m.dataset["error1"])
    with pytest.raises(
        ValueError, match="Megacomplex dimensions do not match for dataset model 'error2'."
    ):
        get_dataset_model_model_dimension(
            fill_item(m.dataset["error2"], model=m, parameters=Parameters.from_list([]))
        )
