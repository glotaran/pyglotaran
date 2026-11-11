from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import pytest

from glotaran.project import Project
from glotaran.project import Scheme
from glotaran.testing.simulated_data.sequential_spectral_decay import DATASET
from glotaran.testing.simulated_data.sequential_spectral_decay import SCHEME_DICT

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr

LABEL = "sequential-decay"


def scheme_dict_with_data(data_path: str) -> dict:
    """``SCHEME_DICT`` with a ``data:`` path for its dataset, as a scheme file can have."""
    scheme_dict = copy.deepcopy(SCHEME_DICT)
    scheme_dict["experiments"][LABEL]["datasets"][LABEL]["data"] = data_path
    return scheme_dict


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.start(tmp_path / "project")


@pytest.fixture
def scheme() -> Scheme:
    return Scheme.from_dict(SCHEME_DICT)


@pytest.fixture
def data() -> xr.Dataset:
    """Every 10th time point of the simulated sequential decay, to keep fits fast."""
    return DATASET.isel(time=slice(None, None, 10)).copy(deep=True)
