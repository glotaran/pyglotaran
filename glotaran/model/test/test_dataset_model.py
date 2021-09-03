"""Tests for glotaran.model.dataset_model.DatasetModel"""
from __future__ import annotations

import pytest

from glotaran.builtin.megacomplexes.baseline import BaselineMegacomplex
from glotaran.builtin.megacomplexes.coherent_artifact import CoherentArtifactMegacomplex
from glotaran.builtin.megacomplexes.damped_oscillation import DampedOscillationMegacomplex
from glotaran.builtin.megacomplexes.decay import DecayMegacomplex
from glotaran.builtin.megacomplexes.spectral import SpectralMegacomplex
from glotaran.model.dataset_model import create_dataset_model_type
from glotaran.model.model import default_dataset_properties


class MockModel:
    """Test Model only containing the megacomplex property.

    Multiple and different kinds of megacomplexes are defined
    but only a subset will be used by the DatsetModel.
    """

    def __init__(self) -> None:
        self.megacomplex = {
            # not unique
            "d1": DecayMegacomplex(),
            "d2": DecayMegacomplex(),
            "d3": DecayMegacomplex(),
            "s1": SpectralMegacomplex(),
            "s2": SpectralMegacomplex(),
            "s3": SpectralMegacomplex(),
            "doa1": DampedOscillationMegacomplex(),
            "doa2": DampedOscillationMegacomplex(),
            # unique
            "b1": BaselineMegacomplex(),
            "b2": BaselineMegacomplex(),
            "c1": CoherentArtifactMegacomplex(),
            "c2": CoherentArtifactMegacomplex(),
        }


@pytest.mark.parametrize(
    "used_megacomplexes, expected_problems",
    (
        (
            ["d1"],
            [],
        ),
        (
            ["d1", "d2", "d3"],
            [],
        ),
        (
            ["s1", "s2", "s3"],
            [],
        ),
        (
            ["d1", "d2", "d3", "s1", "s2", "s3", "doa1", "doa2", "b1", "c1"],
            [],
        ),
        (
            ["d1", "b1", "b2"],
            ["Multiple instances of unique megacomplex type 'baseline' in dataset 'ds1'"],
        ),
        (
            ["d1", "c1", "c2"],
            ["Multiple instances of unique megacomplex type 'coherent-artifact' in dataset 'ds1'"],
        ),
        (
            ["d1", "b1", "b2", "c1", "c2"],
            [
                "Multiple instances of unique megacomplex type 'baseline' in dataset 'ds1'",
                "Multiple instances of unique megacomplex type "
                "'coherent-artifact' in dataset 'ds1'",
            ],
        ),
    ),
)
def test_datasetmodel_ensure_unique_megacomplexes(
    used_megacomplexes: list[str], expected_problems: list[str]
):
    """Only report problems if multiple unique megacomplexes of the same type are used."""
    dataset_model = create_dataset_model_type({**default_dataset_properties})()
    dataset_model.megacomplex = used_megacomplexes  # type:ignore
    dataset_model.label = "ds1"  # type:ignore
    problems = dataset_model.ensure_unique_megacomplexes(MockModel())  # type:ignore

    assert len(problems) == len(expected_problems)
    assert problems == expected_problems
