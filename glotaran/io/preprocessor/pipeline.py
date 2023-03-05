"""A pre-processor pipeline for data."""
from __future__ import annotations

from typing import Annotated

import xarray as xr
from pydantic import BaseModel
from pydantic import Field

from glotaran.io.preprocessor.preprocessor import CorrectBaselineAverage
from glotaran.io.preprocessor.preprocessor import CorrectBaselineValue

PipelineAction = Annotated[
    CorrectBaselineValue | CorrectBaselineAverage,
    Field(discriminator="action"),
]


class PreProcessingPipeline(BaseModel):
    """A pipeline for pre-processors."""

    actions: list[PipelineAction] = Field(default_factory=list)

    def apply(self, original: xr.DataArray) -> xr.DataArray:
        """Apply all pre-processors on data.

        Parameters
        ----------
        original: xr.DataArray
            The data to process.

        Returns
        -------
        xr.DataArray
        """
        result = original.copy()

        for action in self.actions:
            result = action.apply(result)
        return result

    def correct_baseline_value(self, value: float) -> PreProcessingPipeline:
        """Correct a dataset by subtracting baseline value.

        Parameters
        ----------
        value: float
            The value to subtract.

        Returns
        -------
        PreProcessingPipeline
        """
        return PreProcessingPipeline(actions=[*self.actions, CorrectBaselineValue(value=value)])

    def correct_baseline_average(
        self,
        select: dict[str, slice | list[int] | int] | None = None,
        exclude: dict[str, slice | list[int] | int] | None = None,
    ) -> PreProcessingPipeline:
        """Correct a dataset by subtracting the average over a part of the data.

        Parameters
        ----------
        select: dict[str, slice | list[int] | int] | None
            The selection to average as dictionary of dimension and indexer.
            The indexer can be a slice, a list or an integer value.
        exclude: dict[str, slice | list[int] | int] | None
            Excluded regions from the average as dictionary of dimension and indexer.
            The indexer can be a slice, a list or an integer value.

        Returns
        -------
        PreProcessingPipeline
        """
        return PreProcessingPipeline(
            actions=[*self.actions, CorrectBaselineAverage(exclude=exclude, select=select)]
        )
