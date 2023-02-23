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

    def _push_action(self, action: PipelineAction):
        """Push an action.

        Parameters
        ----------
        action: PipelineAction
            The action to push.
        """
        self.actions.append(action)

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
        self._push_action(CorrectBaselineValue(value=value))
        return self

    def correct_baseline_average(
        self, selection: dict[str, slice | list[int] | int]
    ) -> PreProcessingPipeline:
        """Correct a dataset by subtracting the average over a part of the data.

        Parameters
        ----------
        selection: dict[str, slice | list[int] | int]
            The selection to average as dictionary of dimension and indexer.
            The indexer can be a slice, a list or an integer value.

        Returns
        -------
        PreProcessingPipeline
        """
        self._push_action(CorrectBaselineAverage(selection=selection))
        return self
