"""A pre-processor pipeline for data."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict

if TYPE_CHECKING:
    import xarray as xr


class PreProcessor(BaseModel, abc.ABC):
    """A base class for pre=processors."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abc.abstractmethod
    def apply(self, data: xr.DataArray) -> xr.DataArray:
        """Apply the pre-processor.

        Parameters
        ----------
        data: xr.DataArray
            The data to process.

        Returns
        -------
        xr.DataArray

        .. # noqa: DAR202
        """


class CorrectBaselineValue(PreProcessor):
    """Corrects a dataset by subtracting baseline value."""

    action: Literal["baseline-value"] = "baseline-value"
    value: float

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        """Apply the pre-processor.

        Parameters
        ----------
        data: xr.DataArray
            The data to process.

        Returns
        -------
        xr.DataArray
        """
        return data - self.value


class CorrectBaselineAverage(PreProcessor):
    """Corrects a dataset by subtracting the average over a part of the data."""

    action: Literal["baseline-average"] = "baseline-average"
    select: dict[str, slice | list[int] | int] | None = None
    exclude: dict[str, slice | list[int] | int] | None = None

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        """Apply the pre-processor.

        Parameters
        ----------
        data: xr.DataArray
            The data to process.

        Returns
        -------
        xr.DataArray
        """
        return data - data.sel(self.select or {}).drop_sel(self.exclude or {}).mean()
