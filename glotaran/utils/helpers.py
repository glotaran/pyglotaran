"""Module containing general helper functions."""

from typing import Any

import numpy as np


def nan_or_equal(lhs: Any, rhs: Any) -> bool:
    """Compare values which can be nan for equality.

    This helper function is needed because ``np.nan == np.nan`` returns ``False``.

    Parameters
    ----------
    lhs: Any
        Left hand side value.
    rhs: Any
        Right hand side value.

    Returns
    -------
    bool
        Whether or not values are equal.
    """
    if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        return (np.isnan(lhs) and np.isnan(rhs)) or lhs == rhs
    return lhs == rhs
