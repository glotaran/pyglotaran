"""Tests for ``glotaran.utils.io```."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from glotaran.utils.io import normalize_dataframe_columns
from glotaran.utils.io import safe_dataframe_fillna
from glotaran.utils.io import safe_dataframe_replace


def test_normalize_dataframe_columns():
    """Changes to lower case and renames."""
    input_df = pd.DataFrame([], columns=["Capital", "To_Be_Renamed", "already-lower-case"])
    result_df = normalize_dataframe_columns(input_df, {"to_be_renamed": "renamed"})
    assert all(result_df.columns == ["capital", "renamed", "already-lower-case"])
    result_df_no_rename = normalize_dataframe_columns(input_df)
    assert all(result_df_no_rename.columns == ["capital", "to_be_renamed", "already-lower-case"])
    # check no side effect
    assert_frame_equal(
        input_df, pd.DataFrame([], columns=["Capital", "To_Be_Renamed", "already-lower-case"])
    )


def test_safe_dataframe_fillna():
    """Only replace nan if column is present."""
    input_df = pd.DataFrame([np.nan], columns=["nan_col"])
    result_df = safe_dataframe_fillna(input_df, column_name="nan_col", fill_value=1.1)
    assert_frame_equal(result_df, pd.DataFrame([1.1], columns=["nan_col"]))
    result_df_none_existing_column = safe_dataframe_fillna(
        input_df, column_name="not_a_column", fill_value=1.1
    )
    assert_frame_equal(result_df_none_existing_column, input_df)
    # check no side effect
    assert_frame_equal(input_df, pd.DataFrame([np.nan], columns=["nan_col"]))


def test_safe_dataframe_replace():
    """Only replace value if column is present."""
    input_df = pd.DataFrame([1, 2, 3, 4], columns=["replace_col"])
    result_df = safe_dataframe_replace(
        input_df, column_name="replace_col", to_be_replaced_values=1, replace_value=3
    )
    assert_frame_equal(result_df, pd.DataFrame([3, 2, 3, 4], columns=["replace_col"]))
    result_replace_multiple_df = safe_dataframe_replace(
        input_df, column_name="replace_col", to_be_replaced_values=(1, 2), replace_value=3
    )
    assert_frame_equal(
        result_replace_multiple_df, pd.DataFrame([3, 3, 3, 4], columns=["replace_col"])
    )
    result_df_none_existing_column = safe_dataframe_replace(
        input_df, column_name="not_a_column", to_be_replaced_values=1, replace_value=3
    )
    assert_frame_equal(result_df_none_existing_column, input_df)
    # check no side effect
    assert_frame_equal(input_df, pd.DataFrame([1, 2, 3, 4], columns=["replace_col"]))
