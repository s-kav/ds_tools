import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.ds_tool import DSTools

tools = DSTools()

# --- Fixtures for test data ---
@pytest.fixture
def pd_df():
    return pd.DataFrame({
        'col_a': [1, 2, np.nan, 4, 5],
        'col_b': [np.nan, 'B', 'C', np.nan, 'E'],
        'col_c': [10.0, 20.0, 30.0, 40.0, np.nan],
        'col_d': [np.nan, 'Y', 'Z', 'W', 'V']
    })

@pytest.fixture
def pl_df():
    return pl.DataFrame({
        'col_a': [1, 2, None, 4, 5],
        'col_b': [None, 'B', 'C', None, 'E'],
        'col_c': [10.0, 20.0, 30.0, 40.0, None],
        'col_d': [None, 'Y', 'Z', 'W', 'V']
    }, strict=False)

# --- Pandas tests ---

def test_pandas_default(pd_df):
    result = tools.add_missing_value_features(pd_df)
    expected = [2, 0, 1, 1, 1]
    assert result['num_missing'].tolist() == expected
    assert 'num_missing_std' not in result.columns

def test_pandas_with_std(pd_df):
    result = tools.add_missing_value_features(pd_df, add_std=True)
    assert 'num_missing' in result.columns
    assert 'num_missing_std' in result.columns

# --- Polars tests ---

def test_polars_default(pl_df):
    result = tools.add_missing_value_features(pl_df)
    expected = [2, 0, 1, 1, 1]
    assert result['num_missing'].to_list() == expected

def test_polars_with_std_warns(pl_df):
    result = tools.add_missing_value_features(pl_df, add_std=True)
    assert 'num_missing' in result.columns

# --- Error handling ---

def test_invalid_input_type():
    with pytest.raises(TypeError):
        tools.add_missing_value_features([1, 2, 3])
