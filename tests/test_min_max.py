import pytest
import pandas as pd
import polars as pl
from src.ds_tool import DSTools

@pytest.fixture
def sample_data_dict():
    return {
        'col_to_scale_1': [10, 20, 30, 40, 50],  # Simple int column
        'col_to_scale_2': [-5.0, 0.0, 5.0, 10.0, 15.0],  # Float with negative values
        'col_constant': [5, 5, 5, 5, 5],  # Constant column
        'col_ignore': ['A', 'B', 'C', 'D', 'E']  # String column, should be ignored
    }

@pytest.fixture
def pd_df(sample_data_dict):
    return pd.DataFrame(sample_data_dict)

@pytest.fixture
def pl_df(sample_data_dict):
    return pl.DataFrame(sample_data_dict)

@pytest.fixture
def tools():
    return DSTools()

# --- Pandas tests ---

def test_min_max_scale_selected_columns_pandas(pd_df, tools):
    # Scale only specified columns in pandas DataFrame
    result = tools.min_max_scale(pd_df, columns=['col_to_scale_1', 'col_to_scale_2'])
    
    # Check scaling to [0,1] for selected columns
    assert result['col_to_scale_1'].min() == 0.0
    assert result['col_to_scale_1'].max() == 1.0
    assert result['col_to_scale_2'].min() == 0.0
    assert result['col_to_scale_2'].max() == 1.0
    
    # Constant column should remain unchanged
    assert (result['col_constant'] == 5).all()

def test_min_max_scale_all_numeric_columns_pandas(pd_df, tools):
    # Scale all numeric columns (including constant) in pandas DataFrame
    result = tools.min_max_scale(pd_df)
    
    # Constant column scaling results in zeros (min == max)
    assert (result['col_constant'] == 0.0).all()

def test_min_max_scale_constant_fill_value_pandas(pd_df, tools):
    # Use custom fill value for constant columns
    fill_val = 0.5
    result = tools.min_max_scale(pd_df, const_val_fill=fill_val)
    
    # Constant column filled with specified value
    assert (result['col_constant'] == fill_val).all()

# --- Polars tests ---

def test_min_max_scale_all_numeric_columns_polars(pl_df, tools):
    # Scale all numeric columns in polars DataFrame
    result = tools.min_max_scale(pl_df)
    
    # Check scaling on first numeric column
    assert result['col_to_scale_1'].min() == 0.0
    assert result['col_to_scale_1'].max() == 1.0
    
    # Constant column should be filled with zeros
    assert result['col_constant'].min() == 0.0
    assert result['col_constant'].max() == 0.0

def test_min_max_scale_constant_fill_value_polars(pl_df, tools):
    # Use custom fill value for constant columns in polars DataFrame
    fill_val = 0.5
    result = tools.min_max_scale(pl_df, const_val_fill=fill_val)
    
    # Constant column filled with specified value
    assert result['col_constant'].min() == fill_val
    assert result['col_constant'].max() == fill_val

# --- Error and warning handling tests ---

def test_min_max_scale_warning_for_nonexistent_column(pd_df, tools, capsys):
    # Calling with non-existent column should print a warning but not crash
    tools.min_max_scale(pd_df, columns=['col_to_scale_1', 'non_existent_col'])
    
    # Capture stdout and verify warning is printed
    captured = capsys.readouterr()
    assert "Warning" in captured.out or "warning" in captured.out.lower()

def test_min_max_scale_type_error_for_wrong_input(tools):
    # Passing unsupported input type should raise TypeError
    with pytest.raises(TypeError):
        tools.min_max_scale([1, 2, 3])
