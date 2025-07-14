import numpy as np
import pandas as pd
import pytest
from src.ds_tool import DSTools

@pytest.fixture
def tools_instance():
    """Provides a DSTools instance for tests."""
    return DSTools()

@pytest.fixture
def sample_dataframe():
    """Creates a sample DataFrame for testing."""
    data = {
        'user_id': range(1, 101),
        'age': np.random.randint(18, 65, 100),
        'city': ['Kyiv'] * 100,
        'balance': np.random.uniform(1000, 10000, 100),
        'registration_date': pd.to_datetime(pd.date_range(start='2022-01-01', periods=100))
    }
    df = pd.DataFrame(data)
   
    df.loc[5:9, 'balance'] = np.nan
    df.loc[15:19, 'age'] = np.nan
    
    df.loc[100] = df.loc[0]
    return df

def test_df_stats_returns_dict(tools_instance, sample_dataframe):
    """
    Test that df_stats returns a dictionary with the expected keys.
    """
    stats_dict = tools_instance.df_stats(sample_dataframe)
    
    assert isinstance(stats_dict, dict), "Function should return a dictionary."
    
    expected_keys = {
        'columns', 'rows', 'missing_percent', 'memory_mb',
        'numeric_columns', 'categorical_columns', 'datetime_columns',
        'duplicated_rows', 'total_missing_values'
    }
    assert expected_keys.issubset(stats_dict.keys()), "One or more required keys are missing in the output."

def test_df_stats_correct_values(tools_instance, sample_dataframe):
    """
    Test that the values calculated by df_stats are correct.
    """

    stats_dict = tools_instance.df_stats(sample_dataframe)    

    expected_cols = 5
    expected_rows = 101 # 100 + 1 дубликат
    expected_total_missing = 10
    expected_missing_percent = np.round(10 / (101 * 5) * 100, 1)
    
    assert stats_dict['columns'] == expected_cols
    assert stats_dict['rows'] == expected_rows
    assert stats_dict['numeric_columns'] == 3 # user_id, age, balance
    assert stats_dict['categorical_columns'] == 1 # city
    assert stats_dict['datetime_columns'] == 1 # registration_date
    assert stats_dict['duplicated_rows'] == 1
    assert stats_dict['total_missing_values'] == expected_total_missing
    
    assert stats_dict['missing_percent'] == pytest.approx(expected_missing_percent)
    assert stats_dict['memory_mb'] >= 0
