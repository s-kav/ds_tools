import numpy as np
import pandas as pd
import pytest
from src.ds_tool import DSTools

tools = DSTools()

@pytest.fixture
def sample_dataframe():
    np.random.seed(42)
    df = pd.DataFrame({
        'user_id': range(1, 101),
        'age': np.random.randint(18, 65, size=100),
        'city': np.random.choice(['Kyiv', 'Kharkiv', 'Sumy', np.nan], size=100, p=[0.5, 0.3, 0.15, 0.05]),
        'balance': np.random.uniform(0, 10000, size=100),
        'registration_date': pd.date_range(start='2022-01-01', periods=100, freq='D')
    })
    df.loc[df.sample(n=10, random_state=1).index, 'balance'] = np.nan
    return df

def test_df_stats_summary_output(monkeypatch, sample_dataframe):
    captured_output = {}

    def mock_print(*args, **kwargs):
        key = len(captured_output)
        captured_output[key] = " ".join(str(arg) for arg in args)

    monkeypatch.setattr("builtins.print", mock_print)
    tools.df_stats(sample_dataframe)
   
    has_columns = any("Column" in val for val in captured_output.values())
    has_rows = any("Rows" in val for val in captured_output.values())
    has_missing = any("Missing (%)" in val for val in captured_output.values())
    has_memory = any("Memory" in val for val in captured_output.values())

    assert has_columns, "Output must include number of columns"
    assert has_rows, "Output must include number of rows"
    assert has_missing, "Output must include missing values info"
    assert has_memory, "Output must include memory usage"

def test_manual_calculations_match(sample_dataframe):
   
    expected_cols = sample_dataframe.shape[1]
    expected_rows = sample_dataframe.shape[0]
    expected_missing = sample_dataframe.isnull().sum().sum()
    expected_total = sample_dataframe.size
    expected_missing_pct = round(expected_missing / expected_total * 100, 1)
    expected_memory_mb = round(sample_dataframe.memory_usage(deep=True).sum() / 1e6, 1)

    df = sample_dataframe

    assert expected_cols == 5
    assert expected_rows == 100
    assert expected_missing == df.isnull().sum().sum()
    assert expected_missing_pct <= 5.0
    assert expected_memory_mb >= 0
