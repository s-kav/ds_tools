import numpy as np
import pandas as pd
import pytest

from src.ds_tool import DSTools

tools = DSTools()

@pytest.fixture(scope="module")
def sample_dataframe():
    np.random.seed(42)
    N_SAMPLES = 100

    df = pd.DataFrame({
        'user_id': range(N_SAMPLES),
        'status': np.random.choice(['Active', 'Inactive', 'Blocked'], size=N_SAMPLES, p=[0.7, 0.2, 0.1]),
        'country': np.random.choice(['Ukraine', 'Belarus', 'Kazakhstan', 'Armenia', 'Uzbekistan'], size=N_SAMPLES),
        'notes': [np.nan] * N_SAMPLES,
        'registration_date': pd.date_range(start='2023-01-01', periods=N_SAMPLES, freq='D')
    })

    # Add 10% missing to 'status'
    df.loc[df.sample(frac=0.1, random_state=1).index, 'status'] = np.nan
    return df

def test_describe_categorical_structure(sample_dataframe):
    result = tools.describe_categorical(sample_dataframe)

    # Check columns presence
    expected_columns = ['missing (%)', 'unique', 'top', 'freq']
    assert all(col in result.columns for col in expected_columns)

    # Check only categorical columns are included
    assert 'status' in result.index
    assert 'country' in result.index
    assert 'notes' in result.index
    assert 'user_id' not in result.index
    assert 'registration_date' not in result.index

def test_status_column_metrics(sample_dataframe):
    result = tools.describe_categorical(sample_dataframe)

    expected_missing_pct = sample_dataframe['status'].isna().mean() * 100
    assert abs(result.loc['status', 'missing (%)'] - expected_missing_pct) < 1.0

    assert result.loc['status', 'unique'] == 3
    assert result.loc['status', 'top'] in ['Active', 'Inactive', 'Blocked']
    assert result.loc['status', 'freq'] > 0

def test_country_column_metrics(sample_dataframe):
    result = tools.describe_categorical(sample_dataframe)

    assert result.loc['country', 'missing (%)'] == 0.0
    assert result.loc['country', 'unique'] == 5

def test_notes_column_extreme_case(sample_dataframe):
    result = tools.describe_categorical(sample_dataframe)

    assert result.loc['notes', 'missing (%)'] == 100.0
    assert pd.isna(result.loc['notes', 'unique'])

def test_numeric_only_dataframe_returns_empty():
    df_numeric = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4.0, 5.0, 6.0]
    })

    result = tools.describe_categorical(df_numeric)
    assert result.empty
