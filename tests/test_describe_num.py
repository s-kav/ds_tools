import numpy as np
import pandas as pd
import pytest
from src.ds_tool import DSTools

tools = DSTools()

@pytest.fixture(scope="module")
def numeric_dataframe():
    np.random.seed(42)
    N_SAMPLES = 1000

    df = pd.DataFrame({
        'orders_count': np.random.randint(0, 50, size=N_SAMPLES),
        'revenue': np.random.lognormal(mean=8, sigma=1.5, size=N_SAMPLES),
        'uniform_score': np.random.uniform(-1, 1, size=N_SAMPLES),
        'api_version': [2] * N_SAMPLES,
        'user_segment': np.random.choice(['new', 'active', 'churned'], size=N_SAMPLES),
        'last_seen': pd.date_range(start='2023-01-01', periods=N_SAMPLES, freq='h')
    })

    df.loc[df.sample(frac=0.1, random_state=1).index, 'revenue'] = np.nan
    return df

def test_numeric_columns_presence(numeric_dataframe):
    result = tools.describe_numeric(numeric_dataframe)
    assert set(result.index) == {'orders_count', 'revenue', 'uniform_score', 'api_version'}
    assert 'user_segment' not in result.index
    assert 'last_seen' not in result.index

def test_revenue_column_metrics(numeric_dataframe):
    result = tools.describe_numeric(numeric_dataframe)

    expected_missing_pct = numeric_dataframe['revenue'].isna().mean() * 100
    actual_missing_pct = result.loc['revenue', 'missing (%)']
    assert abs(actual_missing_pct - expected_missing_pct) < 1.0

    skewness = result.loc['revenue', 'skew']
    assert skewness > 1.0

def test_uniform_score_kurtosis(numeric_dataframe):
    result = tools.describe_numeric(numeric_dataframe)
    kurtosis = result.loc['uniform_score', 'kurtosis']
    assert kurtosis < -1.0  # Uniform distribution has platykurtic shape

def test_api_version_constant_column(numeric_dataframe):
    result = tools.describe_numeric(numeric_dataframe)
    row = result.loc['api_version']

    assert row['std'] == 0.0
    assert row['min'] == 2.0
    assert row['max'] == 2.0
    assert row['mean'] == 2.0
    assert row['median'] == 2.0

def test_empty_result_on_non_numeric_dataframe():
    df = pd.DataFrame({
        'a': ['foo', 'bar', 'baz'],
        'b': ['one', 'two', 'three']
    })

    result = tools.describe_numeric(df)
    assert result.empty
