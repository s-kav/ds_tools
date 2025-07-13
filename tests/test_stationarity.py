import pytest
import numpy as np
import pandas as pd
from src.ds_tool import DSTools

@pytest.fixture(scope="module")
def tools():
    return DSTools()

@pytest.fixture(scope="module")
def time_series_data():
    np.random.seed(42)
    n = 365
    time_index = pd.date_range(start='2023-01-01', periods=n, freq='D')
    
    stationary = pd.Series(
        np.random.normal(loc=10, scale=5, size=n),
        index=time_index,
        name='Stationary_Series'
    )
    trend = np.arange(n) * 0.5
    non_stationary_trend = pd.Series(
        trend + np.random.normal(loc=0, scale=10, size=n),
        index=time_index,
        name='NonStationary_Trend_Series'
    )
    seasonality = 15 * np.sin(np.arange(n) * (2 * np.pi / 30))
    non_stationary_seasonal = pd.Series(
        trend + seasonality + np.random.normal(loc=0, scale=5, size=n),
        index=time_index,
        name='NonStationary_Seasonal_Series'
    )
    return {
        'stationary': stationary,
        'non_stationary_trend': non_stationary_trend,
        'non_stationary_seasonal': non_stationary_seasonal
    }

# Use matplotlib non-interactive backend to prevent blocking
import matplotlib
matplotlib.use('Agg')

def test_stationarity_stationary_series(tools, time_series_data, capsys):
    tools.test_stationarity(time_series_data['stationary'])
    captured = capsys.readouterr()
    assert "p-value" in captured.out
    assert "STATIONARY" in captured.out or "stationary" in captured.out

def test_stationarity_non_stationary_trend(tools, time_series_data, capsys):
    tools.test_stationarity(time_series_data['non_stationary_trend'])
    captured = capsys.readouterr()
    assert "p-value" in captured.out
    assert "NON-STATIONARY" in captured.out or "non-stationary" in captured.out

def test_stationarity_non_stationary_seasonal(tools, time_series_data, capsys):
    tools.test_stationarity(time_series_data['non_stationary_seasonal'], len_window=60)
    captured = capsys.readouterr()
    assert "p-value" in captured.out
    assert "NON-STATIONARY" in captured.out or "non-stationary" in captured.out

def test_stationarity_with_print_results_flag_false(tools, time_series_data, capsys):
    tools.test_stationarity(time_series_data['stationary'], print_results_flag=False)
    captured = capsys.readouterr()
    # Expect no detailed test report, so p-value string should be absent or minimal output
    assert "p-value" not in captured.out or captured.out.strip() == ""
