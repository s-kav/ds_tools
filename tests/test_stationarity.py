
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import our class
from src.ds_tool import DSTools

# --- 1. Generate test data ---

# Set seed for reproducibility
np.random.seed(42)
N_SAMPLES = 365 # Year of data

# Create time index
time_index = pd.date_range(start='2023-01-01', periods=N_SAMPLES, freq='D')

# Scenario A: Stationary series (white noise)
# Mean and variance are constant.
stationary_series = pd.Series(
np.random.normal(loc=10, scale=5, size=N_SAMPLES),
index=time_index,
name='Stationary_Series'
)

# Scenario B: Non-stationary series (trend only)
# Mean increases over time.
trend = np.arange(N_SAMPLES) * 0.5
non_stationary_trend_series = pd.Series(
trend + np.random.normal(loc=0, scale=10, size=N_SAMPLES), # trend + noise
index=time_index,
name='NonStationary_Trend_Series'
)

# Scenario B: Non-stationary series (trend + seasonality)
# Simulate monthly seasonality (period ~30 days)
seasonality = 15 * np.sin(np.arange(N_SAMPLES) * (2 * np.pi / 30))
non_stationary_seasonal_series = pd.Series(
trend + seasonality + np.random.normal(loc=0, scale=5, size=N_SAMPLES),
index=time_index,
name='NonStationary_Seasonal_Series'
)

# --- 2. Initialization and calls ---
tools = DSTools()

# --- Scenario A: Test on stationary data ---
print("="*60)
print("SCENARIO A: TEST ON STATIONARY DATA (white noise)")
print("Expected: p-value < 0.05, output 'Data is STATIONARY!'")
print("On the graph, the moving average and st.deviation should be horizontal.")
print("="*60)
# IMPORTANT: After each call, a graph will appear.
# The script will continue to run only after you close the window with the graph.
tools.test_stationarity(stationary_series)
print("\n")

# --- Scenario B: Test on data with trend ---
print("="*60)
print("SCENARIO B: TEST ON NON-STATIONARY DATA (trend)")
print("Expected: p-value > 0.05, output 'Data is NON-STATIONARY!'")
print("On the chart, the moving average will increase along with the data.")
print("="*60)
tools.test_stationarity(non_stationary_trend_series)
print("\n")

# --- Scenario C: Test on data with trend and seasonality ---
print("="*60)
print("SCENARIO C: TEST ON NON-STATIONARY DATA (trend + seasonality)")
print("Expected: p-value > 0.05, output 'Data is NON-STATIONARY!'")
print("On the chart, the moving average will increase, and the standard deviation will fluctuate.")
print("="*60)
tools.test_stationarity(non_stationary_seasonal_series, len_window=60) # Let's increase the window for clarity
print("\n")

# --- Scenario D: Test the flag print_results_flag=False ---
print("="*60)
print("SCENARIO D: TEST WITH print_results_flag=False")
print("Expected: The chart will appear, but the detailed Dickey-Fuller test report will NOT be printed.")
print("="*60)
tools.test_stationarity(stationary_series, print_results_flag=False)
print("\n")