
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
import matplotlib.pyplot as plt

# Import our class and its configuration
from src.ds_tool import DSTools, DistributionConfig

def test_analyze_and_compare(generated_data, config: DistributionConfig, title: str):
    """A helper function for analyzing and comparing results."""
    # Convert data to a NumPy array for calculations if it is a Series
    if isinstance(generated_data, (pd.Series, pl.Series)):
        data_arr = generated_data.to_numpy()
    else:
        data_arr = generated_data

    actual_mean = np.mean(data_arr)
    actual_std = np.std(data_arr, ddof=1)
    actual_median = np.median(data_arr)
    actual_skew = stats.skew(data_arr)
    # stats.kurtosis returns (Fisher) kurtosis. We want full (Pearson) kurtosis.
    # But your function seems to use kurtosis, so we'll keep it.
    # If Pearson were required, it would be: stats.kurtosis(data_arr) + 3
    actual_kurtosis = stats.kurtosis(data_arr, fisher=True) # fisher=True is kurtosis
    actual_min = np.min(data_arr)
    actual_max = np.max(data_arr)

    print(f"\n--- {title} ---")
    print(f"| Metric | Goal | Fact |")
    print(f"|------------|------------|")
    print(f"| Mean | {config.mean:<10.2f} | {actual_mean:<10.2f} |")
    print(f"| Median | {config.median:<10.2f} | {actual_median:<10.2f} |") 
    print(f"| Std. Dev. | {config.std:<10.2f} | {actual_std:<10.2f} |") 
    print(f"| Skewness | {config.skewness:<10.2f} | {actual_skew:<10.2f} |") 
    print(f"| Kurtosis | {config.kurtosis:<10.2f} | {actual_kurtosis:<10.2f} |") 
    print(f"| Min | {config.min_val:<10.2f} | {actual_min:<10.2f} |") 
    print(f"| Max | {config.max_val:<10.2f} | {actual_max:<10.2f} |") 

    plt.figure(figsize=(10, 6))
    plt.hist(data_arr, bins=50, density=True, alpha=0.7, label='Generated Distribution')
    plt.title(title)
    plt.grid(True)
    plt.show()

# --- Initialize tools ---
tools = DSTools()

# --- Scenario A: Main call with DistributionConfig object ---
print("="*60)
print("SCENARIO A: Generate from DistributionConfig object, NumPy output")
print("="*60)
config_a = DistributionConfig(
mean=500, median=450, std=150, min_val=100, max_val=2000,
skewness=1.2, kurtosis=5.0, n=5000, outlier_ratio=0.02
)
data_a = tools.generate_distribution_from_metrics(n=5000, metrics=config_a)
test_analyze_and_compare(data_a, config_a, "Scenario A: Result")

# --- Scenario B: Call with dictionary and output to Pandas Series (int) ---
print("\n" + "="*60)
print("SCENARIO B: Generate from dictionary, output Pandas Series (int)")
print("="*60)
metrics_dict_b = {
"mean": 80.0, "median": 75.0, "std": 20.0, "min_val": 10, "max_val": 150,
"skewness": 0.5, "kurtosis": 0.8, "n": 3000, "outlier_ratio": 0.01
}
data_b = tools.generate_distribution_from_metrics(
n=3000,
metrics=metrics_dict_b,
int_flag=True,
output_as='pandas'
)
print("Output type:", type(data_b))
print("Data type in Series:", data_b.dtype)
print(data_b.head())
# For analysis, we pass the Series itself, the function will sort it out
test_analyze_and_compare(data_b, DistributionConfig(**metrics_dict_b), "Scenario B: Result")

# --- Scenario C: Testing moment validation ---
print("\n" + "="*60)
print("SCENARIO C: Trying to create a distribution with impossible moments")
print("Expecting ValueError, since kurtosis < skewness² - 2")
print("="*60)
try:
    invalid_moments = {
    "mean": 100, "median": 100, "std": 15, "min_val": 50, "max_val": 150,
    "skewness": 3.0, "kurtosis": 1.0, "n": 1000 # Impossible: 1.0 < (3.0² - 2)
    }
    tools.generate_distribution_from_metrics(n=1000, metrics=invalid_moments)
except ValueError as e:
    print(f"SUCCESSFULLY caught expected error: {e}")

# --- Scenario D: Testing Pydantic validation ---
print("\n" + "="*60)
print("SCENARIO D: Attempting to create config with invalid data type in dictionary")
print("Expecting ValueError from Pydantic")
print("="*60)
try:
    invalid_pydantic = {
    "mean": 100, "median": "not a number", "std": 15, "min_val": 50, "max_val": 150,
    "skewness": 0, "kurtosis": 3, "n": 1000
    }
    tools.generate_distribution_from_metrics(n=1000, metrics=invalid_pydantic)
except ValueError as e:
    print(f"SUCCESSFULLY caught expected Pydantic error: {e}")
