
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Import our class and its configuration
from ds_tool import DSTools, DistributionConfig

def analyze_and_compare(generated_data: np.ndarray, config: DistributionConfig):
"""A helper function for analyzing and comparing results."""

# Calculate the actual statistics of the generated data
actual_mean = np.mean(generated_data)
actual_std = np.std(generated_data, ddof=1) # ddof=1 for sample st. deviation
actual_skew = stats.skew(generated_data)
# stats.kurtosis returns the kurtosis (Fisher), and we specified the usual (Pearson).
# Pearson Kurtosis = Fisher Kurtosis + 3
actual_kurtosis = stats.kurtosis(generated_data) + 3
actual_min = np.min(generated_data)
actual_max = np.max(generated_data)

print("\n--- Comparison of target and actual metrics ---")
print(f"| Metric | Target | Actual |")
print(f"|------------|-----------|")
print(f"| Mean | {config.mean:<8.2f} | {actual_mean:<8.2f} |")
print(f"| Std. Dev. | {config.std:<8.2f} | {actual_std:<8.2f} |")
print(f"| Skewness | {config.skewness:<8.2f} | {actual_skew:<8.2f} |") 
print(f"| Kurtosis | {config.kurtosis:<8.2f} | {actual_kurtosis:<8.2f} |") 
print(f"| Min | {config.min_val:<8.2f} | {actual_min:<8.2f} |") 
print(f"| Max | {config.max_val:<8.2f} | {actual_max:<8.2f} |") 
print(f"| N | {config.n:<8} | {len(generated_data):<8} |") 

# Visualization 
plt.figure(figsize=(10, 6)) 
plt.hist(generated_data, bins=50, density=True, alpha=0.7, label='Generated Distribution')
plt.title("Histogram of generated distribution")
plt.grid(True)
plt.show()

# --- Initialize tools ---
tools = DSTools()

# --- Scenario A: Main scenario ---
print("="*60)
print("SCENARIO A: Generate a moderately skewed distribution")
print("="*60)
try:
config_a = DistributionConfig(
mean=1000, median=950, std=200, min_val=400, max_val=2500,
skewness=0.8, kurtosis=4.0, n=2000, outlier_ratio=0.01
)
generated_data_a = tools.generate_distribution(config_a)
analyze_and_compare(generated_data_a, config_a)
except ValueError as e:
print(f"An unexpected error occurred: {e}")

# --- Scenario B: High Kurtosis Scenario ---
print("\n" + "="*60)
print("SCENARIO B: Generating a heavy-tailed (high kurtosis) distribution")
print("="*60)
try:
config_b = DistributionConfig(
mean=50, median=48, std=10, min_val=10, max_val=150,
skewness=1.5, kurtosis=8.0, n=2000, outlier_ratio=0.03
)
generated_data_b = tools.generate_distribution(config_b)
analyze_and_compare(generated_data_b, config_b)
except ValueError as e:
print(f"An unexpected error occurred: {e}")

# --- Scenario B: Checking invalid moments ---
print("\n" + "="*60)
print("SCENARIO B: Trying to generate a distribution with invalid moments")
print("Expecting ValueError because kurtosis < skewness² - 2")
print("="*60)
try:
config_c_invalid = DistributionConfig(
mean=100, median=100, std=15, min_val=50, max_val=150,
skewness=2.0, kurtosis=1.0, n=1000 # Impossible: 1.0 < (2.0² - 2)
)
tools.generate_distribution(config_c_invalid)
except ValueError as e:
print(f"\nSUCCESSFULLY caught the expected error: {e}")

# --- Scenario D: Checking Pydantic validation ---
print("\n" + "="*60)
print("SCENARIO D: Trying to create config with max_val < min_val")
print("Expecting ValueError from Pydantic")
print("="*60)
try:
invalid_pydantic_config = DistributionConfig(
mean=100, median=100, std=15, min_val=200, max_val=100, # Impossible
skewness=0, kurtosis=3, n=1000
)
except ValueError as e:
print(f"\nSUCCESSFULLY caught expected Pydantic error: {e}")