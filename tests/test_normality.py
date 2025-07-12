
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data ---

# Set seed for reproducibility
np.random.seed(42)
N_SAMPLES = 500

# Scenario A: Normally distributed data (Gaussian)
# Mean 50, standard deviation 10
normal_data = pd.Series(np.random.normal(loc=50, scale=10, size=N_SAMPLES), name='Normal_Distribution')

# Scenario B: Uniform distribution
# All values ​​from 0 to 100 are equally likely. Obviously not normal.
uniform_data = pd.Series(np.random.uniform(low=0, high=100, size=N_SAMPLES), name='Uniform_Distribution')

# Scenario B: Exponential Distribution
# Very right skewed. Clearly not normal.
exponential_data = pd.Series(np.random.exponential(scale=15, size=N_SAMPLES), name='Exponential_Distribution')

# --- 2. Initialization and calls ---
tools = DSTools()

# --- Scenario A: Testing on normal data ---
print("="*60)
print("SCENARIO A: TEST ON NORMAL DISTRIBUTION")
print("Expected: p-value > 0.05, output 'Data looks Gaussian'")
print("="*60)

# Call the function. It will print the results and show the graph.
# plt.show() blocks execution, so the graphs will appear one after the other.
tools.stat_normal_testing(normal_data)
print("\n")

# --- Scenario B: Testing on Uniform Distribution ---
print("="*60)
print("SCENARIO B: TEST ON UNIFORM DISTRIBUTION")
print("Expected: p-value < 0.05, output 'Data does not look Gaussian'")
print("="*60)

tools.stat_normal_testing(uniform_data)
print("\n")

# --- Scenario C: Testing on Exponential Distribution with describe_flag=True ---
print("="*60)
print("SCENARIO C: TEST ON EXPONENTIAL DISTRIBUTION")
print("Expected: p-value < 0.05, output 'Data does not look Gaussian'")
print("="*60)

tools.stat_normal_testing(uniform_data)
print("\n")

# --- Scenario C: Testing on Exponential Distribution with describe_flag=True ---
print("="*60)
print("SCENARIO C: TEST ON EXPONENTIAL DISTRIBUTION")
print("Expected: p-value < 0.05, output 'Data does not look Gaussian', and additional graphs")
print("="*60)

tools.stat_normal_testing(exponential_data, describe_flag=True)
print("\n")

# --- Scenario D: Testing with DataFrame ---
print("="*60)
print("SCENARIO D: TEST WITH A SINGLE-COLUMN DATAFRAME")
print("Check that the function works correctly if the input is a DataFrame, not a Series")
print("="*60)

# Convert Series to DataFrame
df_normal = pd.DataFrame(normal_data)
tools.stat_normal_testing(df_normal)
print("\n")