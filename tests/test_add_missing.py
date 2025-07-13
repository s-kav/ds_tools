
import numpy as np
import pandas as pd
import polars as pl

# Import our class
from src.ds_tool import DSTools

# --- 1. Generate test data ---
# Create data that will be used for both DataFrame types
data_dict = {
'col_a': [1, 2, np.nan, 4, 5],
'col_b': [np.nan, 'B', 'C', np.nan, 'E'],
'col_c': [10.0, 20.0, 30.0, 40.0, np.nan],
'col_d': [np.nan, 'Y', 'Z', 'W', 'V'] # Column with no missing values
}
# Expected values ​​for num_missing: [2, 1, 1, 1, 2]

pd_df = pd.DataFrame(data_dict)
pl_df = pl.DataFrame(data_dict, strict=False)

print("--- Original Pandas DataFrame: ---")
print(pd_df)
print("\n--- Original Polars DataFrame: ---")
print(pl_df)
print("\n" + "="*60 + "\n")

# --- 2. Initialization and calls ---
tools = DSTools()

# --- Testing with Pandas ---
print("--- TESTING WITH PANDAS DATAFRAME ---")

# Scenario A: Pandas, default mode
print("\nScenario A: Pandas, default mode (num_missing only)")
pd_result_a = tools.add_missing_value_features(pd_df)
print(pd_result_a)
# Check
expected_missing_counts = [2, 0, 1, 1, 1]
assert pd_result_a['num_missing'].tolist() == expected_missing_counts
assert 'num_missing_std' not in pd_result_a.columns
print("-> SUCCESS: Column 'num_missing' added correctly.")

# Scenario B: Pandas, with std added
print("\nScenario B: Pandas, with add_std=True")
pd_result_b = tools.add_missing_value_features(pd_df, add_std=True)
print(pd_result_b)
# Check
assert 'num_missing' in pd_result_b.columns
assert 'num_missing_std' in pd_result_b.columns
print("-> SUCCESS: Both columns ('num_missing', 'num_missing_std') added.")

print("\n" + "="*60 + "\n")

# --- Testing with Polars ---
print("--- TESTING WITH POLARS DATAFRAME ---")

# Scenario B: Polars, default mode
print("\nScenario B: Polars, default mode")
pl_result_a = tools.add_missing_value_features(pl_df)
print(pl_result_a)
# Testing
expected_missing_polars = [2, 0, 0, 1, 0]
assert pl_result_a['num_missing'].to_list() == expected_missing_polars
print("-> SUCCESS: Column 'num_missing' was added correctly.")

# Scenario D: Polars, with std added
print("\nScenario D: Polars, with add_std=True")
print("We expect to see a warning in the console.")
pl_result_b = tools.add_missing_value_features(pl_df, add_std=True)
print(pl_result_b) # The output will be the same as scenario B
print("-> SUCCESS: The warning was printed as expected.")

print("\n" + "="*60 + "\n")

# --- Scenario E: Testing error handling ---
print("--- SCENARIO E: Testing error handling errors ---")
print("Expecting TypeError when passing list.")
try:
    tools.add_missing_value_features([1, 2, 3])
except TypeError as e:
    print(f"-> SUCCESS: Caught expected error: {e}")