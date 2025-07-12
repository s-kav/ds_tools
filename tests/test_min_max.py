
import numpy as np
import pandas as pd
import polars as pl

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data ---
data_dict = {
'col_to_scale_1': [10, 20, 30, 40, 50], # Simple int column
'col_to_scale_2': [-5.0, 0.0, 5.0, 10.0, 15.0], # Float with negative values
'col_constant': [5, 5, 5, 5, 5], # Constant column
'col_ignore': ['A', 'B', 'C', 'D', 'E'] # String column, should be ignored
}

pd_df = pd.DataFrame(data_dict)
pl_df = pl.DataFrame(data_dict)

print("--- Original Pandas DataFrame: ---")
print(pd_df)
print("\n--- Original Polars DataFrame: ---")
print(pl_df)
print("\n" + "="*60 + "\n")

# --- 2. Initialization and calls ---
tools = DSTools()

# --- Testing with Pandas ---
print("--- TESTING WITH PANDAS DATAFRAME ---")

print("\nScenario A: Scaling selected columns")
pd_result_a = tools.min_max_scale(pd_df, columns=['col_to_scale_1', 'col_to_scale_2'])
print(pd_result_a)
assert pd_result_a['col_to_scale_1'].min() == 0.0 and pd_result_a['col_to_scale_1'].max() == 1.0
assert pd_result_a['col_to_scale_2'].min() == 0.0 and pd_result_a['col_to_scale_2'].max() == 1.0
assert (pd_result_a['col_constant'] == 5).all() # Check that the constant column has not changed
print("-> SUCCESS: Selected columns have been scaled.")

print("\nScenario B: Scale all numeric columns (including the constant)")
pd_result_b = tools.min_max_scale(pd_df) # columns=None
print(pd_result_b)
assert (pd_result_b['col_constant'] == 0.0).all() # Check that the constant column is filled with zeros
print("-> SUCCESS: All numeric columns are scaled, the constant column is filled with zero.")

print("\nScenario B: Checking const_val_fill")
pd_result_c = tools.min_max_scale(pd_df, const_val_fill=0.5)
print(pd_result_c)
assert (pd_result_c['col_constant'] == 0.5).all()
print("-> SUCCESS: The constant column is filled with the value 0.5.")

print("\n" + "="*60 + "\n")

# --- Testing with Polars ---
print("--- TESTING WITH POLARS DATAFRAME ---")

print("\nScenario D: Scaling all numeric columns")
pl_result_a = tools.min_max_scale(pl_df)
print(pl_result_a)
assert pl_result_a['col_to_scale_1'].min() == 0.0 and pl_result_a['col_to_scale_1'].max() == 1.0
assert pl_result_a['col_constant'].min() == 0.0 and pl_result_a['col_constant'].max() == 0.0
print("-> SUCCESS: All numeric columns for Polars scaled.")

print("\nScenario E: Checking const_val_fill for Polars")
pl_result_b = tools.min_max_scale(pl_df, const_val_fill=0.5)
print(pl_result_b)
assert pl_result_b['col_constant'].min() == 0.5 and pl_result_b['col_constant'].max() == 0.5
print("-> SUCCESS: Constant column for Polars filled with value 0.5.")

print("\n" + "="*60 + "\n")

# --- Scenario E: Testing error and warning handling ---
print("--- SCENARIO E: Testing error handling ---")

print("\nAttempt to pass non-existent column (expecting warning):")
# There will be no error, but "Warning: ..." should appear in the console
tools.min_max_scale(pd_df, columns=['col_to_scale_1', 'non_existent_col'])
print("-> SUCCESS: Warning printed, program did not crash.\n")

print("Trying to pass list (expecting TypeError):")
try:
tools.min_max_scale([1, 2, 3])
except TypeError as e:
print(f"-> SUCCESS: Expected error caught: {e}")