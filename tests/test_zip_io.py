
import os
import numpy as np
import pandas as pd
import polars as pl

# Import our class
from src.ds_tool import DSTools

# --- 1. Generate test data ---
print("--- 1. Generate test DataFrame ---")

# Create Pandas DataFrame with custom index
pd_index = pd.Index(['id_1', 'id_2', 'id_3', 'id_4'], name='custom_index')
pd_df = pd.DataFrame(
{'A': [1, 2, 3, 4], 'B': ['x', 'y', 'z', 'w']},
index=pd_index
)

# Create Polars DataFrame
pl_df = pl.DataFrame(
{'C': [10.5, 20.5, 30.5], 'D': [True, False, True]}
)

# Dictionary to save
dfs_to_save = {
'pandas_data': pd_df,
'polars_data': pl_df
}

print("Original Pandas DF:\n", pd_df)
print("\nOriginal Polars DF:\n", pl_df)
print("\n" + "="*60 + "\n")

# --- 2. Initialization and calls ---
tools = DSTools()
ZIP_FILENAME = 'test_archive.zip'

# --- Scenario A: Full loop with Parquet and index saving ---
print("--- SCENARIO A: Saving/Reading in Parquet Format ---")

# --- Step 2.1: Saving ---
print(f"\n2.1. Saving to '{ZIP_FILENAME}' with save_index=True...")
tools.save_dataframes_to_zip(
dataframes=dfs_to_save,
zip_filename=ZIP_FILENAME,
format='parquet',
save_index=True
)
assert os.path.exists(ZIP_FILENAME)
print("-> SUCCESS: ZIP archive created.")

# --- Step 2.2: Reading with Polars ---
print("\n2.2. Reading with Polars backend...")
loaded_with_polars = tools.read_dataframes_from_zip(
zip_filename=ZIP_FILENAME,
backend='polars'
)
# Polars data comparison
# .equals() checks for exact data and type match
assert loaded_with_polars['polars_data'].equals(pl_df)
print("-> SUCCESS: Polars DataFrame restored correctly.")

# --- Step 2.3: Reading with Pandas ---
print("\n2.3. Reading with Pandas backend...")
loaded_with_pandas = tools.read_dataframes_from_zip(
zip_filename=ZIP_FILENAME,
backend='pandas'
)
# Pandas data comparison
# pd.testing.assert_frame_equal() checks for exact match, including index
pd.testing.assert_frame_equal(loaded_with_pandas['pandas_data'], pd_df)
print("-> SUCCESS: Pandas DataFrame restored correctly, including custom index.")

# --- Step 2.4: Cleanup ---
os.remove(ZIP_FILENAME)
print(f"\n2.4. Archive '{ZIP_FILENAME}' removed.")
print("\n" + "="*60 + "\n")

# --- Scenario B: Full Cycle with CSV ---
print("--- SCENARIO B: Save/Read in CSV Format ---")
CSV_ZIP_FILENAME = 'test_archive_csv.zip'

# --- Step 3.1: Save ---
print(f"\n3.1. Save to '{CSV_ZIP_FILENAME}' in CSV format...")
tools.save_dataframes_to_zip(
dataframes=dfs_to_save,
zip_filename=CSV_ZIP_FILENAME,
format='csv',
save_index=True
)
assert os.path.exists(CSV_ZIP_FILENAME)
print("-> SUCCESS: CSV ZIP archive created.")

# --- Step 3.2: Reading ---
print("\n3.2. Reading CSV archive with Pandas...")
loaded_csv = tools.read_dataframes_from_zip(
zip_filename=CSV_ZIP_FILENAME,
format='csv',
backend='pandas'
)
# When reading from CSV, the index becomes a normal one column
# Check that the data is generally the same (resetting the index for comparison)
pd.testing.assert_frame_equal(
loaded_csv['pandas_data'].reset_index(drop=True),
pd_df.reset_index(drop=True)
)
print("-> SUCCESS: Data from CSV recovered (taking into account format peculiarities).")

# --- Step 3.3: Cleaning ---
os.remove(CSV_ZIP_FILENAME)
print(f"\n3.3. Archive '{CSV_ZIP_FILENAME}' deleted.")
print("\n" + "="*60 + "\n")

print("All tests passed successfully!")