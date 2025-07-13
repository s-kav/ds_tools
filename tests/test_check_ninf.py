
import pandas as pd
import numpy as np

# Import our class
from src.ds_tool import DSTools

# --- 1. Generate test data ---

# Create base data for each case
data_clean = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
data_with_nan = {'col1': [1, np.nan, 3], 'col2': [4, 5, 6]}
data_with_inf = {'col1': [1, 2, 3], 'col2': [4, np.inf, 6]}
data_with_both = {'col1': [1, np.nan, 3], 'col2': [-np.inf, 5, 6]}

# Create DataFrames
df_clean = pd.DataFrame(data_clean)
df_with_nan = pd.DataFrame(data_with_nan)
df_with_inf = pd.DataFrame(data_with_inf)
df_with_both = pd.DataFrame(data_with_both)

# Create NumPy arrays
arr_clean = df_clean.values
arr_with_nan = df_with_nan.values
arr_with_inf = df_with_inf.values
arr_with_both = df_with_both.values


# --- 2. Initialization and calls ---
tools = DSTools()

# --- Testing with pandas.DataFrame ---
print("="*50)
print("TESTING WITH PANDAS DATAFRAME")
print("="*50)

print("\n--- Case 1: Clean DataFrame ---")
print("Expected output: 'Dataset has no NaN or infinite values'")
tools.check_NINF(df_clean)

print("\n--- Case 2: DataFrame with NaN ---")
print("Expected output: 'Dataset has NaN values ​​but no infinite values'")
tools.check_NINF(df_with_nan)

print("\n--- Case 3: DataFrame with Inf ---")
print("Expected output: 'Dataset has infinite values ​​but no NaN values'")
tools.check_NINF(df_with_inf)

print("\n--- Case 4: DataFrame with NaN and Inf ---")
print("Expected output: 'Dataset has both NaN and infinite values'")
tools.check_NINF(df_with_both)

# --- Testing with numpy.ndarray ---
print("\n\n" + "="*50)
print("TESTING WITH NUMPY NDARRAY")
print("="*50)

print("\n--- Case 1: Clean array ---")
print("Expected output: 'Dataset has no NaN or infinite values'")
tools.check_NINF(arr_clean)

print("\n--- Case 2: Array with NaN ---")
print("Expected output: 'Dataset has NaN values ​​but no infinite values'")
tools.check_NINF(arr_with_nan)

print("\n--- Case 3: Array with Inf ---")
print("Expected output: 'Dataset has infinite values ​​but no NaN values'")
tools.check_NINF(arr_with_inf)

print("\n--- Case 4: Array with NaN and Inf ---")
print("Expected output: 'Dataset has both NaN and infinite values'")
tools.check_NINF(arr_with_both)

print("\n\n" + "="*50)
