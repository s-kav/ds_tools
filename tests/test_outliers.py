
import pandas as pd
import numpy as np

# Import our class and its configuration
from src.ds_tool import DSTools, OutlierConfig

# --- 1. Generate test data ---

# Set seed for reproducibility
np.random.seed(42)

# Create "normal" data
normal_data = np.random.normal(loc=100, scale=20, size=90)

# Create obvious outliers
outliers = np.array([-50, -40, 250, 300, 310])

# Concatenate and shuffle
full_data = np.concatenate([normal_data, outliers])
np.random.shuffle(full_data)

# Create DataFrame
df = pd.DataFrame({'value': full_data})
# Add a categorical column to check if the entire row is deleted
df['category'] = np.random.choice(['A', 'B'], size=len(df))

print("--- Original DataFrame ---")
print("Statistics (note min/max and std):")
print(df['value'].describe())
print("\n" + "="*50 + "\n")

# Calculate IQR bounds manually to check (with sigma=1.5)
q1 = df['value'].quantile(0.25)
q3 = df['value'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
print(f"Expected bounds for sigma=1.5: Lower ~{lower_bound:.2f}, Upper ~{upper_bound:.2f}")
print("Anything less or greater than these bounds will be considered an outlier.")
print("\n" + "="*50 + "\n")

# --- 2. Initialization and calls ---
tools = DSTools()
target_column = 'value'

# --- Scenario A: Default mode (replacing outliers) ---
print(f"--- Scenario A: Replacing outliers in column '{target_column}' (sigma=1.5) ---")
# Using default config
df_replaced, p_upper, p_lower = tools.remove_outliers_iqr(df.copy(), target_column)

print(f"\nFound upper outliers: {p_upper}%")
print(f"Found lower outliers: {p_lower}%")
print("\nStatistics AFTER replacement (min/max should be equal to bounds):")
print(df_replaced['value'].describe())
print("\n" + "="*50 + "\n")

# --- Scenario B: Row removal mode ---
print(f"--- Scenario B: Removing rows with outliers in column '{target_column}' ---")
config_remove = OutlierConfig(change_remove=False)
df_removed, _, _ = tools.remove_outliers_iqr(df.copy(), target_column, config=config_remove)

print(f"\nSize of original DataFrame: {df.shape}")
print(f"DataFrame size AFTER rows are removed: {df_removed.shape}")
print("\nStats AFTER removal (min/max should be within normal limits):")
print(df_removed['value'].describe())
print("\n" + "="*50 + "\n")

# --- Scenario B: Custom sigma and no percentage return ---
print(f"--- Scenario B: Replacing outliers with stricter sigma=1.0 and no percentage return ---")
# Stricter sigma will find more outliers
config_custom = OutlierConfig(sigma=1.0, percentage=False)
# The function will return only the DataFrame, not a tuple
df_strict = tools.remove_outliers_iqr(df.copy(), target_column, config=config_custom)

print(f"\nReturn object type: {type(df_strict)}")
print("Check that this is a DataFrame and not a tuple was successful.")
print("\nStatistics AFTER replacement (boundaries are narrower than in scenario A):")
print(df_strict['value'].describe())
print("\n" + "="*50 + "\n")

# --- Scenario D: Testing error handling ---
print("--- Scenario D: Attempting to call on a non-existent column ---")

try:
    tools.remove_outliers_iqr(df, 'non_existent_column')
except ValueError as e:
    print(f"\nExpected error successfully caught: {e}")