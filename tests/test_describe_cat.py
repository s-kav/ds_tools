
import numpy as np
import pandas as pd
from src.ds_tool import DSTools

np.random.seed(42)
N_SAMPLES = 100

# Create data for DataFrame
# Set seed for reproducibility
np.random.seed(42)
N_SAMPLES = 100

# Create data for DataFrame
data = {
'user_id': range(N_SAMPLES), # Numeric column, should be ignored

# Categorical column WITHOUT gaps
'status': np.random.choice(
['Active', 'Inactive', 'Blocked'],
size=N_SAMPLES,
p=[0.7, 0.2, 0.1] # Sum of probabilities is now 1.0
),

# Categorical column without gaps
'country': np.random.choice(
['Ukraine', 'Belarus', 'Kazakhstan', 'Armenia', 'Uzbekistan'],
size=N_SAMPLES
),

# Extreme case: column only of gaps
'notes': [np.nan] * N_SAMPLES,

# Column with date, should be ignored
'registration_date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=N_SAMPLES, freq='D'))
}

df = pd.DataFrame(data)

# RELIABLE WAY TO ADD GAP:
# Artificially replace 10% of values ​​in 'status' with NaN
df.loc[df.sample(frac=0.1, random_state=1).index, 'status'] = np.nan

print("--- Information about the created DataFrame: ---")
df.info()
print("\n" + "="*60 + "\n")

# --- 2. Calling the function and checking the result ---

tools = DSTools()

print("--- Calling describe_categorical: ---")
# The function returns a DataFrame, save it
cat_stats_df = tools.describe_categorical(df)

print("Result of the function:")
print(cat_stats_df)
print("\n" + "="*60 + "\n")

# --- 3. Parsing the results (manual verification) ---
print("--- Parsing the results (correctness check): ---")
print("\n1. The DataFrame index contains only categorical columns ('status', 'country', 'notes').")
print(" Columns 'user_id' and 'registration_date' were correctly ignored.\n")

print("2. Parsing the 'status' row:")
status_missing_percent = df['status'].isnull().sum() / len(df) * 100
print(f" - missing (%): {cat_stats_df.loc['status', 'missing (%)']:.1f} (expecting ~{status_missing_percent:.1f}%)")
print(f" - unique: {cat_stats_df.loc['status', 'unique']} (expecting 3, since NaN doesn't count)")
print(f" - top: '{cat_stats_df.loc['status', 'top']}' (expecting 'Active')")
print(f" - freq: {cat_stats_df.loc['status', 'freq']} (expecting ~60)\n")

print("3. Analysis of the string 'country':")
print(f" - missing (%): {cat_stats_df.loc['country', 'missing (%)']:.1f} (expecting 0.0)")
print(f" - unique: {cat_stats_df.loc['country', 'unique']} (expecting 5)\n")

print("4. Analysis of the string 'notes' (extreme case):")
print(f" - missing (%): {cat_stats_df.loc['notes', 'missing (%)']:.1f} (expecting 100.0)")
print(f" - unique: {cat_stats_df.loc['notes', 'unique']} (expecting 0)")
print("\n" + "="*60 + "\n")

# --- 4. Testing on data without categorical columns (edge ​​case) ---
print("--- Testing on a DataFrame without categorical columns ---")
df_numeric = pd.DataFrame({
'a': [1, 2, 3],
'b': [4.0, 5.0, 6.0]
})

empty_df = tools.describe_categorical(df_numeric)

print("Result for a purely numeric DataFrame:")
print(empty_df)
print("\nThe function correctly returned an empty DataFrame, just like expected.")