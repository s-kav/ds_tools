
import pandas as pd
import numpy as np

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data ---

# Set seed for reproducibility
np.random.seed(42)
N_SAMPLES = 1000

# Create data for DataFrame
data = {
# Standard numeric column (int)
'orders_count': np.random.randint(0, 50, size=N_SAMPLES),

# Numeric column (float) with missing values ​​and strong right skew
'revenue': np.random.lognormal(mean=8, sigma=1.5, size=N_SAMPLES),

# Column with negative kurtosis (light tails, flatter than normal)
'uniform_score': np.random.uniform(-1, 1, size=N_SAMPLES),

# Constant column (edge ​​case, std=0)
'api_version': [2] * N_SAMPLES,

# Non-numeric columns to be ignored
'user_segment': np.random.choice(['new', 'active', 'churned'], size=N_SAMPLES),
'last_seen': pd.to_datetime(pd.date_range(start='2023-01-01', periods=N_SAMPLES, freq='h'))
}
df = pd.DataFrame(data)

# Artificially add gaps in the 'revenue' column (10% gaps)
df.loc[df.sample(frac=0.1, random_state=1).index, 'revenue'] = np.nan

print("--- Information about the created DataFrame: ---")
df.info()
print("\n" + "="*60 + "\n")

# --- 2. Calling the function and checking the result ---

tools = DSTools()

print("--- Calling describe_numeric: ---")
# The function returns a DataFrame, save it
num_stats_df = tools.describe_numeric(df)

# Transposing the DataFrame for easy viewing
print("Result of the function (transposed for readability):")
print(num_stats_df.T)
print("\n" + "="*60 + "\n")

# --- 3. Parsing the results (checking correctness) ---
print("--- Parsing results (validation): ---")
print("\n1. DataFrame index contains only numeric columns. 'user_segment' and 'last_seen' are ignored.")

print("\n2. Parsing column 'revenue':")
expected_missing = df['revenue'].isnull().sum() / len(df) * 100
print(f" - missing (%): {num_stats_df.loc['revenue', 'missing (%)']:.1f} (expecting ~{expected_missing:.1f}%)")
print(f" - skew: {num_stats_df.loc['revenue', 'skew']:.2f} (expecting large positive value, > 1)")

print("\n3. Analysis of the 'uniform_score' column:")
print(f" - kurtosis: {num_stats_df.loc['uniform_score', 'kurtosis']:.2f} (expecting a negative value, ~ -1.2)")

print("\n4. Analysis of the 'api_version' column (extreme case):")
print(f" - std: {num_stats_df.loc['api_version', 'std']:.2f} (expecting 0.0, since all values ​​are the same)")
print(f" - min, max, mean, median: all equal to 2.0")
print("\n" + "="*60 + "\n")

# --- 4. Testing on data without numeric columns ---
print("--- Testing on a DataFrame without numeric columns ---")
df_categorical = pd.DataFrame({
'a': ['x', 'y', 'z'],
'b': ['foo', 'bar', 'baz']
})

empty_df = tools.describe_numeric(df_categorical)

print("Result for pure categorical DataFrame:")
print(empty_df)
print("\nThe function correctly returned an empty DataFrame, as expected.")