
import pandas as pd
import numpy as np

# Import our class
from src.ds_tool import DSTools

# --- 1. Generate test data ---

# Set seed for reproducibility
np.random.seed(42)

# Create data
data = {
'user_id': range(1, 101),
'age': np.random.randint(18, 65, size=100),
'city': np.random.choice(['Kyiv', 'Kharkiv', 'Sumy', np.nan], size=100, p=[0.5, 0.3, 0.15, 0.05]),
'balance': np.random.uniform(0, 10000, size=100),
'registration_date': pd.to_datetime(pd.date_range(start='2022-01-01', periods=100, freq='D'))
}

df = pd.DataFrame(data)

# Let's artificially add more gaps to the 'balance' column
df.loc[df.sample(n=10, random_state=1).index, 'balance'] = np.nan

print("--- Information about the created DataFrame: ---")
print(df.info())
print("\n" + "="*50 + "\n")

# --- 2. Initialization and call to df_stats ---

tools = DSTools()

print("--- Call to df_stats: ---")
print("We expect to see summary statistics for DataFrame.")
tools.df_stats(df)
print("\n" + "="*50 + "\n")

# --- 3. Manual calculation check for control ---

# Calculate values ​​manually for verification
manual_cols = df.shape[1]
manual_rows = df.shape[0]
manual_missing_count = df.isnull().sum().sum()
manual_total_size = df.size
manual_missing_percent = np.round(manual_missing_count / manual_total_size * 100, 1)
manual_memory_mb = np.round(df.memory_usage(deep=True).sum() / 10**6, 1)

print("--- Manual calculation check: ---")
print(f"Columns: \t{manual_cols}")
print(f"Rows: \t{manual_rows}")
print(f"Gaps (%): \t{manual_missing_percent}%")
print(f"Memory (MB): \t{manual_memory_mb} MB")
