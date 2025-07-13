
import pandas as pd
import numpy as np

# Import our class
from src.ds_tool import DSTools

# --- 1. Generate test data ---

# Create a list of categories with different frequencies
# 'C' is the most frequent, 'B' is average, 'A' is the rarest, 'D' is very rare
categories_list = ['C'] * 50 + ['B'] * 30 + ['A'] * 15 + ['D'] * 5

# Shuffle them so they are not in order
np.random.seed(42)
np.random.shuffle(categories_list)

df = pd.DataFrame({'product_category': categories_list})

print("--- Original DataFrame (first 10 rows): ---")
print(df.head(10))
print("\n--- Category frequency (for checking): ---")
print(df['product_category'].value_counts())
print("\n" + "="*50 + "\n")

# --- 2. Initialization and calling labeling ---

tools = DSTools()
target_column = 'product_category'

# --- Scenario A: Encoding with ordering by frequency (order_flag=True) ---
print(f"--- Scenario A: Encoding column '{target_column}' with order_flag=True ---")
print("We expect the rarest category 'D' to be coded 0, 'A' -> 1, 'B' -> 2, 'C' -> 3.")

# The function returns a new DataFrame, so we save the result
df_ordered = tools.labeling(df, target_column, order_flag=True)

print("\n--- Result of encoding (order_flag=True): ---")
print(df_ordered.head(10))

print("\n--- Result check (match codes and original values): ---")
# Let's create a table for visual comparison
comparison_ordered = pd.DataFrame({
'original': df[target_column],
'encoded': df_ordered[target_column]
})
print(comparison_ordered.drop_duplicates().sort_values('encoded'))
print("\n" + "="*50 + "\n")

# --- Scenario B: Encoding without ordering (order_flag=False) ---
print(f"--- Scenario B: Encoding column '{target_column}' with order_flag=False ---")
print("Expect categories to receive arbitrary numeric codes.")

df_simple = tools.labeling(df, target_column, order_flag=False)

print("\n--- Result of encoding (order_flag=False): ---")
print(df_simple.head(10))

print("\n--- Checking the result (matching codes and original values): ---")
comparison_simple = pd.DataFrame({
'original': df[target_column],
'encoded': df_simple[target_column]
})
print(comparison_simple.drop_duplicates().sort_values('encoded'))
print("\n" + "="*50 + "\n")

# --- Scenario B: Testing error handling ---
print("--- Scenario B: Attempting to call on non-existent column ---")
print("Expecting to see ValueError.")

try:
    tools.labeling(df, 'non_existent_column')
except ValueError as e:
    print(f"\nSuccessfully caught expected error: {e}")