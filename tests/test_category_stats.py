import numpy as np
import pandas as pd

# Import our class
from src.ds_tool import DSTools

# --- 1. Generate test data ---

# Set seed for reproducibility
np.random.seed(42)

# Create lists with categories
cities = ['Moscow', 'Saint Petersburg', 'Novosibirsk', 'Ekaterinburg', 'Kazan']
status = ['Active', 'Inactive', 'Pending', 'Archive']
product_type = ['Electronics', 'Clothing', 'Books']

# Generate data
data = {
'City': np.random.choice(cities, size=100, p=[0.4, 0.3, 0.1, 0.1, 0.1]),
'Customer_status': np.random.choice(status + [np.nan], size=100, p=[0.5, 0.2, 0.1, 0.1, 0.1]),
'Product_type': np.random.choice(product_type, size=100)
}

df = pd.DataFrame(data)

print("--- Generated DataFrame (first 10 rows): ---")
print(df.head(10))
print("\n" + "="*50 + "\n")

# --- 2. Call category_stats ---

# Initialize our toolkit
tools = DSTools()

# --- Scenario A: Call for 'City' column ---
print("--- Scenario A: Statistics for the 'City' column ---")
print("Expect to see a table with count and percentage for each city.")
tools.category_stats(df, 'City')
print("\n" + "="*50 + "\n")

# --- Scenario B: Call for the 'Customer_Status' column (with gaps) ---
print("--- Scenario B: Statistics for the 'Customer_status' column ---")
print("Note that missing values ​​(NaN) are not taken into account in the calculations.")
tools.category_stats(df, 'Customer_status')
print("\n" + "="*50 + "\n")

# --- Scenario C: Testing error handling ---
print("--- Scenario C: Attempting to call for a non-existent column ---")
print("Expecting to see ValueError.")

try:
    tools.category_stats(df, 'Non-existent_column')
except ValueError as e:
    print(f"\nExpected error successfully caught: {e}")

print("\n" + "="*50)
