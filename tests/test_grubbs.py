
import numpy as np
import pandas as pd

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data ---

# Set seed for reproducibility
np.random.seed(42)

# Data WITHOUT obvious outliers (from normal distribution)
data_normal = np.random.normal(loc=100, scale=10, size=30)

# Data WITH OBVIOUS outlier. Take normal data and add one extreme value.
data_with_outlier = np.append(data_normal, 150)

# Data where all values ​​are the same
data_constant = np.full(10, 50.0)

# --- 2. Initialization and calls ---
tools = DSTools()

# --- Scenario A: Test on data WITH outlier ---
print("="*60)
print("SCENARIO A: Test on data WITH outlier (value 200)")
print("Expected: is_outlier=True, outlier_value=200")
print("="*60)

result_a = tools.grubbs_test(data_with_outlier)
print(f"Result of Pydantic model: {result_a}")
if result_a.is_outlier:
print(f"-> SUCCESS: Outlier found! Value: {result_a.outlier_value}, Index: {result_a.outlier_index}")
else:
print("-> ERROR: Outlier was not found.")
print("\n")

# --- Scenario B: Test on data WITHOUT outliers ---
print("="*60)
print("SCENARIO B: Test on 'pure' normal data")
print("Expected: is_outlier=False")
print("="*60)

result_b = tools.grubbs_test(data_normal)
print(f"Pydantic model result: {result_b}")
if not result_b.is_outlier:
print(f"-> SUCCESS: Outlier was not found, as expected.")
else:
print("-> ERROR: False outlier.")
print("\n")

# --- Scenario B: Test with different alpha ---
print("="*60)
print("SCENARIO B: Test on data with MODERATE outlier (150)")
print("Expect: Outlier found first (alpha=0.05), then not (alpha=0.01)")
print("="*60)

# First, let's test with default alpha=0.05. Outlier should be found.
print("\nStep 1: alpha=0.05")
result_c1 = tools.grubbs_test(data_with_outlier, alpha=0.05)
print(f"Result: {result_c1}")
if result_c1.is_outlier:
print("-> SUCCESS: Outlier (150) found as expected.")
else:
print("-> ERROR: Outlier not found.")

# Now using a stricter alpha. Now G-calculated should be less than G-critical.
print("\nStep 2: alpha=0.01")
result_c2 = tools.grubbs_test(data_with_outlier, alpha=0.01)
print(f"Result: {result_c2}")
print(f" G-calculated: {result_c2.g_calculated:.4f}")
print(f" G-critical (greater): {result_c2.g_critical:.4f}")

if not result_c2.is_outlier:
print(f"-> SUCCESS: Outlier not found, because G-calculated < G-critical.")
else:
print("-> ERROR: Outlier found, but shouldn't have with such alpha.")
print("\n")

# --- Scenario G: Test on constant data ---
print("="*60)
print("SCENARIO D: Test on data with equal values")
print("Expected: is_outlier=False")
print("="*60)
result_d = tools.grubbs_test(data_constant)
print(f"Result of Pydantic model: {result_d}")
if not result_d.is_outlier:
print(f"-> SUCCESS: No outlier found, as expected.")
else:
print("-> ERROR: False outlier found.")
print("\n")

# --- Scenario D: Testing error handling ---
print("="*60)
print("SCENARIO D: Testing error handling")
print("="*60)

print("\nAttempt to pass an array with 2 elements (expected ValueError):")
try:
tools.grubbs_test(np.array([1, 2]))
except ValueError as e:
print(f"-> SUCCESS: Caught error: {e}")

print("\nTrying to pass a list instead of NumPy/Pandas (expecting TypeError):")
try:
# Your function accepts lists too, so this check may fail.
# But if there was strong typing, it would work.
# The current implementation of the code will not throw an error, since it converts the list to an array.
# To throw an error, pass an invalid type, such as a string.
tools.grubbs_test("invalid_type")
except TypeError as e:
print(f"-> SUCCESS: Caught error: {e}")