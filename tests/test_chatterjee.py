
import numpy as np
import pandas as pd

# Import our class
from src.ds_tool import DSTools

# --- 1. Generate test data ---

# Set seed for reproducibility
np.random.seed(42)
N_SAMPLES = 200

# Create base independent variable
x = np.linspace(-10, 10, N_SAMPLES)

# Scenario 1: Perfect linear dependence
y_linear = 2 * x + 5

# Scenario 2: Perfect nonlinear (quadratic) dependence
y_quadratic = x**2 + np.random.normal(0, 0.1, N_SAMPLES) # Add some noise

# Scenario 3: Complete independence (random noise)
y_random = np.random.randn(N_SAMPLES) * 10

# --- 2. Initialization and calls ---
tools = DSTools()

# --- Scenario A: Linear dependence ---
print("="*60)
print("SCENARIO A: Linear dependence (y = 2x + 5)")
print("Expected: Xi close to 1.0")
print("="*60)
xi_linear = tools.chatterjee_correlation(x, y_linear)
print(f"Result: {xi_linear:.4f}\n")
assert xi_linear > 0.95

# --- Scenario B: Nonlinear dependence ---
print("="*60)
print("SCENARIO B: Nonlinear quadratic dependence (y = x^2)")
print("Expected: Xi close to 1.0 (Pearson would show ~0 here)")
print("="*60)
xi_quadratic = tools.chatterjee_correlation(x, y_quadratic)
print(f"Result: {xi_quadratic:.4f}\n")
assert xi_quadratic > 0.95

# --- Scenario B: No correlation ---
print("="*60)
print("SCENARIO B: No correlation (random noise)")
print("Expected: Xi close to 0.0")
print("="*60)
xi_random = tools.chatterjee_correlation(x, y_random)
print(f"Result: {xi_random:.4f}\n")
assert xi_random < 0.1

# --- Scenario D: Check for skewness ---
print("="*60)
print("SCENARIO D: Check for skewness Xi(x, y) != Xi(y, x)")
print("="*60)
xi_xy = tools.chatterjee_correlation(x, y_quadratic)
xi_yx = tools.chatterjee_correlation(y_quadratic, x)
print(f"Xi(x, y) = {xi_xy:.4f} (y is a function of x)")
print(f"Xi(y, x) = {xi_yx:.4f} (x is not a single-valued function of y)")
print("-> Values ​​must be different. SUCCESS.\n")
assert abs(xi_xy - xi_yx) > 0.1

# --- Scenario D: Check for standard_flag ---
print("="*60)
print("SCENARIO D: Checking the standard_flag flag")
print("Expected: results with standard_flag=True and False should be different.")
print("="*60)
xi_standard = tools.chatterjee_correlation(x, y_quadratic, standard_flag=True)
xi_original = tools.chatterjee_correlation(x, y_quadratic, standard_flag=False)
print(f"Result with standard formula: {xi_standard:.4f}")
print(f"Result with original formula: {xi_original:.4f}")
print("-> Values ​​are different. Flag works. SUCCESS.\n")
assert xi_standard != xi_original

# --- Scenario E: Testing error handling ---
print("="*60)
print("SCENARIO E: Testing Error Handling (Arrays of Different Lengths)")
print("Expecting: ValueError")
print("="*60)

try:
    tools.chatterjee_correlation(x[:-1], y_linear)
except ValueError as e:
    print(f"SUCCESSFULLY caught expected error: {e}")