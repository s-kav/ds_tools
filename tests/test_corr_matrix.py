import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import our class and its configuration
from src.ds_tool import DSTools, CorrelationConfig

# --- 1. Generate test data ---

# Set seed for reproducibility of results
np.random.seed(42)
N_SAMPLES = 100

print("--- Generate DataFrame with different types of dependencies ---")

# Create base feature
feature_A = np.linspace(-10, 10, N_SAMPLES)

# Create other features based on it:
# feature_B: Strong positive linear correlation with A
feature_B = feature_A * 2 + np.random.normal(0, 2, N_SAMPLES)

# feature_C: Strong negative linear correlation with A
feature_C = -feature_A * 1.5 + np.random.normal(0, 3, N_SAMPLES)

# feature_D: Nonlinear (quadratic) dependence on A.
# Pearson should show low correlation, and Spearman - high.
feature_D = feature_A**2 + np.random.normal(0, 5, N_SAMPLES)

# feature_E: No correlation (random noise)
feature_E = np.random.rand(N_SAMPLES) * 20

# feature_F: Moderate positive correlation with B
feature_F = feature_B * 0.5 + np.random.normal(0, 10, N_SAMPLES)

# Put it all in a DataFrame
df = pd.DataFrame({
'feature_A': feature_A,
'feature_B': feature_B,
'feature_C': feature_C,
'feature_D': feature_D,
'feature_E': feature_E,
'feature_F': feature_F
})

print("First 5 rows generated data:")
print(df.head())
print("-" * 50, "\n")

# --- 2. Call corr_matrix ---

# Initialize our toolkit
tools = DSTools()

# --- Scenario A: Call with default configuration ---
print("--- Scenario A: Correlation matrix (default Pearson method) ---")
print("Expect to see a graph. Note that the correlation of A and D (non-linear relationship) will be low.")
# plt.show() is blocking, so we'll call it at the end.
# But in real use the call would be:
tools.corr_matrix(df)

# --- Scenario B: Call with custom config ---
print("\n--- Scenario B: Correlation matrix (Spearman method, custom view) ---")
print("Expect to see the second plot with a different correlation method.")
print("Now the correlation of A and D (non-linear relationship) should be high, since Spearman is based on ranks.")

custom_config = CorrelationConfig(
build_method='spearman',
font_size=10,
image_size=(12, 12)
)
tools.corr_matrix(df, config=custom_config)

# --- Scenario C: Checking Pydantic validation ---
print("\n--- Scenario C: Testing validation ---")
print("Attempt to create a configuration with an invalid method. Expecting to see a ValueError.")

try:
    invalid_config = CorrelationConfig(build_method='invalid_method')
except ValueError as e:
    print(f"\nSuccessfully caught the expected error: {e}")

print("-" * 50)