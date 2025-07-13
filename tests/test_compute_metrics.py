import numpy as np
import pandas as pd

# Import our class and its configuration
from src.ds_tool import DSTools, MetricsConfig

# --- 1. Generate test data ---

# Set a seed for reproducibility
np.random.seed(42)

# Determine the number of examples in our test set
N_SAMPLES = 200

# Create an array of true labels (y_true)
# Roughly equal number of 0 and 1
y_true = np.random.randint(0, 2, size=N_SAMPLES)

# Create an array of predicted probabilities (y_predict_proba)
# We want the model to be "good but not perfect".
# To do this, we generate probabilities so that they generally match the
# true labels, but with some random "noise".
# Where y_true = 1, the probabilities will be shifted to 1.0.
# Where y_true = 0, the probabilities will be shifted to 0.0.

# Create "base" probabilities: 0.8 for class 1 and 0.2 for class 0
base_probs = np.where(y_true == 1, 0.8, 0.2)
# Add normal noise to make predictions more realistic
noise = np.random.normal(0, 0.25, size=N_SAMPLES)
y_predict_proba = base_probs + noise

# Limit probabilities to the range [0, 1], since noise could take them outside the range
y_predict_proba = np.clip(y_predict_proba, 0, 1)

# Create an array of predicted labels (y_predict) based on a threshold of 0.5
THRESHOLD = 0.5
y_predict = (y_predict_proba >= THRESHOLD).astype(int)

print("--- Data generation complete ---")
print(f"Number of samples: {N_SAMPLES}")
print(f"Example of true labels (y_true): {y_true[:10]}")
print(f"Example of probabilities (y_predict_proba): {y_predict_proba[:10].round(2)}")
print(f"Example of predicted labels (y_predict): {y_predict[:10]}")
print("-" * 35, "\n")

# --- 2. Calling compute_metrics ---

# Initializing our toolkit
tools = DSTools()

# --- Case A: Calling with default configuration ---
# In this case, config.error_vis = True, so we should see a plot
print("--- Running compute_metrics (default configuration) ---")
print("Expect to see a plot of 'Error Rates vs Threshold Levels'...")

metrics_df_default = tools.compute_metrics(y_true, y_predict, y_predict_proba)

print("\nResult as a DataFrame:")
print(metrics_df_default)
print("-" * 55, "\n")

# --- Option B: Call with custom configuration ---
# Disable visualization and enable printing of metric values ​​directly to the console
print("--- Running compute_metrics (custom configuration) ---")
print("We expect to see metric values ​​printed in the console...")

custom_config = MetricsConfig(
error_vis=False, # Disable graph
print_values=True # Enable printing
)

metrics_df_custom = tools.compute_metrics(y_true, y_predict, y_predict_proba, config=custom_config)

print("\nResult as DataFrame:")
print(metrics_df_custom)
print("-" * 55)