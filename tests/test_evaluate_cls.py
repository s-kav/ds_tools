
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data ---
# This data is identical to what we generated for compute_metrics,
# as it is ideal for a binary classification problem.

# Set seed for reproducibility of results
np.random.seed(42)
N_SAMPLES = 500

# Create array of true labels (y_true)
y_true = np.random.randint(0, 2, size=N_SAMPLES)

# Create array of predicted probabilities (y_predict_proba)
# Bias probabilities towards true labels by adding noise
base_probs = np.where(y_true == 1, 0.75, 0.25)
noise = np.random.normal(0, 0.2, size=N_SAMPLES)
y_predict_proba = np.clip(base_probs + noise, 0.01, 0.99) # Clip to avoid 0 and 1

print("--- Data generation complete ---")
print(f"Form y_true: {y_true.shape}")
print(f"Form y_predict_proba: {y_predict_proba.shape}")
print("-" * 35, "\n")

# --- 2. Initialization and calls ---
tools = DSTools()

# --- Scenario A: Main call with threshold 0.5 ---
print("="*60)
print("SCENARIO A: Call with default threshold (0.5)")
print("Waiting for full report in console and window with two graphs.")
print("="*60)

# IMPORTANT: The script will stop and wait for the window with the graph to close.
returned_metrics_a = tools.evaluate_classification(
true_labels=y_true,
pred_probs=y_predict_proba,
threshold=0.5
)

print("\n--- Checking the returned dictionary (Scenario A): ---")
print(f"The type of the returned object: {type(returned_metrics_a)}")
print("Keys in the dictionary:", list(returned_metrics_a.keys()))
print(f"ROC AUC from the dictionary: {returned_metrics_a['roc_auc']:.4f}")
print("\n")

# --- Scenario B: Call with a different threshold (0.7) ---
print("="*60)
print("SCENARIO B: Call with a higher threshold (0.7)")
print("We expect that 'Accuracy' and the report will change, but ROC AUC will not.")
print("="*60)

returned_metrics_b = tools.evaluate_classification(
true_labels=y_true,
pred_probs=y_predict_proba,
threshold=0.7 # Use a higher threshold
)

print("\n--- Comparison of results (Scenario A vs Scenario B): ---")
print(f"Accuracy (threshold=0.5): {returned_metrics_a['accuracy']:.4f}")
print(f"Accuracy (threshold=0.7): {returned_metrics_b['accuracy']:.4f}")
print("-> Accuracy values ​​should be different.\n")

print(f"ROC AUC (threshold=0.5): {returned_metrics_a['roc_auc']:.4f}")
print(f"ROC AUC (threshold=0.7): {returned_metrics_b['roc_auc']:.4f}")
print("-> ROC AUC values ​​should be identical, since it does not depend on the threshold.")
print("\n")

# --- Scenario B: Testing error handling ---
print("="*60)
print("SCENARIO B: Testing error handling (shape mismatch)")
print("Expecting ValueError.")
print("="*60)

try:
y_true_short = y_true[:-10] # Shorten one of the arrays
tools.evaluate_classification(y_true_short, y_predict_proba)
except ValueError as e:
print(f"SUCCESSFULLY caught expected error: {e}")