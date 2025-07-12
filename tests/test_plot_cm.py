
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data ---

# Set seed for reproducibility
np.random.seed(42)
N_SAMPLES = 200

# --- Data for binary classification (classes 0 and 1) ---
y_true_binary = np.random.randint(0, 2, size=N_SAMPLES)
# Simulate predictions of a model that sometimes gets it wrong
# With 85% probability the prediction will be correct
y_pred_binary = np.where(np.random.rand(N_SAMPLES) < 0.85, y_true_binary, 1 - y_true_binary)

# --- Data for multi-class classification (classes 0, 1, 2) ---
y_true_multi = np.random.randint(0, 3, size=N_SAMPLES)
# Simulate predictions. With a probability of 75%, the class is predicted correctly, otherwise - a random one of the other two.
correct_preds = np.random.rand(N_SAMPLES) < 0.75
random_errors = np.random.randint(1, 3, size=N_SAMPLES)
y_pred_multi = np.where(correct_preds, y_true_multi, (y_true_multi + random_errors) % 3)

# --- 2. Initialization and calls ---
tools = DSTools()

# --- Scenario A: Binary matrix without custom labels ---
print("="*60)
print("SCENARIO A: Binary adjacency matrix (default labels)")
print("Expected: 2x2 plot with '0' and '1' labels on axes.")
print("="*60)
# IMPORTANT: The script will wait for the window to close with plot
tools.plot_confusion_matrix(y_true_binary, y_pred_binary)

# --- Scenario B: Binary matrix with custom labels and theme ---
print("\n" + "="*60)
print("SCENARIO B: Binary matrix with custom labels and different color scheme")
print("Expected: 2x2 plot with labels 'Negative' and 'Positive'.")
print("="*60)
tools.plot_confusion_matrix(
y_true_binary,
y_pred_binary,
class_labels=['Negative (0)', 'Positive (1)'],
title='Binary Classification Performance',
cmap='Greens'
)

# --- Scenario C: Multi-class matrix with custom labels ---
print("\n" + "="*60)
print("SCENARIO B: Multiclass Adjacency Matrix (3x3)")
print("Expecting: 3x3 plot with animal class labels.")
print("="*60)
tools.plot_confusion_matrix(
y_true_multi,
y_pred_multi,
class_labels=['Cat', 'Dog', 'Bird'],
title='Multi-Class Classification (Animals)',
cmap='YlGnBu'
)

# --- Scenario D: Testing Error Handling ---
print("\n" + "="*60)
print("SCENARIO D: Testing Error Handling (Invalid Number of Labels)")
print("Expecting: ValueError.")
print("="*60)
try:
tools.plot_confusion_matrix(
y_true_binary,
y_pred_binary,
class_labels=['One label'] # Pass 1 label for 2 classes
)
except ValueError as e:
print(f"SUCCESSFULLY caught expected error: {e}")