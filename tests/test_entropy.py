
import numpy as np

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data (probability distributions) ---

# Highly certain distribution (one outcome dominates)
dist_ordered = np.array([0.9, 0.05, 0.05])

# Uniform distribution (maximum uncertainty)
dist_uniform = np.array([1/3, 1/3, 1/3])

# Intermediate distribution
dist_mixed = np.array([0.5, 0.3, 0.2])

# Deterministic distribution (complete certainty)
dist_deterministic = np.array([1.0, 0.0, 0.0])

# Unnormalized data (sum is not 1)
dist_unnormalized = np.array([1, 4, 5]) # Equivalent to [0.1, 0.4, 0.5]

# --- 2. Initialization and calls ---
tools = DSTools()

# --- Scenario A: Comparing entropy of different distributions ---
print("="*60)
print("SCENARIO A: Comparing entropy (in nats)")
print("Expect: E(deterministic) < E(ordered) < E(mixed) < E(uniform)")
print("="*60)

entropy_det = tools.calculate_entropy(dist_deterministic)
entropy_ord = tools.calculate_entropy(dist_ordered)
entropy_mix = tools.calculate_entropy(dist_mixed)
entropy_uni = tools.calculate_entropy(dist_uniform)

print(f"Entropy of deterministic [1, 0, 0]: {entropy_det:.4f} (should be 0.0)")
print(f"Entropy of ordered [0.9, 0.05, 0.05]: {entropy_ord:.4f}")
print(f"Entropy of mixed [0.5, 0.3, 0.2]: {entropy_mix:.4f}")
print(f"Entropy of uniform [0.33, 0.33, 0.33]: {entropy_uni:.4f} (should be maximal)")

# Assertions
assert np.isclose(entropy_det, 0.0)
assert entropy_det < entropy_ord < entropy_mix < entropy_uni
print("-> SUCCESS: Entropy hierarchy is correct.\n")

# --- Scenario B: Calculation in bits ---
print("="*60)
print("SCENARIO B: Calculate entropy in bits (base=2)")
print("="*60)

entropy_mix_nats = tools.calculate_entropy(dist_mixed) # base=None
entropy_mix_bits = tools.calculate_entropy(dist_mixed, base=2)

print(f"Entropy for [0.5, 0.3, 0.2] in nats: {entropy_mix_nats:.4f}")
print(f"Entropy for [0.5, 0.3, 0.2] in bits: {entropy_mix_bits:.4f}")
print("-> Values ​​are different. Flag 'base' works. SUCCESS.\n")
assert entropy_mix_nats != entropy_mix_bits

# --- Scenario B: Working with unnormalized data ---
print("="*60)
print("SCENARIO B: Test on unnormalized data [1, 4, 5]")
print("Expected: the result should be the same as for [0.1, 0.4, 0.5]")
print("="*60)

normalized_equivalent = np.array([0.1, 0.4, 0.5])
entropy_normalized = tools.calculate_entropy(normalized_equivalent)
entropy_unnormalized = tools.calculate_entropy(dist_unnormalized)

print(f"Entropy for [0.1, 0.4, 0.5]: {entropy_normalized:.4f}")
print(f"Entropy for [1, 4, 5]: {entropy_unnormalized:.4f}")
assert np.isclose(entropy_normalized, entropy_unnormalized)
print("-> Values ​​match. Automatic normalization works. SUCCESS.\n")

# --- Scenario D: Testing error handling ---
print("="*60)
print("SCENARIO D: Testing error handling (negative probabilities)")
print("Expecting: ValueError")
print("="*60)
try:
invalid_dist = np.array([1.5, -0.5, 0.0])
tools.calculate_entropy(invalid_dist)
except ValueError as e:
print(f"SUCCESSFULLY caught expected error: {e}")