
import numpy as np

# Import our class
from src.ds_tool import DSTools

# --- 1. Generate test data (probability distributions) ---

# "True" P distribution
P = np.array([0.1, 0.7, 0.2])

# Q1 distribution, good approximation to P
Q1_good_approx = np.array([0.15, 0.65, 0.2])

# Q2 distribution, bad approximation to P (uniform)
Q2_bad_approx = np.array([1/3, 1/3, 1/3])

# Unnormalized distributions
P_unnormalized = np.array([1, 7, 2]) # Equivalent to P
Q1_unnormalized = np.array([15, 65, 20]) # Equivalent to Q1

# --- 2. Initialization and calls ---
tools = DSTools()

# --- Scenario A: D_KL(P || P) = 0 ---
print("="*60)
print("SCENARIO A: Distribution diverges from itself")
print("Expected: D_KL(P || P) = 0")
print("="*60)
kl_self = tools.calculate_kl_divergence(P, P)
print(f"D_KL(P || P) = {kl_self:.6f}")
assert np.isclose(kl_self, 0)
print("-> SUCCESS: Result is close to zero.\n")

# --- Scenario B: Skewness ---
print("="*60)
print("SCENARIO B: Checking skewness D_KL(P || Q) != D_KL(Q || P)")
print("="*60)
kl_pq = tools.calculate_kl_divergence(P, Q1_good_approx)
kl_qp = tools.calculate_kl_divergence(Q1_good_approx, P)
print(f"D_KL(P || Q1) = {kl_pq:.4f}")
print(f"D_KL(Q1 || P) = {kl_qp:.4f}")
assert not np.isclose(kl_pq, kl_qp)
print("-> SUCCESS: Values ​​are not equal, skewness confirmed.\n")

# --- Scenario C: Comparing divergences ---
print("="*60)
print("SCENARIO B: Comparing Approximations")
print("Expected: D_KL(P || good_approx) < D_KL(P || bad_approx)")
print("="*60)
kl_good = tools.calculate_kl_divergence(P, Q1_good_approx)
kl_bad = tools.calculate_kl_divergence(P, Q2_bad_approx)
print(f"Divergence from good approximation (Q1): {kl_good:.4f}")
print(f"Divergence from bad approximation (Q2): {kl_bad:.4f}")
assert kl_good < kl_bad
print("-> SUCCESS: Divergence hierarchy is correct.\n")

# --- Scenario D: Calculating in bits ---
print("="*60)
print("SCENARIO D: Calculate in bits (base=2)")
print("="*60)
kl_nats = tools.calculate_kl_divergence(P, Q1_good_approx)
kl_bits = tools.calculate_kl_divergence(P, Q1_good_approx, base=2)
print(f"D_KL(P || Q1) in nats: {kl_nats:.4f}")
print(f"D_KL(P || Q1) in bits: {kl_bits:.4f}")
assert not np.isclose(kl_nats, kl_bits)
print("-> SUCCESS: Values ​​are different, 'base' flag works.\n")

# --- Scenario D: Working with non-normalized data ---
print("="*60)
print("SCENARIO D: Test on unnormalized data")
print("="*60)
kl_normalized = tools.calculate_kl_divergence(P, Q1_good_approx)
kl_unnormalized = tools.calculate_kl_divergence(P_unnormalized, Q1_unnormalized)
print(f"Divergence for normalized P and Q1: {kl_normalized:.4f}")
print(f"Divergence for unnormalized P and Q1: {kl_unnormalized:.4f}")
assert np.isclose(kl_normalized, kl_unnormalized)
print("-> SUCCESS: Values ​​match, normalization works.\n")

# --- Scenario E: Testing error handling ---
print("="*60)
print("SCENARIO E: Testing Error Handling")
print("="*60)
print("\nAttempt to pass distributions of different lengths:")
try:
    tools.calculate_kl_divergence([0.5, 0.5], [0.3, 0.3, 0.4])
except ValueError as e:
    print(f"-> SUCCESS: Caught error: {e}")

print("\nAttempt to pass distributions with negative values:")
try:
    tools.calculate_kl_divergence([1.5, -0.5], [0.5, 0.5])
except ValueError as e:
    print(f"-> SUCCESS: Caught error: {e}")