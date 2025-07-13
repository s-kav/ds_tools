
import numpy as np
import re

# Import our class
from src.ds_tool import DSTools

# --- Initialize the toolkit ---
tools = DSTools()

# --- Scenario A: Main call ---
print("="*60)
print("SCENARIO A: Generate 5 codes of length 12")
print("="*60)

N_CODES = 5
CODE_LENGTH = 12
codes_a = tools.generate_alphanum_codes(n=N_CODES, length=CODE_LENGTH)

print("Generated codes:")
print(codes_a)

# Checks
print("\nChecks:")
# 1. Type and shape
assert isinstance(codes_a, np.ndarray)
assert codes_a.shape == (N_CODES,)
print("-> SUCCESS: The output type and format are correct.")

# 2. The length of each code
all_lengths_correct = all(len(code) == CODE_LENGTH for code in codes_a)
assert all_lengths_correct
print(f"-> SUCCESS: The length of all codes is {CODE_LENGTH}.")

# 3. The composition of characters
# Create a regular expression that matches a string consisting only of allowed characters
allowed_chars_pattern = re.compile(f"^[0-9A-Z]{{{CODE_LENGTH}}}$")
all_chars_correct = all(allowed_chars_pattern.match(code) for code in codes_a)
assert all_chars_correct
print("-> SUCCESS: All characters in the codes belong to the alphabet '0-9' and 'A-Z'.\n")

# --- Scenario B: Check for uniqueness ---
print("="*60)
print("SCENARIO B: Generate a large number of codes to check for uniqueness")
print("="*60)

LARGE_N = 10000
codes_b = tools.generate_alphanum_codes(n=LARGE_N, length=10)

# Compare the number of generated codes with the number of unique codes
num_unique = len(np.unique(codes_b))
assert num_unique == LARGE_N
print(f"{LARGE_N} codes generated, of which {num_unique} are unique.")
print("-> SUCCESS: All generated codes are unique (no collisions found).\n")

# --- Scenario C: Edge cases ---
print("="*60)
print("SCENARIO B: Testing edge cases (n=0, length=0)")
print("="*60)

# n = 0
print("Request for 0 codes (n=0):")
codes_n0 = tools.generate_alphanum_codes(n=0, length=8)
print(f"Result: {codes_n0}, Shape: {codes_n0.shape}")
assert codes_n0.shape == (0,)
print("-> SUCCESS: Empty array of correct shape returned.\n")

# length = 0
print("Request for zero length codes (length=0):")
codes_l0 = tools.generate_alphanum_codes(n=3, length=0)
print(f"Result: {codes_l0}")
assert all(code == '' for code in codes_l0)
print("-> SUCCESS: An array of empty strings was returned.\n")

# --- Scenario D: Testing error handling ---
print("="*60)
print("SCENARIO D: Testing error handling (negative values)")
print("="*60)
print("Attempt to pass n = -1 (expecting ValueError):")
try:
    tools.generate_alphanum_codes(n=-1)
except ValueError as e:
    print(f"-> SUCCESS: Expected error caught: {e}\n")
    print("Attempt to pass length = -5 (expecting ValueError):")
try:
    tools.generate_alphanum_codes(n=10, length=-5)
except ValueError as e:
    print(f"-> SUCCESS: Expected error caught: {e}")