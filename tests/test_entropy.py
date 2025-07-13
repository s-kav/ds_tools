import numpy as np
import pytest
from src.ds_tool import DSTools

tools = DSTools()

def test_entropy_hierarchy():
    # discrete distribution
    dist_deterministic = np.array([1.0, 0.0, 0.0])
    dist_ordered = np.array([0.9, 0.05, 0.05])
    dist_mixed = np.array([0.5, 0.3, 0.2])
    dist_uniform = np.array([1/3, 1/3, 1/3])

    # calculations
    e_det = tools.calculate_entropy(dist_deterministic)
    e_ord = tools.calculate_entropy(dist_ordered)
    e_mix = tools.calculate_entropy(dist_mixed)
    e_uni = tools.calculate_entropy(dist_uniform)

    # checks
    assert np.isclose(e_det, 0.0), "Deterministic entropy must be 0"
    assert e_det < e_ord < e_mix < e_uni, "The order of entropies is violated"
    
def test_entropy_bits_vs_nats():
    dist = np.array([0.5, 0.3, 0.2])
    e_nats = tools.calculate_entropy(dist)
    e_bits = tools.calculate_entropy(dist, base=2)
    
    assert not np.isclose(e_nats, e_bits), "Entropies in nats and bits must be different"

def test_entropy_normalization_equivalence():
    normalized = np.array([0.1, 0.4, 0.5])
    unnormalized = np.array([1, 4, 5])
    
    e1 = tools.calculate_entropy(normalized)
    e2 = tools.calculate_entropy(unnormalized)
    
    assert np.isclose(e1, e2), "The entropy of normalized and unnormalized must be the same"

def test_entropy_invalid_distribution():
    dist_invalid = np.array([1.5, -0.5, 0.0])
    with pytest.raises(ValueError, match="Probabilities cannot be negative"):
        tools.calculate_entropy(dist_invalid)
