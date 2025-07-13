import numpy as np
import pytest
from src.ds_tool import DSTools

tools = DSTools()

# Distributions for reuse
P = np.array([0.1, 0.7, 0.2])
Q1 = np.array([0.15, 0.65, 0.2])
Q2 = np.array([1/3, 1/3, 1/3])
P_unnormalized = np.array([1, 7, 2])
Q1_unnormalized = np.array([15, 65, 20])

def test_kl_self_divergence_zero():
    kl = tools.calculate_kl_divergence(P, P)
    assert np.isclose(kl, 0.0), "KL(P || P) should be 0"

def test_kl_skewness():
    kl_pq = tools.calculate_kl_divergence(P, Q1)
    kl_qp = tools.calculate_kl_divergence(Q1, P)
    assert not np.isclose(kl_pq, kl_qp), "KL divergence is not symmetric"

def test_kl_divergence_quality():
    kl_good = tools.calculate_kl_divergence(P, Q1)
    kl_bad = tools.calculate_kl_divergence(P, Q2)
    assert kl_good < kl_bad, "KL divergence from good approx should be lower than from bad approx"

def test_kl_base_flag_effect():
    kl_nats = tools.calculate_kl_divergence(P, Q1)
    kl_bits = tools.calculate_kl_divergence(P, Q1, base=2)
    assert not np.isclose(kl_nats, kl_bits), "Base flag must affect result"

def test_kl_normalization():
    kl_norm = tools.calculate_kl_divergence(P, Q1)
    kl_unnorm = tools.calculate_kl_divergence(P_unnormalized, Q1_unnormalized)
    assert np.isclose(kl_norm, kl_unnorm), "Unnormalized and normalized distributions must yield same KL"

def test_kl_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="Input distributions P and Q must have the same shape."):
        tools.calculate_kl_divergence([0.5, 0.5], [0.3, 0.3, 0.4])

def test_kl_raises_on_negative_values():
    with pytest.raises(ValueError, match="Probabilities cannot be negative."):
        tools.calculate_kl_divergence([1.5, -0.5], [0.5, 0.5])
