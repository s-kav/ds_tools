import numpy as np
import pytest

from src.ds_tool import DSTools

tools = DSTools()
np.random.seed(42)
N_SAMPLES = 200
x = np.linspace(-10, 10, N_SAMPLES)

@pytest.fixture
def data_linear():
    return 2 * x + 5

@pytest.fixture
def data_quadratic():
    return x**2 + np.random.normal(0, 0.1, N_SAMPLES)

@pytest.fixture
def data_random():
    return np.random.randn(N_SAMPLES) * 10

def test_chatterjee_linear(data_linear):
    xi = tools.chatterjee_correlation(x, data_linear)
    assert xi > 0.95

def test_chatterjee_quadratic(data_quadratic):
    xi = tools.chatterjee_correlation(x, data_quadratic)
    assert xi > 0.95

def test_chatterjee_random(data_random):
    xi = tools.chatterjee_correlation(x, data_random)
    assert xi < 0.1

def test_chatterjee_skewness(data_quadratic):
    xi_xy = tools.chatterjee_correlation(x, data_quadratic)
    xi_yx = tools.chatterjee_correlation(data_quadratic, x)
    assert abs(xi_xy - xi_yx) > 0.1

def test_chatterjee_standard_flag_differs(data_quadratic):
    xi_std = tools.chatterjee_correlation(x, data_quadratic, standard_flag=True)
    xi_orig = tools.chatterjee_correlation(x, data_quadratic, standard_flag=False)
    assert xi_std != xi_orig

def test_chatterjee_length_mismatch(data_linear):
    with pytest.raises(ValueError):
        tools.chatterjee_correlation(x[:-1], data_linear)
