import numpy as np
import pytest
from src.ds_tool import DSTools

tools = DSTools()

@pytest.fixture
def data_normal():
    np.random.seed(42)
    return np.random.normal(loc=100, scale=10, size=30)

@pytest.fixture
def data_with_outlier(data_normal):
    return np.append(data_normal, 150)

@pytest.fixture
def data_constant():
    return np.full(10, 50.0)

def test_outlier_detected(data_with_outlier):
    result = tools.grubbs_test(data_with_outlier)
    assert result.is_outlier
    assert result.outlier_value == 150
    assert result.outlier_index == len(data_with_outlier) - 1

def test_no_outlier_detected(data_normal):
    result = tools.grubbs_test(data_normal)
    assert not result.is_outlier

@pytest.mark.parametrize("alpha, expected", [
    (0.05, True),
    (0.01, True),
])
def test_outlier_sensitivity_to_alpha(data_with_outlier, alpha, expected):
    result = tools.grubbs_test(data_with_outlier, alpha=alpha)
    assert result.is_outlier == expected
    if expected:
        assert result.g_calculated > result.g_critical
    else:
        assert result.g_calculated <= result.g_critical

def test_constant_data(data_constant):
    result = tools.grubbs_test(data_constant)
    assert not result.is_outlier

def test_too_few_values_raises_value_error():
    with pytest.raises(ValueError, match="Grubbs test requires at least 3 data points."):
        tools.grubbs_test(np.array([1, 2]))

def test_invalid_input_type_raises_type_error():
    with pytest.raises(TypeError, match="Input data x must be a NumPy array or Pandas Series."):
        tools.grubbs_test("invalid_type")
