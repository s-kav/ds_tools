# tests/test_models.py
"""
Tests for all Pydantic models in ds_tool.models.
This file validates default values, constraints, and custom validation logic.
 *
 * Copyright (c) [2025] [Sergii Kavun]
 *
 * This software is dual-licensed:
 * - PolyForm Noncommercial 1.0.0 (default)
 * - Commercial license available
 *
 * See LICENSE for details
 *
"""
import pytest

from models import (
    CorrelationConfig,
    DistributionConfig,
    GrubbsTestResult,
    MetricsConfig,
    OutlierConfig,
)

# --- Tests for MetricsConfig ---


def test_metrics_config_defaults():
    """Tests the default values of MetricsConfig."""
    config = MetricsConfig()
    assert config.error_vis is True
    assert config.print_values is False


def test_metrics_config_custom_values():
    """Tests setting custom values in MetricsConfig."""
    config = MetricsConfig(error_vis=False, print_values=True)
    assert config.error_vis is False
    assert config.print_values is True


# --- Tests for CorrelationConfig ---


def test_correlation_config_defaults():
    """Tests the default values of CorrelationConfig."""
    config = CorrelationConfig()
    assert config.build_method == "pearson"
    assert config.font_size == 14
    assert config.image_size == (16, 16)


@pytest.mark.parametrize("method", ["pearson", "kendall", "spearman"])
def test_correlation_config_valid_methods(method):
    """Tests that all valid correlation methods are accepted."""
    config = CorrelationConfig(build_method=method)
    assert config.build_method == method


def test_correlation_config_invalid_method_raises_error():
    """Tests that an invalid correlation method raises a ValueError."""
    with pytest.raises(ValueError, match="Method must be one of"):
        CorrelationConfig(build_method="invalid_method")


@pytest.mark.parametrize("size", [8, 20])
def test_correlation_config_font_size_boundaries(size):
    """Tests the inclusive boundaries for font_size."""
    config = CorrelationConfig(font_size=size)
    assert config.font_size == size


@pytest.mark.parametrize("size", [7, 21])
def test_correlation_config_font_size_out_of_bounds_raises_error(size):
    """Tests that font_size outside of [8, 20] raises a ValueError."""
    with pytest.raises(ValueError):
        CorrelationConfig(font_size=size)


# --- Tests for OutlierConfig ---


def test_outlier_config_defaults():
    """Tests the default values of OutlierConfig."""
    config = OutlierConfig()
    assert config.sigma == 1.5
    assert config.change_remove is True
    assert config.percentage is True


# --- Tests for DistributionConfig ---


def test_distribution_config_valid_creation():
    """Tests successful creation of a valid DistributionConfig."""
    config = DistributionConfig(
        mean=10, median=9, std=2, min_val=1, max_val=20, skewness=0.1, kurtosis=3, n=100
    )
    assert config.max_val > config.min_val


def test_distribution_config_max_less_than_min_raises_error():
    """Tests that max_val <= min_val raises a ValueError."""
    with pytest.raises(ValueError, match="max_val must be greater than min_val"):
        DistributionConfig(
            mean=10,
            median=9,
            std=2,
            min_val=20,
            max_val=10,
            skewness=0.1,
            kurtosis=3,
            n=100,
        )


def test_distribution_config_max_equal_to_min_raises_error():
    """Tests that max_val <= min_val raises a ValueError."""
    with pytest.raises(ValueError, match="max_val must be greater than min_val"):
        DistributionConfig(
            mean=10,
            median=9,
            std=2,
            min_val=10,
            max_val=10,
            skewness=0.1,
            kurtosis=3,
            n=100,
        )


# --- Tests for GrubbsTestResult ---


def test_grubbs_test_result_instantiation():
    """Tests the successful creation of a GrubbsTestResult instance."""
    result = GrubbsTestResult(
        is_outlier=True,
        g_calculated=3.5,
        g_critical=3.1,
        outlier_value=100.0,
        outlier_index=10,
    )
    assert result.is_outlier is True
    assert result.outlier_value == 100.0
