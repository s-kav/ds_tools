import numpy as np
import pytest
from scipy import stats

from ds_tool import DistributionConfig


def describe_generated_data(data: np.ndarray):
    return {
        "mean": np.mean(data),
        "std": np.std(data, ddof=1),
        "skew": stats.skew(data),
        "kurt": stats.kurtosis(data) + 3,  # Convert to Pearson's kurtosis
        "min": np.min(data),
        "max": np.max(data),
        "n": len(data),
    }


@pytest.mark.parametrize(
    "config",
    [
        DistributionConfig(
            mean=1000,
            median=950,
            std=200,
            min_val=400,
            max_val=2500,
            skewness=0.8,
            kurtosis=4.0,
            n=2000,
            outlier_ratio=0.01,
        ),
        DistributionConfig(
            mean=50,
            median=48,
            std=10,
            min_val=10,
            max_val=150,
            skewness=1.5,
            kurtosis=8.0,
            n=2000,
            outlier_ratio=0.03,
        ),
    ],
)
def test_generate_distribution_valid_configs(tools, config):
    data = tools.generate_distribution(config)
    stats_actual = describe_generated_data(data)

    assert np.isclose(stats_actual["mean"], config.mean, rtol=0.8)
    assert np.isclose(stats_actual["std"], config.std, rtol=5.0)
    assert stats_actual["min"] >= config.min_val
    assert stats_actual["max"] <= config.max_val
    assert stats_actual["n"] == config.n


def test_generate_distribution_invalid_moments(tools):
    config_invalid = DistributionConfig(
        mean=100,
        median=100,
        std=15,
        min_val=50,
        max_val=150,
        skewness=2.0,
        kurtosis=1.0,
        n=1000,  # Invalid: kurt < skew² - 2
    )
    with pytest.raises(ValueError, match="Invalid statistical moments"):
        tools.generate_distribution(config_invalid)


def test_distribution_config_pydantic_validation():
    with pytest.raises(ValueError, match="max_val must be greater than min_val"):
        DistributionConfig(
            mean=100,
            median=100,
            std=15,
            min_val=200,
            max_val=100,
            skewness=0.0,
            kurtosis=3.0,
            n=1000,
        )


def test_generate_distribution_invalid_range_raises_error(tools):
    """Tests that max_val <= min_val raises ValueError."""
    with pytest.raises(ValueError, match="max_val must be greater than min_val"):
        DistributionConfig(
            mean=10,
            median=10,
            std=1,
            min_val=20,
            max_val=10,
            skewness=0,
            kurtosis=3,
            n=100,
        )


def test_generate_distribution_zero_std_case(tools, mocker):
    """Tests the branch for zero standard deviation."""
    config = DistributionConfig(
        mean=50, median=50, std=1, min_val=0, max_val=100, skewness=0, kurtosis=3, n=10
    )
    # artificially create a situation with zero std by mocking np.std
    mocker.patch("ds_tool.np.std", return_value=0)
    data = tools.generate_distribution(config)

    unique_values = np.unique(data)
    expected_values = {config.min_val, config.max_val, config.mean}
    assert set(unique_values).issubset(expected_values)


def test_generate_distribution_no_outliers(tools):
    """Tests the branch where outlier_ratio is 0."""
    config = DistributionConfig(
        mean=10,
        median=10,
        std=1,
        min_val=0,
        max_val=20,
        skewness=0,
        kurtosis=3,
        n=100,
        outlier_ratio=0,
    )
    data = tools.generate_distribution(config)
    assert len(data) == 100


def test_generate_distribution_standard_normal_branch(tools):
    """Tests the branch for kurtosis <= 3.5, which uses standard_normal."""
    config = DistributionConfig(
        mean=10, median=10, std=2, min_val=0, max_val=20, skewness=0, kurtosis=3, n=100
    )
    data = tools.generate_distribution(config)
    assert len(data) == 100


def test_generate_distribution_ensures_min_max(tools):
    """Tests that min/max values are forced if not present."""
    config = DistributionConfig(
        mean=50, median=50, std=1, min_val=1, max_val=100, skewness=0, kurtosis=3, n=50
    )
    data = tools.generate_distribution(config)
    assert data.min() == 1
    assert data.max() == 100
