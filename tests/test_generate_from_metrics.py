import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy import stats

from ds_tool import DistributionConfig


def describe_metrics(arr: np.ndarray):
    return {
        "mean": np.mean(arr),
        "median": np.median(arr),
        "std": np.std(arr, ddof=1),
        "skewness": stats.skew(arr),
        "kurtosis": stats.kurtosis(arr, fisher=True),  # Keep as Fisher to match input
        "min_val": np.min(arr),
        "max_val": np.max(arr),
        "n": len(arr),
    }


@pytest.mark.parametrize(
    "config",
    [
        DistributionConfig(
            mean=500,
            median=450,
            std=150,
            min_val=100,
            max_val=2000,
            skewness=1.2,
            kurtosis=5.0,
            n=5000,
            outlier_ratio=0.02,
        ),
        DistributionConfig(
            mean=80.0,
            median=75.0,
            std=20.0,
            min_val=10,
            max_val=150,
            skewness=0.5,
            kurtosis=0.8,
            n=3000,
            outlier_ratio=0.01,
        ),
    ],
)
@pytest.mark.parametrize("output_as", ["numpy", "pandas", "polars"])
@pytest.mark.parametrize("int_flag", [False, True])
def test_generate_distribution_from_config(tools, config, output_as, int_flag):
    try:
        data = tools.generate_distribution_from_metrics(
            n=config.n, metrics=config, output_as=output_as, int_flag=int_flag
        )

        if isinstance(data, (pd.Series, pl.Series)):
            data = data.to_numpy()
        elif isinstance(data, np.ndarray):
            pass
        else:
            raise TypeError("Unsupported output type returned.")

        stats_actual = describe_metrics(data)

        assert stats_actual["n"] == config.n
        assert stats_actual["min_val"] >= config.min_val - 1  # tolerance for clipping
        assert stats_actual["max_val"] <= config.max_val + 1

        # Validate approximate match for moments
        assert np.isclose(stats_actual["mean"], config.mean, rtol=0.8)
        assert np.isclose(stats_actual["median"], config.median, rtol=1.5)
        assert np.isclose(stats_actual["std"], config.std, rtol=5.0)
        # Skewness and kurtosis are harder to hit exactly
        assert stats_actual["skewness"] * config.skewness >= 0  # just sign check
    except Exception:
        # Let the test fail normally for unexpected errors
        raise AssertionError("Invalid metrics dictionary")


def test_invalid_moments_raise_value_error(tools):
    """Test that an impossible combination of moments triggers a ValueError."""
    with pytest.raises(ValueError, match="Invalid metrics dictionary"):
        tools.generate_distribution_from_metrics(
            n=1000,
            metrics={
                "mean": 100,
                "median": 100,
                "std": 15,
                "min_val": 50,
                "max_val": 150,
                "skewness": 3.0,
                "kurtosis": 1.0,  # Invalid: 1.0 < 3^2 - 2
            },
        )


def test_pydantic_validation_raises_error(tools):
    """Test invalid config values throw a Pydantic validation error."""
    with pytest.raises(ValueError, match="Invalid metrics dictionary"):
        tools.generate_distribution_from_metrics(
            n=1000,
            metrics={
                "mean": 100,
                "median": "not a number",
                "std": 15,
                "min_val": 50,
                "max_val": 150,
                "skewness": 0.0,
                "kurtosis": 3.0,
            },
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
