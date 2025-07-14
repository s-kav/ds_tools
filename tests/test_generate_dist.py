import numpy as np
import pytest
from scipy import stats
from src.ds_tool import DSTools, DistributionConfig

tools = DSTools()

def describe_generated_data(data: np.ndarray):
    return {
        "mean": np.mean(data),
        "std": np.std(data, ddof=1),
        "skew": stats.skew(data),
        "kurt": stats.kurtosis(data) + 3,  # Convert to Pearson's kurtosis
        "min": np.min(data),
        "max": np.max(data),
        "n": len(data)
    }

@pytest.mark.parametrize("config", [
    DistributionConfig(
        mean=1000, median=950, std=200, min_val=400, max_val=2500,
        skewness=0.8, kurtosis=4.0, n=2000, outlier_ratio=0.01
    ),
    DistributionConfig(
        mean=50, median=48, std=10, min_val=10, max_val=150,
        skewness=1.5, kurtosis=8.0, n=2000, outlier_ratio=0.03
    )
])
def test_generate_distribution_valid_configs(config):
    data = tools.generate_distribution(config)
    stats_actual = describe_generated_data(data)
    
    assert np.isclose(stats_actual["mean"], config.mean, rtol=0.8)
    assert np.isclose(stats_actual["std"], config.std, rtol=5.0)
    assert stats_actual["min"] >= config.min_val
    assert stats_actual["max"] <= config.max_val
    assert stats_actual["n"] == config.n

def test_generate_distribution_invalid_moments():
    config_invalid = DistributionConfig(
        mean=100, median=100, std=15, min_val=50, max_val=150,
        skewness=2.0, kurtosis=1.0, n=1000  # Invalid: kurt < skew² - 2
    )
    with pytest.raises(ValueError, match="Invalid statistical moments"):
        tools.generate_distribution(config_invalid)

def test_distribution_config_pydantic_validation():
    with pytest.raises(ValueError, match="max_val must be greater than min_val"):
        DistributionConfig(
            mean=100, median=100, std=15, min_val=200, max_val=100,
            skewness=0.0, kurtosis=3.0, n=1000
        )
