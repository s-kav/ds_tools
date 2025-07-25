import numpy as np
import pandas as pd
import pytest
from scipy import stats


@pytest.fixture(scope="module")
def sample_data():
    np.random.seed(42)
    n = 500
    return {
        "normal": pd.Series(
            np.random.normal(loc=50, scale=10, size=n), name="Normal_Distribution"
        ),
        "uniform": pd.Series(
            np.random.uniform(low=0, high=100, size=n), name="Uniform_Distribution"
        ),
        "exponential": pd.Series(
            np.random.exponential(scale=15, size=n), name="Exponential_Distribution"
        ),
    }


def test_normal_distribution(tools, sample_data, capsys):
    # Expect p-value > 0.05, "Data looks Gaussian"
    tools.stat_normal_testing(sample_data["normal"])
    captured = capsys.readouterr()
    assert "p =" in captured.out
    assert "looks Gaussian" in captured.out or "looks normal" in captured.out


def test_uniform_distribution(tools, sample_data, capsys):
    # Expect p-value < 0.05, "Data does not look Gaussian"
    tools.stat_normal_testing(sample_data["uniform"])
    captured = capsys.readouterr()
    assert "p =" in captured.out
    assert "does not look Gaussian" in captured.out or "not normal" in captured.out


def test_exponential_distribution_basic(tools, sample_data, capsys):
    # Same expectations as uniform, p-value < 0.05 and "does not look Gaussian"
    tools.stat_normal_testing(sample_data["exponential"])
    captured = capsys.readouterr()
    assert "p =" in captured.out
    assert "does not look Gaussian" in captured.out or "not normal" in captured.out


def test_exponential_distribution_with_describe_flag(tools, sample_data, capsys):
    # With describe_flag=True, still expect same message and p-value
    tools.stat_normal_testing(sample_data["exponential"], describe_flag=True)
    captured = capsys.readouterr()
    assert "p =" in captured.out
    assert "does not look Gaussian" in captured.out or "not normal" in captured.out
    # Optionally check for mention of plots or additional descriptions (if printed)
    # e.g. assert "skewness" in captured.out or "kurtosis" in captured.out


def test_input_dataframe(tools, sample_data, capsys):
    # Provide DataFrame instead of Series, should work equivalently
    df = pd.DataFrame(sample_data["normal"])
    tools.stat_normal_testing(df)
    captured = capsys.readouterr()
    assert "p =" in captured.out
    assert "looks Gaussian" in captured.out or "looks normal" in captured.out


def test_stat_normal_testing_near_zero_moments(tools, mocker, capsys):
    """Tests branches for near-zero kurtosis and skewness."""
    mocker.patch("matplotlib.pyplot.show")

    # generate data that is very close to normal
    # stats.norm.rvs gives "cleaner" normal data
    data = pd.Series(stats.norm.rvs(loc=0, scale=1, size=5000, random_state=42))

    tools.stat_normal_testing(data)
    captured = capsys.readouterr()

    assert "Distribution has normal tail weight" in captured.out
    assert "Data are fairly symmetrical" in captured.out
