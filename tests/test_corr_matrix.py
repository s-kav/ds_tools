import numpy as np
import pandas as pd
import pytest

from ds_tool import CorrelationConfig

N_SAMPLES = 100


@pytest.fixture(scope="module")
def test_dataframe():
    np.random.seed(42)
    feature_A = np.linspace(-10, 10, N_SAMPLES)
    feature_B = feature_A * 2 + np.random.normal(0, 2, N_SAMPLES)
    feature_C = -feature_A * 1.5 + np.random.normal(0, 3, N_SAMPLES)
    feature_D = feature_A**2 + np.random.normal(0, 5, N_SAMPLES)
    feature_E = np.random.rand(N_SAMPLES) * 20
    feature_F = feature_B * 0.5 + np.random.normal(0, 10, N_SAMPLES)

    return pd.DataFrame(
        {
            "feature_A": feature_A,
            "feature_B": feature_B,
            "feature_C": feature_C,
            "feature_D": feature_D,
            "feature_E": feature_E,
            "feature_F": feature_F,
        }
    )


def test_corr_matrix_default(tools, test_dataframe):
    # Should execute without error and show Pearson matrix
    tools.corr_matrix(test_dataframe)


def test_corr_matrix_spearman_custom_view(tools, test_dataframe):
    config = CorrelationConfig(
        build_method="spearman", font_size=10, image_size=(10, 10)
    )
    tools.corr_matrix(test_dataframe, config=config)


def test_corr_matrix_invalid_method_raises():
    with pytest.raises(ValueError):
        CorrelationConfig(build_method="invalid_method")


@pytest.mark.parametrize(
    "num_cols, expected_size",
    [
        (4, (8, 8)),  # n_cols < 5
        (8, (10, 10)),  # n_cols < 9
        (14, (22, 22)),  # n_cols < 15
        (20, (16, 16)),  # else, default config size
    ],
)
def test_corr_matrix_fig_size_logic(tools, mocker, num_cols, expected_size):
    """Tests the figure size selection logic based on number of columns."""
    # Mock plt.subplots to access its arguments
    mock_subplots = mocker.patch(
        "matplotlib.pyplot.subplots", return_value=(mocker.Mock(), mocker.Mock())
    )
    mocker.patch("matplotlib.pyplot.show")  # We mock show so as not to block the test

    df = pd.DataFrame({f"col_{i}": range(10) for i in range(num_cols)})
    tools.corr_matrix(df)

    # Checking what figsize subplots was called with
    mock_subplots.assert_called_once()
    call_args, call_kwargs = mock_subplots.call_args
    assert call_kwargs.get("figsize") == expected_size
