import numpy as np
import pandas as pd
import pytest

from src.ds_tool import DSTools, CorrelationConfig

tools = DSTools()
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

    return pd.DataFrame({
        'feature_A': feature_A,
        'feature_B': feature_B,
        'feature_C': feature_C,
        'feature_D': feature_D,
        'feature_E': feature_E,
        'feature_F': feature_F
    })

def test_corr_matrix_default(test_dataframe):
    # Should execute without error and show Pearson matrix
    tools.corr_matrix(test_dataframe)

def test_corr_matrix_spearman_custom_view(test_dataframe):
    config = CorrelationConfig(
        build_method='spearman',
        font_size=10,
        image_size=(10, 10)
    )
    tools.corr_matrix(test_dataframe, config=config)

def test_corr_matrix_invalid_method_raises():
    with pytest.raises(ValueError):
        CorrelationConfig(build_method='invalid_method')
