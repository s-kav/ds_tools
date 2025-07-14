import numpy as np
import pytest

from src.ds_tool import DSTools, MetricsConfig

tools = DSTools()
N_SAMPLES = 200
THRESHOLD = 0.5

@pytest.fixture(scope="module")
def classification_data():
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=N_SAMPLES)
    base_probs = np.where(y_true == 1, 0.8, 0.2)
    noise = np.random.normal(0, 0.25, size=N_SAMPLES)
    y_predict_proba = np.clip(base_probs + noise, 0, 1)
    y_predict = (y_predict_proba >= THRESHOLD).astype(int)
    return y_true, y_predict, y_predict_proba

def test_compute_metrics_default(classification_data):
    y_true, y_pred, y_proba = classification_data
    df = tools.compute_metrics(y_true, y_pred, y_proba)
    assert isinstance(df, np.ndarray) or hasattr(df, 'shape')
    assert df.shape[0] >= 1  # Sanity check on rows
    expected_cols = {'Average_precision, %', 'Balanced_accuracy, %', 'Likelihood_ratios+', 'Likelihood_ratios-', 'Kappa_score, %'}
    assert expected_cols.issubset(df.columns)

def test_compute_metrics_custom_config(classification_data, capsys):
    y_true, y_pred, y_proba = classification_data
    config = MetricsConfig(error_vis=False, print_values=True)
    df = tools.compute_metrics(y_true, y_pred, y_proba, config=config)
    captured = capsys.readouterr()
    assert "Average_precision" in captured.out and "Balanced_accuracy" in captured.out
    assert df.shape[0] >= 1
