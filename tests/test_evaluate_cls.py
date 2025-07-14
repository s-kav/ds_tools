import numpy as np
import pytest
from src.ds_tool import DSTools

tools = DSTools()

@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    N = 500
    y_true = np.random.randint(0, 2, size=N)
    base_probs = np.where(y_true == 1, 0.75, 0.25)
    noise = np.random.normal(0, 0.2, size=N)
    y_probs = np.clip(base_probs + noise, 0.01, 0.99)
    return y_true, y_probs

def test_evaluate_classification_structure(synthetic_data):
    y_true, y_probs = synthetic_data
    result = tools.evaluate_classification(true_labels=y_true, pred_probs=y_probs, threshold=0.5)
    
    assert isinstance(result, dict), "The return value must be a dictionary."
    required_keys = {'accuracy', 'roc_auc', 'average_precision', 'Kolmogorov-Smirnov', 'classification_report'}
    assert required_keys.issubset(result.keys()), "Mandatory metrics are missing"
    assert 0.0 <= result['roc_auc'] <= 1.0, "ROC AUC should be in the range [0, 1]"

def test_threshold_effect_on_accuracy(synthetic_data):
    y_true, y_probs = synthetic_data
    
    metrics_low = tools.evaluate_classification(y_true, y_probs, threshold=0.5)
    metrics_high = tools.evaluate_classification(y_true, y_probs, threshold=0.7)

    assert metrics_low['accuracy'] != metrics_high['accuracy'], "Accuracy should change with threshold"
    assert np.isclose(metrics_low['roc_auc'], metrics_high['roc_auc'], atol=1e-5), "ROC AUC is independent of threshold"

def test_mismatched_shapes_raises(synthetic_data):
    y_true, y_probs = synthetic_data
    with pytest.raises(ValueError, match="Shape of true_labels and pred_probs must match."):
        tools.evaluate_classification(y_true[:-10], y_probs)
