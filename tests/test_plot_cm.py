import pytest
import numpy as np
from src.ds_tool import DSTools

@pytest.fixture(scope="module")
def tools():
    return DSTools()

@pytest.fixture(scope="module")
def binary_data():
    np.random.seed(42)
    n = 200
    y_true = np.random.randint(0, 2, size=n)
    y_pred = np.where(np.random.rand(n) < 0.85, y_true, 1 - y_true)
    return y_true, y_pred

@pytest.fixture(scope="module")
def multiclass_data():
    np.random.seed(42)
    n = 200
    y_true = np.random.randint(0, 3, size=n)
    correct_preds = np.random.rand(n) < 0.75
    random_errors = np.random.randint(1, 3, size=n)
    y_pred = np.where(correct_preds, y_true, (y_true + random_errors) % 3)
    return y_true, y_pred

# Helper to suppress plotting during tests
import matplotlib
matplotlib.use('Agg')

def test_plot_confusion_matrix_binary_default_labels(tools, binary_data):
    y_true, y_pred = binary_data
    # Should run without error and produce a 2x2 confusion matrix plot
    tools.plot_confusion_matrix(y_true, y_pred)

def test_plot_confusion_matrix_binary_custom_labels_and_cmap(tools, binary_data):
    y_true, y_pred = binary_data
    tools.plot_confusion_matrix(
        y_true,
        y_pred,
        class_labels=['Negative (0)', 'Positive (1)'],
        title='Binary Classification Performance',
        cmap='Greens'
    )

def test_plot_confusion_matrix_multiclass_custom_labels(tools, multiclass_data):
    y_true, y_pred = multiclass_data
    tools.plot_confusion_matrix(
        y_true,
        y_pred,
        class_labels=['Cat', 'Dog', 'Bird'],
        title='Multi-Class Classification (Animals)',
        cmap='YlGnBu'
    )

def test_plot_confusion_matrix_raises_value_error_for_invalid_labels(tools, binary_data):
    y_true, y_pred = binary_data
    with pytest.raises(ValueError):
        tools.plot_confusion_matrix(
            y_true,
            y_pred,
            class_labels=['One label']  # Incorrect number of labels for 2 classes
        )
