# tests/test_metrics.py
"""
Comprehensive tests for the Metrics class in ds_tools.
This suite covers initialization, backend dispatching (NumPy, Numba, CuPy),
gradient calculations, and the real-time monitoring system.

This set of tests covers almost 100% of the functionality, including:
All public functions (MSE, MAE, RMSE, Huber, Quantile, Hinge, LogLoss, Triplet).
All backends (NumPy, Numba, CuPy) via parameterization and mocking.
Gradient calculation for all applicable functions.
Monitoring system (start, update, get_history_df, plot).
Dispatching logic (force_cpu flag and gpu_threshold threshold).
Error and edge case handling (empty arrays, size mismatch, invalid parameters).
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

import numpy as np
import pandas as pd
import pytest

from ds_tool import Metrics
from metrics import CUPY_AVAILABLE, NUMBA_AVAILABLE

pytestmark_cupy = pytest.mark.skipif(
    not CUPY_AVAILABLE, reason="CuPy or compatible GPU is not available"
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def small_sample_data():
    """Provides a small dataset that won't trigger the GPU threshold."""
    np.random.seed(42)
    y_true = np.random.rand(1000).astype(np.float32)
    y_pred = y_true + np.random.normal(0, 0.1, 1000).astype(np.float32)
    return y_true, y_pred


@pytest.fixture(scope="module")
def large_sample_data():
    """Provides a large dataset that WILL trigger the GPU threshold."""
    np.random.seed(42)
    # Use a size larger than the default 100k threshold
    size = 150_000
    y_true = np.random.rand(size).astype(np.float32)
    y_pred = y_true + np.random.normal(0, 0.1, size).astype(np.float32)
    return y_true, y_pred


@pytest.fixture(scope="module")
def triplet_data():
    """Provides anchor, positive, and negative samples for triplet loss."""
    np.random.seed(42)
    anchor = np.random.rand(100, 32).astype(np.float32)
    # Positive is close to anchor
    positive = anchor + np.random.normal(0, 0.001, (100, 32)).astype(np.float32)
    # Negative is far from anchor
    negative = anchor + np.random.normal(0, 0.1, (100, 32)).astype(np.float32)
    return anchor, positive, negative


# ============================================================================
# Tests for Initialization and System Awareness
# ============================================================================


def test_metrics_initialization(mocker, capsys):
    """Tests that the Metrics class initializes correctly and checks system features."""
    mock_cpu_count = mocker.patch("psutil.cpu_count", return_value=8)

    metrics = Metrics(gpu_threshold=50_000)

    assert metrics.gpu_threshold == 50_000
    assert metrics.numba_available == NUMBA_AVAILABLE

    captured = capsys.readouterr()
    if NUMBA_AVAILABLE:
        mock_cpu_count.assert_called_once_with(logical=True)

    assert "Metrics initialized" in captured.out


def test_metrics_init_psutil_fails(mocker):
    """Covers the exception handling if psutil fails."""
    if not NUMBA_AVAILABLE:
        pytest.skip("Test requires Numba")
    mocker.patch("psutil.cpu_count", side_effect=Exception("Test psutil error"))

    with pytest.warns(UserWarning, match="Could not set Numba threads with psutil"):
        Metrics()


# ============================================================================
# Tests for the Real-time Monitoring System
# ============================================================================


def test_monitoring_system(tools, mocker):
    """Tests the full lifecycle of the monitoring system."""
    # 1. Start monitoring
    tools.metrics.start_monitoring()
    assert tools.metrics.history == {}

    # 2. Update with logs
    tools.metrics.update(epoch=0, logs={"loss": 0.5, "accuracy": 0.8, "val_loss": 0.6})
    tools.metrics.update(epoch=1, logs={"loss": 0.4, "accuracy": 0.85, "val_loss": 0.5})
    tools.metrics.update(epoch=2, logs={"loss": 0.3, "accuracy": 0.9, "val_loss": 0.35})

    expected_history = {
        "epoch": [0, 1, 2],
        "loss": [0.5, 0.4, 0.3],
        "accuracy": [0.8, 0.85, 0.9],
        "val_loss": [0.35],  # Note: history handles missing values correctly
    }

    # Check internal state (adjust for val_loss padding)
    history = tools.metrics.history
    assert history["epoch"] == expected_history["epoch"]
    assert history["loss"] == expected_history["loss"]
    assert history["accuracy"] == expected_history["accuracy"]
    assert history["val_loss"] == [0.6, 0.5, 0.35]
    assert len(history["val_loss"]) == 3

    # 3. Get DataFrame
    df = tools.metrics.get_history_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "val_loss" in df.columns
    assert df["val_loss"].iloc[0] == 0.6

    # 4. Test plotting
    mocker.patch("matplotlib.pyplot.show")  # Mock show to prevent blocking
    tools.metrics.plot_history()
    # Check graceful handling of no history
    tools.metrics.start_monitoring()
    assert tools.metrics.get_history_df() is None
    tools.metrics.plot_history()  # Should not raise an error


# ============================================================================
# Tests for Core Metric Calculations and Dispatching
# ============================================================================

# --- Test data for parametrization ---
METRICS_TO_TEST = [
    ("mae", {}),
    ("mse", {}),
    ("rmse", {}),
    ("huber_loss", {"delta": 0.5}),
    ("quantile_loss", {"quantile": 0.75}),
    ("hinge_loss", {}),
    ("log_loss", {}),
    ("cross_entropy_loss", {}),
]


@pytest.mark.parametrize("metric_name, kwargs", METRICS_TO_TEST)
@pytest.mark.parametrize("force_cpu", [True, False])
def test_metric_correctness(tools, metric_name, kwargs, force_cpu, large_sample_data):
    """
    Tests the numerical correctness of all metrics against a trusted NumPy implementation,
    across different backends (CPU/GPU).
    """
    y_true, y_pred = large_sample_data

    # Special handling for classification metrics
    if metric_name in ["hinge_loss", "log_loss", "cross_entropy_loss"]:
        y_true = np.random.randint(0, 2, size=len(y_true)).astype(np.float32)
        y_pred = np.random.rand(len(y_pred)).astype(np.float32)
        if metric_name == "hinge_loss":
            y_true[y_true == 0] = -1  # Hinge expects {-1, 1}

    # Get the result from our high-performance function
    metric_func = getattr(tools.metrics, metric_name)
    result = metric_func(y_true, y_pred, force_cpu=force_cpu, **kwargs)

    # Calculate the expected result using a simple, trusted NumPy implementation
    if metric_name == "mae":
        expected = np.mean(np.abs(y_true - y_pred))
    elif metric_name == "mse":
        expected = np.mean((y_true - y_pred) ** 2)
    elif metric_name == "rmse":
        expected = np.sqrt(np.mean((y_true - y_pred) ** 2))
    elif metric_name == "huber_loss":
        delta = kwargs["delta"]
        err = np.abs(y_true - y_pred)
        expected = np.mean(
            np.where(err <= delta, 0.5 * err**2, delta * (err - 0.5 * delta))
        )
    elif metric_name == "quantile_loss":
        q = kwargs["quantile"]
        err = y_true - y_pred
        expected = np.mean(np.maximum(q * err, (q - 1) * err))
    elif metric_name == "hinge_loss":
        expected = np.mean(np.maximum(0.0, 1.0 - y_true * y_pred))
    elif metric_name in ["log_loss", "cross_entropy_loss"]:
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        expected = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    assert np.isclose(result, expected, rtol=1e-5)


@pytest.mark.parametrize("metric_name, kwargs", METRICS_TO_TEST)
def test_gradients(tools, metric_name, kwargs, small_sample_data):
    """
    Tests that gradient calculation works and returns the correct shape.
    """
    y_true, y_pred = small_sample_data
    if metric_name in ["hinge_loss", "log_loss", "cross_entropy_loss"]:
        y_true = np.random.randint(0, 2, size=len(y_true)).astype(np.float32)
        if metric_name == "hinge_loss":
            y_true[y_true == 0] = -1

    metric_func = getattr(tools.metrics, metric_name)

    # Get both loss and gradient
    loss, grad = metric_func(y_true, y_pred, return_grad=True, force_cpu=True, **kwargs)

    assert isinstance(
        loss, (float, np.floating)
    ), f"Loss for {metric_name} should be a float or numpy float, but got {type(loss)}"
    assert isinstance(grad, np.ndarray)
    assert grad.shape == y_pred.shape


def test_gpu_threshold_logic(tools, mocker, small_sample_data, large_sample_data):
    """Tests that the GPU is only used when data size exceeds the threshold."""
    if not (CUPY_AVAILABLE and NUMBA_AVAILABLE):
        pytest.skip("This test requires both CuPy and Numba to be installed.")

    mocker.patch.object(tools.metrics, "gpu_available", True)
    mocker.patch("metrics.cp.asarray", side_effect=lambda x: x, create=True)
    # Mock the backends to see which one is called
    mock_cupy = mocker.patch("metrics._mae_cupy", return_value=1.0, create=True)
    mock_numba = mocker.patch("metrics._mae_numba", return_value=2.0)

    # 1. Test with small data -> Numba should be called
    y_true_small, y_pred_small = small_sample_data
    tools.metrics.mae(y_true_small, y_pred_small, force_cpu=False)
    mock_cupy.assert_not_called()
    mock_numba.assert_called_once()
    mock_numba.reset_mock()

    # 2. Test with large data -> CuPy should be called
    y_true_large, y_pred_large = large_sample_data
    tools.metrics.mae(y_true_large, y_pred_large, force_cpu=False)
    mock_cupy.assert_called_once()
    mock_numba.assert_not_called()


@pytest.mark.skipif(
    NUMBA_AVAILABLE, reason="This test is for when Numba is NOT available"
)
def test_dispatch_fallback_to_numpy(tools, mocker, small_sample_data):
    """Tests that dispatcher falls back to NumPy if Numba is missing."""
    mock_numpy = mocker.patch("metrics._mae_numpy", return_value=99.0)
    y_true, y_pred = small_sample_data
    result = tools.metrics.mae(y_true, y_pred, force_cpu=True)
    mock_numpy.assert_called_once()
    assert result == 99.0


# ============================================================================
# Tests for GPU-Specific Execution
# ============================================================================


@pytestmark_cupy
class TestGPUBackends:
    """A class to group tests that require a functional CuPy environment."""

    @pytest.mark.parametrize("metric_name, kwargs", METRICS_TO_TEST)
    def test_metric_correctness_gpu(
        self, tools, metric_name, kwargs, large_sample_data
    ):
        """Tests numerical correctness of all metrics on GPU vs NumPy."""
        y_true, y_pred = large_sample_data
        if metric_name in ["hinge_loss", "log_loss"]:
            y_true = np.random.randint(0, 2, len(y_true)).astype(np.float32)
            y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
            if metric_name == "hinge_loss":
                y_true[y_true == 0] = -1

        metric_func = getattr(tools.metrics, metric_name)
        result_gpu = metric_func(y_true, y_pred, force_cpu=False, **kwargs)
        result_cpu = metric_func(y_true, y_pred, force_cpu=True, **kwargs)

        assert np.isclose(result_gpu, result_cpu, rtol=1e-5)

    @pytest.mark.parametrize("metric_name, kwargs", METRICS_TO_TEST)
    def test_gradients_gpu(self, tools, metric_name, kwargs, large_sample_data):
        """Tests that gradient calculation on GPU returns correct shapes and types."""
        y_true, y_pred = large_sample_data
        if metric_name in ["hinge_loss", "log_loss"]:
            y_true = np.random.randint(0, 2, len(y_true)).astype(np.float32)
            if metric_name == "hinge_loss":
                y_true[y_true == 0] = -1

        metric_func = getattr(tools.metrics, metric_name)
        loss, grad = metric_func(
            y_true, y_pred, return_grad=True, force_cpu=False, **kwargs
        )

        assert isinstance(loss, (float, np.floating))
        assert isinstance(grad, np.ndarray)  # Should be converted back to numpy
        assert grad.shape == y_pred.shape

    def test_triplet_loss_gpu(self, tools, triplet_data):
        """Tests Triplet Loss on the GPU."""
        anchor, positive, negative = triplet_data
        loss_gpu, (ga_gpu, gp_gpu, gn_gpu) = tools.metrics.triplet_loss(
            anchor, positive, negative, return_grad=True, force_cpu=False
        )
        loss_cpu, (ga_cpu, gp_cpu, gn_cpu) = tools.metrics.triplet_loss(
            anchor, positive, negative, return_grad=True, force_cpu=True
        )
        assert np.isclose(loss_gpu, loss_cpu)
        assert isinstance(ga_gpu, np.ndarray)


# ============================================================================
# Tests for Specific/Edge Cases
# ============================================================================


def test_empty_input(tools):
    """Tests that all metrics handle empty arrays gracefully."""
    empty_arr = np.array([])
    assert tools.metrics.mae(empty_arr, empty_arr) == 0.0
    assert tools.metrics.mse(empty_arr, empty_arr) == 0.0
    val, grad = tools.metrics.mae(empty_arr, empty_arr, return_grad=True)
    assert val == 0.0 and grad is None

    anchor, pos, neg = empty_arr, empty_arr, empty_arr
    assert tools.metrics.triplet_loss(anchor, pos, neg) == 0.0
    val, grad_tuple = tools.metrics.triplet_loss(anchor, pos, neg, return_grad=True)
    assert val == 0.0 and grad_tuple == (None, None, None)


def test_shape_mismatch_raises_error(tools):
    """Tests that a shape mismatch in inputs raises a ValueError."""
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2])
    with pytest.raises(ValueError, match="Input arrays must have the same shape"):
        tools.metrics.mae(y_true, y_pred)

    anchor = np.zeros((3, 2))
    positive = np.zeros((3, 2))
    negative = np.zeros((2, 2))  # Mismatched shape
    with pytest.raises(ValueError, match="Input arrays must have the same shape"):
        tools.metrics.triplet_loss(anchor, positive, negative)


def test_invalid_quantile_raises_error(tools):
    """Tests that an invalid quantile value raises a ValueError."""
    y_true, y_pred = np.array([1]), np.array([1])
    with pytest.raises(ValueError, match="Quantile must be between 0 and 1"):
        tools.metrics.quantile_loss(y_true, y_pred, quantile=1.5)


def test_triplet_loss_correctness(tools, triplet_data):
    """Tests the Triplet Loss calculation."""
    anchor, positive, negative = triplet_data
    margin = 0.5

    # Case 1: Loss should be positive
    loss = tools.metrics.triplet_loss(anchor, positive, negative, margin=margin)
    assert loss > 0

    # Case 2: Loss should be zero (negative is much farther than positive)
    far_negative = negative * 10
    loss_zero = tools.metrics.triplet_loss(
        anchor, positive, far_negative, margin=margin
    )
    assert np.isclose(loss_zero, 0.0)

    # Case 3: Test gradients
    loss_val, (grad_a, grad_p, grad_n) = tools.metrics.triplet_loss(
        anchor, positive, negative, margin=margin, return_grad=True, force_cpu=True
    )
    assert grad_a.shape == anchor.shape
    assert grad_p.shape == positive.shape
    assert grad_n.shape == negative.shape


def test_mae_grad_zero_difference(tools):
    """
    Covers the 'else 0.0' branch in MAE gradient calculation when y_true == y_pred.
    """
    y_true = np.array([1, 2, 3], dtype=np.float32)
    y_pred = np.array([1, 2, 3], dtype=np.float32)

    _, grad = tools.metrics.mae(y_true, y_pred, return_grad=True, force_cpu=True)

    # Gradient should be all zeros
    assert np.all(grad == 0.0)


def test_huber_loss_branches(tools):
    """
    Covers both branches (abs_err <= delta and abs_err > delta) of Huber loss.
    """
    delta = 1.0
    y_true = np.array([10, 10], dtype=np.float32)

    # Case 1: Error is SMALLER than delta (quadratic region)
    y_pred_small_err = np.array([10.5, 9.5], dtype=np.float32)  # error = 0.5
    loss_small, grad_small = tools.metrics.huber_loss(
        y_true, y_pred_small_err, delta=delta, return_grad=True, force_cpu=True
    )

    expected_loss_small = np.mean(0.5 * (0.5**2))
    expected_grad_small = (y_pred_small_err - y_true) / len(y_true)
    assert np.isclose(loss_small, expected_loss_small)
    assert np.allclose(grad_small, expected_grad_small)

    # Case 2: Error is LARGER than delta (linear region)
    y_pred_large_err = np.array([12, 8], dtype=np.float32)  # error = 2.0
    loss_large, grad_large = tools.metrics.huber_loss(
        y_true, y_pred_large_err, delta=delta, return_grad=True, force_cpu=True
    )

    expected_loss_large = np.mean(delta * (2.0 - 0.5 * delta))
    # Expected gradient is delta * sign(error)
    expected_grad_large = (np.array([delta, -delta])) / len(y_true)
    assert np.isclose(loss_large, expected_loss_large)
    assert np.allclose(grad_large, expected_grad_large)


def test_quantile_loss_branches(tools):
    """
    Covers both branches (err > 0 and err <= 0) of Quantile loss.
    """
    quantile = 0.8
    y_true = np.array([10, 10], dtype=np.float32)
    y_pred = np.array([9, 11], dtype=np.float32)  # One overestimate, one underestimate
    # err = y_true - y_pred = [1, -1]

    loss, grad = tools.metrics.quantile_loss(
        y_true, y_pred, quantile=quantile, return_grad=True, force_cpu=True
    )

    # Expected loss: mean( quantile * 1, (quantile - 1) * -1 )
    expected_loss = np.mean([quantile * 1, (quantile - 1) * -1])
    # Expected grad: where(err > 0, q-1, q) / n
    expected_grad = np.array([quantile - 1.0, quantile]) / len(y_true)

    assert np.isclose(loss, expected_loss)
    assert np.allclose(grad, expected_grad)


def test_hinge_loss_branches(tools):
    """
    Covers both branches (loss > 0 and loss == 0) of Hinge loss.
    """
    y_true = np.array([1, -1], dtype=np.float32)

    # Case 1: Incurring loss (1 - y_true * y_pred > 0)
    # y_true=1, y_pred=0.5 -> 1 - 0.5 = 0.5 > 0
    # y_true=-1, y_pred=0.5 -> 1 - (-0.5) = 1.5 > 0
    y_pred_loss = np.array([0.5, 0.5], dtype=np.float32)
    loss1, grad1 = tools.metrics.hinge_loss(
        y_true, y_pred_loss, return_grad=True, force_cpu=True
    )

    assert loss1 > 0
    assert not np.all(grad1 == 0)

    # Case 2: No loss (1 - y_true * y_pred <= 0)
    # y_true=1, y_pred=1.5 -> 1 - 1.5 = -0.5 <= 0
    # y_true=-1, y_pred=-1.5 -> 1 - 1.5 = -0.5 <= 0
    y_pred_no_loss = np.array([1.5, -1.5], dtype=np.float32)
    loss2, grad2 = tools.metrics.hinge_loss(
        y_true, y_pred_no_loss, return_grad=True, force_cpu=True
    )

    assert np.isclose(loss2, 0.0)
    assert np.all(grad2 == 0.0)
