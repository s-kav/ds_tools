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
from ds_tools.metrics import CUPY_AVAILABLE, NUMBA_AVAILABLE, Metrics

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


@pytest.fixture(scope="module")
def fft_signal_power2():
    """
    Generates a composite sine wave signal with N=128 (Power of 2).
    Frequencies: 5Hz and 20Hz.
    """
    N = 128
    t = np.linspace(0.0, 1.0, N, endpoint=False)
    # Signal: 1.0*sin(5Hz) + 0.5*sin(20Hz)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    return signal


@pytest.fixture(scope="module")
def fft_signal_odd():
    """Generates a random signal with non-power-of-2 length (N=100)."""
    np.random.seed(42)
    return np.random.random(100).astype(np.float64)


# ============================================================================
# Tests for Initialization and System Awareness
# ============================================================================


def test_metrics_initialization(mocker, capsys):
    """Tests that the Metrics class initializes correctly and checks system features."""

    metrics = Metrics(gpu_threshold=50_000)

    assert metrics.gpu_threshold == 50_000
    assert metrics.numba_available == NUMBA_AVAILABLE

    captured = capsys.readouterr()

    assert "Metrics initialized" in captured.out


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
    ("focal_loss", {"alpha": 0.25, "gamma": 2.0}),
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
    mocker.patch("ds_tools.metrics.cp.asarray", side_effect=lambda x: x, create=True)
    # Mock the backends to see which one is called
    mock_cupy = mocker.patch(
        "ds_tools.metrics._mae_cupy", return_value=1.0, create=True
    )
    mock_numba = mocker.patch("ds_tools.metrics._mae_numba", return_value=2.0)

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
    mock_numpy = mocker.patch("ds_tools.metrics._mae_numpy", return_value=99.0)
    y_true, y_pred = small_sample_data
    result = tools.metrics.mae(y_true, y_pred, force_cpu=True)
    mock_numpy.assert_called_once()
    assert result == 99.0


@pytest.mark.parametrize("engine", ["numpy", "numba"])
def test_fft_correctness_cpu(tools, fft_signal_power2, engine):
    """
    Verifies that CPU engines correctly identify frequencies in a signal.
    """
    if engine == "numba" and not NUMBA_AVAILABLE:
        pytest.skip("Numba is not available")

    # Compute FFT
    spectrum = tools.metrics.fft(fft_signal_power2, engine=engine)

    # 1. Check Output Type and Shape
    assert isinstance(spectrum, np.ndarray)
    assert len(spectrum) == 128

    # 2. Verify Spectral Peaks
    # We expect peaks at index 5 and 20 (since N=128 and T=1s, indices map to Hz)
    magnitudes = np.abs(spectrum)
    # Zero out DC component (index 0) for peak finding
    magnitudes[0] = 0

    peak_indices = np.argsort(magnitudes)[
        -4:
    ]  # Top 4 peaks (positive & negative freqs)

    assert 5 in peak_indices, f"Failed to detect 5Hz component with {engine}"
    assert 20 in peak_indices, f"Failed to detect 20Hz component with {engine}"


@pytest.mark.parametrize("engine", ["numpy", "numba"])
def test_fft_inverse_cpu(tools, fft_signal_power2, engine):
    """
    Verifies that IFFT reconstructs the original signal (Round-trip test).
    """
    if engine == "numba" and not NUMBA_AVAILABLE:
        pytest.skip("Numba is not available")

    # Forward
    spectrum = tools.metrics.fft(fft_signal_power2, engine=engine)
    # Inverse
    reconstructed = tools.metrics.fft(spectrum, inverse=True, engine=engine)

    # Numba implementation returns complex array, take real part
    assert np.allclose(reconstructed.real, fft_signal_power2, atol=1e-5)


def test_numba_padding_logic(tools, fft_signal_odd):
    """
    Tests that Numba implementation automatically pads non-power-of-2 inputs.
    Input N=100 -> Expected Output N=128.
    """
    if not NUMBA_AVAILABLE:
        pytest.skip("Numba is not available")

    spectrum = tools.metrics.fft(fft_signal_odd, engine="numba")
    assert len(spectrum) == 128
    assert np.iscomplexobj(spectrum)


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

    def test_fft_correctness_cupy(self, tools, fft_signal_power2):
        """
        Verifies CuPy FFT execution and correctness against NumPy baseline.
        """
        import cupy as cp

        # 1. Execute on GPU
        spectrum_gpu = tools.metrics.fft(fft_signal_power2)

        # Check it stays on GPU (is a cupy array)
        assert isinstance(spectrum_gpu, cp.ndarray)

        # 2. Compare values with NumPy
        spectrum_cpu = tools.metrics.ft(fft_signal_power2)

        # Transfer back to CPU for comparison
        np.testing.assert_allclose(cp.asnumpy(spectrum_gpu), spectrum_cpu, atol=1e-5)

    def test_fft_inverse_cupy(self, tools, fft_signal_power2):
        """
        Verifies CuPy IFFT round-trip.
        """
        import cupy as cp

        # Forward
        spectrum_gpu = tools.metrics.fft(fft_signal_power2)
        # Inverse
        reconstructed_gpu = tools.metrics.fft(spectrum_gpu, inverse=True)

        reconstructed_cpu = cp.asnumpy(reconstructed_gpu).real
        np.testing.assert_allclose(reconstructed_cpu, fft_signal_power2, atol=1e-5)

    def test_fft_cupy_input_types(self, tools, fft_signal_power2):
        """
        Tests that CuPy engine accepts both list and numpy array inputs
        by handling conversion internally.
        """
        import cupy as cp

        # Pass list
        data_list = fft_signal_power2.tolist()
        res_list = tools.metrics.fft(data_list)
        assert isinstance(res_list, cp.ndarray)


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


# --- Additional Metric Tests ---
def test_focal_loss_correctness(tools):
    """Tests Focal Loss against a simple manual calculation."""
    # y=1, p=0.9, alpha=0.25, gamma=2
    # loss = -0.25 * (1-0.9)^2 * log(0.9) = -0.25 * 0.01 * -0.10536 = 0.00026
    y_true = np.array([1.0], dtype=np.float32)
    y_pred = np.array([0.9], dtype=np.float32)

    loss = tools.metrics.focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0)

    # Approx calc: -0.25 * 0.01 * np.log(0.9)
    expected = -0.25 * (1 - 0.9) ** 2 * np.log(0.9)
    assert np.isclose(loss, expected, rtol=1e-4)


def test_categorical_crossentropy_correctness(tools):
    """Tests CCE with one-hot inputs."""
    # 2 samples, 3 classes
    y_true = np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float32)
    y_pred = np.array([[0.1, 0.8, 0.1], [0.2, 0.2, 0.6]], dtype=np.float32)

    loss = tools.metrics.categorical_crossentropy(y_true, y_pred, from_logits=False)

    # Manual: -(log(0.8) + log(0.6)) / 2
    expected = -(np.log(0.8) + np.log(0.6)) / 2
    assert np.isclose(loss, expected, rtol=1e-5)


def test_contrastive_loss_correctness(tools):
    """Tests Contrastive Loss and its gradients (covering _contrastive_loss_grad_numpy)."""
    emb1 = np.array([[1.0, 0.0]], dtype=np.float32)
    emb2 = np.array([[1.0, 0.0]], dtype=np.float32)

    # --- Case 1: Similar pair (y=1) ---
    y_sim = np.array([1.0], dtype=np.float32)
    # Distance is 0. Loss should be dist^2 = 0
    loss = tools.metrics.contrastive_loss(y_sim, emb1, emb2)
    assert np.isclose(loss, 0.0)

    # Check Gradient for Similar pair (force_cpu=True triggers _contrastive_loss_grad_numpy)
    loss_val, (g1, g2) = tools.metrics.contrastive_loss(
        y_sim, emb1, emb2, return_grad=True, force_cpu=True
    )
    assert isinstance(g1, np.ndarray)
    assert g1.shape == emb1.shape

    # --- Case 2: Dissimilar pair (y=0) ---
    y_dissim = np.array([0.0], dtype=np.float32)
    margin = 1.0
    # Loss should be max(0, 1 - 0)^2 = 1. (Our impl returns 0.5 * loss)
    loss_dissim = tools.metrics.contrastive_loss(y_dissim, emb1, emb2, margin=margin)
    assert np.isclose(loss_dissim, 0.5)

    # Check Gradient for Dissimilar pair
    loss_val_d, (g1_d, g2_d) = tools.metrics.contrastive_loss(
        y_dissim, emb1, emb2, margin=margin, return_grad=True, force_cpu=True
    )
    assert isinstance(g1_d, np.ndarray)
    # Gradient shouldn't be None (ensures we entered the calculation block)
    assert not np.isnan(g1_d).any()


def test_mmd_loss_kernel(tools):
    """Tests MMD loss (identity)."""
    X = np.array([[1.0], [2.0]], dtype=np.float32)
    # MMD between same distributions should be close to 0
    loss = tools.metrics.mmd_loss(X, X)
    assert np.isclose(loss, 0.0, atol=1e-6)


def test_wasserstein_approx(tools):
    """Tests 1D Wasserstein approximation."""
    # Dist 1: [1, 2], Dist 2: [2, 3] (shifted by 1)
    # Earth mover distance should be 1.0
    u = np.array([1, 2], dtype=np.float32)
    v = np.array([2, 3], dtype=np.float32)
    dist = tools.metrics.wasserstein_approx(u, v)
    assert np.isclose(dist, 1.0)


def test_cohens_d_magnitude(tools):
    """
    Specific test for Cohen's d magnitude interpretation.
    Scenario:
        Group 1: Mean ~10, Std ~2
        Group 2: Mean ~12, Std ~2
    Expected:
        Difference in means = -2
        Pooled Std approx = 2
        Cohen's d approx = -1.0
    """
    np.random.seed(42)
    # Use samples > 1000 to avoid small sample warning
    g1 = np.random.normal(10, 2, 2000).astype(np.float32)
    g2 = np.random.normal(12, 2, 2000).astype(np.float32)

    d_val = tools.metrics.cohens_d(g1, g2, force_cpu=True)

    # Allow small tolerance for random generation variance
    assert np.isclose(d_val, -1.0, atol=0.1)


def test_cohens_d_small_sample_warning(tools):
    """
    Tests that Cohen's d issues a UserWarning when sample sizes are small (< 1000),
    as reliability decreases with small N.
    """
    g1 = np.random.rand(50)
    g2 = np.random.rand(50)

    with pytest.warns(UserWarning, match="Small sample sizes"):
        tools.metrics.cohens_d(g1, g2, force_cpu=True)


def test_cohens_d_zero_variance(tools):
    """
    Tests Cohen's d handling when pooled standard deviation is zero
    (e.g., identical constant arrays). Should avoid DivisionByZero error.
    """
    # Constant arrays have variance = 0
    g1 = np.ones(2000, dtype=np.float32) * 5
    g2 = np.ones(2000, dtype=np.float32) * 5

    d_val = tools.metrics.cohens_d(g1, g2, force_cpu=True)
    assert d_val == 0.0


def test_cohens_d_unequal_sizes(tools):
    """
    Tests that Cohen's d supports unequal group sizes, unlike standard
    element-wise metrics (MAE/MSE) which require shape matching.
    """
    np.random.seed(42)
    g1 = np.random.normal(0, 1, 1500)
    g2 = np.random.normal(0, 1, 2500)

    # Should execute without raising "Input arrays must have the same shape"
    d_val = tools.metrics.cohens_d(g1, g2, force_cpu=True)
    assert isinstance(d_val, float)


def test_list_metrics_function(tools):
    """Tests that list_metrics returns a valid DataFrame."""
    df = tools.metrics.list_metrics()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Metric" in df.columns
    assert "mae" in df["Metric"].tolist()
    assert "focal_loss" in df["Metric"].tolist()
