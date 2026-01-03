# src/ds_tools/metrics.py
"""
/*
 * Copyright (c) [2025] [Sergii Kavun]
 *
 * This software is dual-licensed:
 * - PolyForm Noncommercial 1.0.0 (default)
 * - Commercial license available
 *
 * See LICENSE for details
 */


This module provides a collection of high-performance, system-aware metric
and loss function implementations, with support for gradient calculation and
real-time monitoring.
"""
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# --- Environment Checks for Optional Dependencies ---
try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn(
        "Numba not installed. CPU parallel implementations will be unavailable."
    )

try:
    import cupy as cp

    try:
        if cp.cuda.runtime.getDeviceCount() > 0:
            CUPY_AVAILABLE = True
            cp.cuda.Device(0).use()
        else:
            CUPY_AVAILABLE = False
            warnings.warn("CuPy is installed, but no compatible CUDA GPU was found.")
    except cp.cuda.runtime.CUDARuntimeError as e:
        CUPY_AVAILABLE = False
        warnings.warn(f"CuPy installed but failed to initialize: {e}")
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not installed. GPU acceleration will be unavailable.")

# ============================================================================
# NUMBA IMPLEMENTATIONS (JIT-compiled, CPU optimized)
# ============================================================================


EPSILON = 1e-15  # for numerical stability

if NUMBA_AVAILABLE:
    # --- MSE Implementations ---
    @jit("float64(float32[:], float32[:])", nopython=True, parallel=True, fastmath=True)
    def _mse_numba(y_true, y_pred):  # pragma: no cover
        n = y_true.shape[0]
        error = 0.0
        for i in prange(n):
            error += (y_true[i] - y_pred[i]) ** 2
        return error / n

    @jit(
        "float32[:](float32[:], float32[:])",
        nopython=True,
        parallel=True,
        fastmath=True,
    )
    def _mse_grad_numba(y_true, y_pred):  # pragma: no cover
        n = y_true.shape[0]
        grad = np.empty_like(y_pred)
        for i in prange(n):
            grad[i] = 2.0 / n * (y_pred[i] - y_true[i])
        return grad

    # --- MAE Implementations ---
    @jit("float64(float32[:], float32[:])", nopython=True, parallel=True, fastmath=True)
    def _mae_numba(y_true, y_pred):  # pragma: no cover
        n = y_true.shape[0]
        error = 0.0
        for i in prange(n):
            error += abs(y_true[i] - y_pred[i])
        return error / n

    @jit(
        "float32[:](float32[:], float32[:])",
        nopython=True,
        parallel=True,
        fastmath=True,
    )
    def _mae_grad_numba(y_true, y_pred):  # pragma: no cover
        n = y_true.shape[0]
        grad = np.empty_like(y_pred)
        for i in prange(n):
            diff = y_pred[i] - y_true[i]
            grad[i] = 1.0 if diff > 0 else -1.0 if diff < 0 else 0.0
        return grad / n

    # --- Huber Loss Implementations ---
    @jit(
        "float64(float32[:], float32[:], float32)",
        nopython=True,
        parallel=True,
        fastmath=True,
    )
    def _huber_numba(y_true, y_pred, delta):  # pragma: no cover
        n = y_true.shape[0]
        error = 0.0
        for i in prange(n):
            abs_err = abs(y_true[i] - y_pred[i])
            if abs_err <= delta:
                error += 0.5 * abs_err**2
            else:
                error += delta * (abs_err - 0.5 * delta)
        return error / n

    @jit(
        "float32[:](float32[:], float32[:], float32)",
        nopython=True,
        parallel=True,
        fastmath=True,
    )
    def _huber_grad_numba(y_true, y_pred, delta):  # pragma: no cover
        n = y_true.shape[0]
        grad = np.empty_like(y_pred)
        for i in prange(n):
            err = y_pred[i] - y_true[i]
            if abs(err) <= delta:
                grad[i] = err
            else:
                grad[i] = delta if err > 0 else -delta
        return grad / n

    # --- Quantile Loss Implementations ---
    @jit(
        "float64(float32[:], float32[:], float32)",
        nopython=True,
        parallel=True,
        fastmath=True,
    )
    def _quantile_numba(y_true, y_pred, quantile):  # pragma: no cover
        n = y_true.shape[0]
        error = 0.0
        for i in prange(n):
            err = y_true[i] - y_pred[i]
            if err > 0:
                error += quantile * err
            else:
                error += (quantile - 1.0) * err
        return error / n

    @jit(
        "float32[:](float32[:], float32[:], float32)",
        nopython=True,
        parallel=True,
        fastmath=True,
    )
    def _quantile_grad_numba(y_true, y_pred, quantile):  # pragma: no cover
        n = y_true.shape[0]
        grad = np.empty_like(y_pred)
        for i in prange(n):
            err = y_true[i] - y_pred[i]
            grad[i] = quantile if err <= 0 else quantile - 1.0
        return grad / n

    # --- Hinge Loss Implementations ---
    @jit("float64(float32[:], float32[:])", nopython=True, parallel=True, fastmath=True)
    def _hinge_numba(y_true, y_pred):  # pragma: no cover
        n = y_true.shape[0]
        loss = 0.0
        for i in prange(n):
            loss += max(0.0, 1.0 - y_true[i] * y_pred[i])
        return loss / n

    @jit(
        "float32[:](float32[:], float32[:])",
        nopython=True,
        parallel=True,
        fastmath=True,
    )
    def _hinge_grad_numba(y_true, y_pred):  # pragma: no cover
        n = y_true.shape[0]
        grad = np.empty_like(y_pred)
        for i in prange(n):
            grad[i] = -y_true[i] if 1.0 - y_true[i] * y_pred[i] > 0 else 0.0
        return grad / n

    # --- LogLoss / Cross-Entropy Implementations ---
    @jit("float64(float32[:], float32[:])", nopython=True, parallel=True, fastmath=True)
    def _logloss_numba(y_true, y_pred):  # pragma: no cover
        n = y_true.shape[0]
        loss = 0.0
        for i in prange(n):
            pred_clipped = max(EPSILON, min(1.0 - EPSILON, y_pred[i]))
            loss -= y_true[i] * np.log(pred_clipped) + (1.0 - y_true[i]) * np.log(
                1.0 - pred_clipped
            )
        return loss / n

    @jit(
        "float32[:](float32[:], float32[:])",
        nopython=True,
        parallel=True,
        fastmath=True,
    )
    def _logloss_grad_numba(y_true, y_pred):  # pragma: no cover
        n = y_true.shape[0]
        grad = np.empty_like(y_pred)
        for i in prange(n):
            pred_clipped = max(EPSILON, min(1.0 - EPSILON, y_pred[i]))
            grad[i] = (pred_clipped - y_true[i]) / (pred_clipped * (1.0 - pred_clipped))
        return grad / n

    # --- Focal Loss (Binary) ---
    @jit(
        "float64(float32[:], float32[:], float32, float32)",
        nopython=True,
        parallel=True,
        fastmath=True,
    )
    def _focal_loss_numba(y_true, y_pred, alpha, gamma):  # pragma: no cover
        n = y_true.shape[0]
        loss = 0.0
        for i in prange(n):
            # Clip predictions to avoid log(0)
            p = max(EPSILON, min(1.0 - EPSILON, y_pred[i]))
            pt = p if y_true[i] == 1 else 1 - p
            alpha_t = alpha if y_true[i] == 1 else 1 - alpha
            loss -= alpha_t * (1 - pt) ** gamma * np.log(pt)
        return loss / n

    @jit(
        "float32[:](float32[:], float32[:], float32, float32)",
        nopython=True,
        parallel=True,
        fastmath=True,
    )
    def _focal_loss_grad_numba(y_true, y_pred, alpha, gamma):  # pragma: no cover
        n = y_true.shape[0]
        grad = np.empty_like(y_pred)

        for i in prange(n):
            p = max(EPSILON, min(1.0 - EPSILON, y_pred[i]))

            # Gradient of focal loss with respect to p (prediction)
            # Standard Cross Entropy grad is (p - y), weighted by modulating factor (1-pt)^gamma
            # Plus extra term from derivative of modulating factor

            # Simplified for binary case p (sigmoid output):
            # dL/dp = alpha_t * (1-pt)^gamma * (gamma * pt * log(pt) - (1-pt)) / (pt * (1-pt)) * sign
            # This is complex, implementing the standard form derivative:
            # For y=1: -alpha * (1-p)^gamma * (1/p - gamma*log(p)/(1-p))
            # For y=0: (1-alpha) * p^gamma * (1/(1-p) - gamma*log(1-p)/p)

            if y_true[i] == 1:
                grad[i] = (
                    alpha * (1 - p) ** gamma * (gamma * p * np.log(p) + p - 1) / p
                    if p < 1.0
                    else 0.0
                )
            else:
                grad[i] = (
                    (1 - alpha)
                    * p**gamma
                    * (gamma * (1 - p) * np.log(1 - p) + p)
                    / (1 - p)
                    if p > 0.0
                    else 0.0
                )

        return grad / n

    # --- Contrastive Loss ---
    @jit(
        "float64(float32[:], float32[:], float32)",
        nopython=True,
        parallel=True,
        fastmath=True,
    )
    def _contrastive_loss_numba(y_true, dist_sq, margin):  # pragma: no cover
        # y_true: 1 for similar, 0 for dissimilar
        # dist_sq: squared euclidean distance between pairs
        n = y_true.shape[0]
        loss = 0.0

        for i in prange(n):
            dist = np.sqrt(dist_sq[i])
            if y_true[i] == 1:
                loss += dist_sq[i]
            else:
                loss += max(0.0, margin - dist) ** 2
        return 0.5 * loss / n

    @jit(
        "float32[:](float32[:], float32[:], float32, float32[:])",
        nopython=True,
        parallel=True,
        fastmath=True,
    )
    def _contrastive_loss_grad_numba(
        y_true, dist_sq, margin, diff_norm
    ):  # pragma: no cover
        # Returns gradient w.r.t distance (which is often intermediate)
        # Typically gradient is w.r.t embeddings, but here we return grad w.r.t distance for simplicity in this signature
        # dL/d(dist)
        n = y_true.shape[0]
        grad = np.empty_like(dist_sq)
        for i in prange(n):
            dist = np.sqrt(dist_sq[i]) + EPSILON
            if y_true[i] == 1:
                grad[i] = dist  # d(0.5*d^2)/dd = d
            else:
                if dist < margin:
                    grad[i] = -(margin - dist)  # d(0.5*(m-d)^2)/dd = -(m-d)
                else:
                    grad[i] = 0.0
        return grad / n

    @jit(nopython=True, cache=True, parallel=True, fastmath=True)
    def _cohens_d_numba(
        group1: np.ndarray, group2: np.ndarray
    ) -> float:  # pragma: no cover
        """
        Numba-optimized implementation of Cohen's d.

        Cohen's d (Effect Size) is a ruler that uses the standard deviation as its unit,
        quantifying the difference in a standard, relatable metric:
        how many standard deviations separate the two means.
        As it is not influenced by the sample size, we can rely on it for Big Data.
        The beauty of Cohen's d is its standardized nature.
        It tells you the magnitude of the difference regardless of the scale
        of your metric (seconds, dollars, clicks).

        Cohen's d measures the difference between the means in units of
        standard deviation (the normal variation).

        0.0–0.2 | Small effect (Trivial) | The difference is so small that you can barely taste it.
        0.2–0.5 | Medium Effect | You may notice the difference in sweetness, but it's not a drastic change.
        0.5–0.8 | Medium to Large Effect | The difference is clearly perceptible to most people.
        0.8+    | Large Effect | Brand B is much sweeter than Brand A; it's an obvious and substantial change.
        """
        # 1. Calculate Sample Sizes
        n1, n2 = len(group1), len(group2)

        # Safety check for insufficient data
        if n1 < 2 or n2 < 2:
            return 0.0

        # 2. Calculate Group Means
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)

        # 3. Calculate Biased-Corrected Standard Deviations (ddof=1)
        # Numba supports np.var with ddof
        std1_sq = np.var(group1) * (
            n1 / (n1 - 1)
        )  # Explicit ddof=1 adjustment for older Numba versions compatibility
        std2_sq = np.var(group2) * (n2 / (n2 - 1))

        # 4. Calculate Pooled Standard Deviation (the denominator)
        # This formula is for two groups with unequal sizes
        pooled_std = np.sqrt(((n1 - 1) * std1_sq + (n2 - 1) * std2_sq) / (n1 + n2 - 2))

        # 5. Calculate Cohen's d
        if pooled_std == 0:
            return 0.0  # Avoid division by zero for identical data

        d = (mean1 - mean2) / pooled_std
        return d

    @jit(nopython=True, cache=True)
    def _next_power_of_2(n: int) -> int:  # pragma: no cover
        """Helper to find the next power of 2 for Radix-2 FFT."""
        if n == 0:
            return 1
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        return n + 1

    @jit(nopython=True, cache=True)
    def _fft_radix2_numba(
        a: np.ndarray, invert: bool
    ) -> np.ndarray:  # pragma: no cover
        """
        Core Recursive Cooley-Tukey FFT implementation compatible with Numba.
        Requires input length to be a power of 2.
        """
        n = len(a)
        if n == 1:
            return a

        # Split into even and odd
        even = _fft_radix2_numba(a[0::2], invert)
        odd = _fft_radix2_numba(a[1::2], invert)

        combined = np.zeros(n, dtype=np.complex128)

        # Calculate angle
        # Forward: -2j * pi / n, Inverse: 2j * pi / n
        factor = -2j if not invert else 2j

        for k in range(n // 2):
            t = np.exp(factor * np.pi * k / n) * odd[k]
            combined[k] = even[k] + t
            combined[k + n // 2] = even[k] - t

        return combined

    @jit(nopython=True, cache=True)
    def _fft_numba(
        data: np.ndarray, inverse: bool = False
    ) -> np.ndarray:  # pragma: no cover
        """
        Numba-optimized Fast Fourier Transform using the Cooley-Tukey algorithm.

        The Fourier Transform decomposes a function (of time or signal)
        into its constituent frequencies.

        This implementation uses the Radix-2 algorithm.
        Note: If the input length is not a power of 2, it is zero-padded
        to the next power of 2 automatically.

        Parameters
        ----------
        data : np.ndarray
            Input array (1D). Can be real or complex.
        inverse : bool
            If True, performs Inverse FFT.

        Returns
        -------
        np.ndarray
            Complex array representing the frequency spectrum (or reconstructed signal).
        """
        n_orig = len(data)
        n_padded = _next_power_of_2(n_orig)

        # Prepare complex input buffer padded to power of 2
        input_buffer = np.zeros(n_padded, dtype=np.complex128)
        # Copy data (cast to complex)
        for i in range(n_orig):
            input_buffer[i] = data[i]

        # Perform FFT
        output = _fft_radix2_numba(input_buffer, inverse)

        # Normalize if inverse
        if inverse:
            output /= n_padded
            return output[:n_orig]
        # Truncate back to original length if it was padded
        # Note: Standard FFT libraries often keep the padded result or standard behavior.
        # Here, we return the full spectrum of the padded signal to maintain frequency bin integrity,
        # or slice it. For consistency with spectral analysis, usually, we analyze the padded signal.
        # However, to mimic numpy roughly, we return the calculated buffer.
        return output


# ==============================================================================
# CUPY IMPLEMENTATIONS (GPU accelerated)
# ==============================================================================


if CUPY_AVAILABLE:

    def _mse_cupy(y_true, y_pred):  # pragma: no cover
        return cp.mean((y_true - y_pred) ** 2)

    def _mse_grad_cupy(y_true, y_pred):  # pragma: no cover
        return 2.0 / y_true.shape[0] * (y_pred - y_true)

    def _mae_cupy(y_true, y_pred):  # pragma: no cover
        return cp.mean(cp.abs(y_true - y_pred))

    def _mae_grad_cupy(y_true, y_pred):  # pragma: no cover
        return cp.sign(y_pred - y_true) / y_true.shape[0]

    def _huber_cupy(y_true, y_pred, delta):  # pragma: no cover
        err = y_true - y_pred
        abs_err = cp.abs(err)
        return cp.mean(
            cp.where(
                abs_err <= delta, 0.5 * abs_err**2, delta * (abs_err - 0.5 * delta)
            )
        )

    def _huber_grad_cupy(y_true, y_pred, delta):  # pragma: no cover
        err = y_pred - y_true
        return (
            cp.where(cp.abs(err) <= delta, err, cp.sign(err) * delta) / y_true.shape[0]
        )

    def _quantile_cupy(y_true, y_pred, quantile):  # pragma: no cover
        err = y_true - y_pred
        return cp.mean(cp.maximum(quantile * err, (quantile - 1) * err))

    def _quantile_grad_cupy(y_true, y_pred, quantile):  # pragma: no cover
        err = y_true - y_pred
        return cp.where(err <= 0, quantile, quantile - 1.0) / y_true.shape[0]

    def _hinge_cupy(y_true, y_pred):  # pragma: no cover
        return cp.mean(cp.maximum(0.0, 1.0 - y_true * y_pred))

    def _hinge_grad_cupy(y_true, y_pred):  # pragma: no cover
        return cp.where(1.0 - y_true * y_pred > 0, -y_true, 0.0) / y_true.shape[0]

    def _logloss_cupy(y_true, y_pred):  # pragma: no cover
        pred_clipped = cp.clip(y_pred, EPSILON, 1.0 - EPSILON)
        return -cp.mean(
            y_true * cp.log(pred_clipped) + (1.0 - y_true) * cp.log(1.0 - pred_clipped)
        )

    def _logloss_grad_cupy(y_true, y_pred):  # pragma: no cover
        pred_clipped = cp.clip(y_pred, EPSILON, 1.0 - EPSILON)
        return (
            (pred_clipped - y_true)
            / (pred_clipped * (1.0 - pred_clipped))
            / y_true.shape[0]
        )

    def _focal_loss_cupy(y_true, y_pred, alpha, gamma):  # pragma: no cover
        p = cp.clip(y_pred, EPSILON, 1.0 - EPSILON)
        pt = cp.where(y_true == 1, p, 1 - p)
        alpha_t = cp.where(y_true == 1, alpha, 1 - alpha)
        loss = -alpha_t * (1 - pt) ** gamma * cp.log(pt)
        return cp.mean(loss)

    def _focal_loss_grad_cupy(y_true, y_pred, alpha, gamma):  # pragma: no cover
        p = cp.clip(y_pred, EPSILON, 1.0 - EPSILON)
        # Implementing derivative logic for y=1 and y=0
        term_pos = (
            -alpha * (1 - p) ** gamma * (1.0 / p + (gamma * p * cp.log(p)) / (1.0 - p))
        )
        term_neg = (
            (1 - alpha)
            * p**gamma
            * (1.0 / (1.0 - p) + (gamma * (1 - p) * cp.log(1 - p)) / p)
        )
        return cp.where(y_true == 1, term_pos, term_neg) / y_true.shape[0]

    def _contrastive_loss_cupy(y_true, dist_sq, margin):  # pragma: no cover
        dist = cp.sqrt(dist_sq)
        loss_sim = dist_sq
        loss_dissim = cp.maximum(0.0, margin - dist) ** 2
        return 0.5 * cp.mean(cp.where(y_true == 1, loss_sim, loss_dissim))

    def _contrastive_loss_grad_cupy(
        y_true, dist_sq, margin, diff_norm
    ):  # pragma: no cover
        dist = cp.sqrt(dist_sq) + EPSILON
        grad_sim = dist
        grad_dissim = -cp.maximum(0.0, margin - dist)  # if dist > margin, grad is 0
        return cp.where(y_true == 1, grad_sim, grad_dissim) / y_true.shape[0]

    # --- MMD (Maximum Mean Discrepancy) using Gaussian Kernel ---
    def _mmd_cupy(X, Y, gamma):  # pragma: no cover
        # X: (n, d), Y: (m, d)
        XX = cp.dot(X, X.T)
        XY = cp.dot(X, Y.T)
        YY = cp.dot(Y, Y.T)

        X_sqnorms = cp.diag(XX)
        Y_sqnorms = cp.diag(YY)

        K_XX = cp.exp(-gamma * (X_sqnorms[:, None] + X_sqnorms[None, :] - 2 * XX))
        K_XY = cp.exp(-gamma * (X_sqnorms[:, None] + Y_sqnorms[None, :] - 2 * XY))
        K_YY = cp.exp(-gamma * (Y_sqnorms[:, None] + Y_sqnorms[None, :] - 2 * YY))

        return cp.mean(K_XX) + cp.mean(K_YY) - 2 * cp.mean(K_XY)

    def _cohens_d_cupy(
        group1: cp.ndarray, group2: cp.ndarray
    ) -> float:  # pragma: no cover
        """
        CuPy (GPU) implementation of Cohen's d.

        Cohen's d (Effect Size) is a ruler that uses the standard deviation as its unit,
        quantifying the difference in a standard, relatable metric:
        how many standard deviations separate the two means.
        As it is not influenced by the sample size, we can rely on it for Big Data.
        The beauty of Cohen's d is its standardized nature.
        It tells you the magnitude of the difference regardless of the scale
        of your metric (seconds, dollars, clicks).

        Cohen's d measures the difference between the means in units of
        standard deviation (the normal variation).

        0.0–0.2 | Small effect (Trivial) | The difference is so small that you can barely taste it.
        0.2–0.5 | Medium Effect | You may notice the difference in sweetness, but it's not a drastic change.
        0.5–0.8 | Medium to Large Effect | The difference is clearly perceptible to most people.
        0.8+    | Large Effect | Brand B is much sweeter than Brand A; it's an obvious and substantial change.
        """
        # Ensure inputs are CuPy arrays
        if not isinstance(group1, cp.ndarray):
            group1 = cp.asarray(group1)
        if not isinstance(group2, cp.ndarray):
            group2 = cp.asarray(group2)

        # 1. Calculate Group Means
        mean1, mean2 = cp.mean(group1), cp.mean(group2)

        # 2. Calculate Sample Sizes
        n1, n2 = len(group1), len(group2)

        if n1 < 1e3 or n2 < 1e3:
            warnings.warn(
                f"Warning: Small sample sizes (n1={n1}, n2={n2}). "
                "Cohen's d may be unreliable. Consider: "
                "1) Using non-parametric tests (e.g., Mann-Whitney U), "
                "2) Reporting confidence intervals for effect size, "
                "3) Being cautious with interpretation"
            )

        # Safety check for insufficient data logic
        if n1 < 2 or n2 < 2:
            return 0.0

        # 3. Calculate Biased-Corrected Standard Deviations (ddof=1)
        std1_sq = cp.var(group1, ddof=1)  # Variance of group 1
        std2_sq = cp.var(group2, ddof=1)  # Variance of group 2

        # 4. Calculate Pooled Standard Deviation (the denominator)
        pooled_std = cp.sqrt(((n1 - 1) * std1_sq + (n2 - 1) * std2_sq) / (n1 + n2 - 2))

        # 5. Calculate Cohen's d
        if pooled_std == 0:
            return 0.0  # Avoid division by zero for identical data

        d = (mean1 - mean2) / pooled_std

        # Return as python float (move from GPU if necessary)
        return float(d)

    def _fft_cupy(
        data: cp.ndarray, inverse: bool = False
    ) -> cp.ndarray:  # pragma: no cover
        """
        CuPy (GPU) implementation of Fast Fourier Transform.
        Uses cuFFT via CuPy.

        Parameters
        ----------
        data : cp.ndarray
            Input array on GPU.
        inverse : bool
            If True, performs Inverse FFT.

        Returns
        -------
        cp.ndarray
            Complex array of frequency components.
        """
        # Ensure input is 1D for this implementation, or let CuPy handle broadcasting
        if not isinstance(data, cp.ndarray):
            data = cp.asarray(data)

        if inverse:
            return cp.fft.ifft(data)
        else:
            return cp.fft.fft(data)


# ==============================================================================
# NUMPY IMPLEMENTATIONS (baseline, always available)
# ==============================================================================


def _mse_numpy(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def _mse_grad_numpy(y_true, y_pred):
    return 2.0 / y_true.shape[0] * (y_pred - y_true)


def _mae_numpy(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def _mae_grad_numpy(y_true, y_pred):
    return np.sign(y_pred - y_true) / y_true.shape[0]


def _huber_numpy(y_true, y_pred, delta):
    err = y_true - y_pred
    abs_err = np.abs(err)
    return np.mean(
        np.where(abs_err <= delta, 0.5 * abs_err**2, delta * (abs_err - 0.5 * delta))
    )


def _huber_grad_numpy(y_true, y_pred, delta):
    err = y_pred - y_true
    return np.where(np.abs(err) <= delta, err, np.sign(err) * delta) / y_true.shape[0]


def _quantile_numpy(y_true, y_pred, quantile):
    err = y_true - y_pred
    return np.mean(np.maximum(quantile * err, (quantile - 1) * err))


def _quantile_grad_numpy(y_true, y_pred, quantile):
    err = y_true - y_pred
    return np.where(err <= 0, quantile, quantile - 1.0) / y_true.shape[0]


def _hinge_numpy(y_true, y_pred):
    return np.mean(np.maximum(0.0, 1.0 - y_true * y_pred))


def _hinge_grad_numpy(y_true, y_pred):
    return np.where(1.0 - y_true * y_pred > 0, -y_true, 0.0) / y_true.shape[0]


def _logloss_numpy(y_true, y_pred):
    pred_clipped = np.clip(y_pred, EPSILON, 1.0 - EPSILON)
    return -np.mean(
        y_true * np.log(pred_clipped) + (1.0 - y_true) * np.log(1.0 - pred_clipped)
    )


def _logloss_grad_numpy(y_true, y_pred):
    pred_clipped = np.clip(y_pred, EPSILON, 1.0 - EPSILON)
    return (
        (pred_clipped - y_true)
        / (pred_clipped * (1.0 - pred_clipped))
        / y_true.shape[0]
    )


def _focal_loss_numpy(y_true, y_pred, alpha, gamma):
    p = np.clip(y_pred, EPSILON, 1.0 - EPSILON)
    pt = np.where(y_true == 1, p, 1 - p)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    loss = -alpha_t * (1 - pt) ** gamma * np.log(pt)
    return np.mean(loss)


def _focal_loss_grad_numpy(y_true, y_pred, alpha, gamma):
    p = np.clip(y_pred, EPSILON, 1.0 - EPSILON)
    term_pos = (
        -alpha * (1 - p) ** gamma * (1.0 / p + (gamma * p * np.log(p)) / (1.0 - p))
    )
    term_neg = (
        (1 - alpha)
        * p**gamma
        * (1.0 / (1.0 - p) + (gamma * (1 - p) * np.log(1 - p)) / p)
    )
    return np.where(y_true == 1, term_pos, term_neg) / y_true.shape[0]


def _contrastive_loss_numpy(y_true, dist_sq, margin):
    dist = np.sqrt(dist_sq)
    loss_sim = dist_sq
    loss_dissim = np.maximum(0.0, margin - dist) ** 2
    return 0.5 * np.mean(np.where(y_true == 1, loss_sim, loss_dissim))


def _contrastive_loss_grad_numpy(y_true, dist_sq, margin, diff_norm):
    dist = np.sqrt(dist_sq) + EPSILON
    grad_sim = dist
    grad_dissim = -np.maximum(0.0, margin - dist)
    return np.where(y_true == 1, grad_sim, grad_dissim) / y_true.shape[0]


def _mmd_numpy(X, Y, gamma):
    XX = np.dot(X, X.T)
    XY = np.dot(X, Y.T)
    YY = np.dot(Y, Y.T)
    X_sqnorms = np.diag(XX)
    Y_sqnorms = np.diag(YY)
    K_XX = np.exp(-gamma * (X_sqnorms[:, None] + X_sqnorms[None, :] - 2 * XX))
    K_XY = np.exp(-gamma * (X_sqnorms[:, None] + Y_sqnorms[None, :] - 2 * XY))
    K_YY = np.exp(-gamma * (Y_sqnorms[:, None] + Y_sqnorms[None, :] - 2 * YY))
    return np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)


def _cohens_d_numpy(
    group1: Union[List[float], np.ndarray], group2: Union[List[float], np.ndarray]
) -> float:
    """
    Standard NumPy implementation of Cohen's d.

    Cohen's d (Effect Size) is a ruler that uses the standard deviation as its unit,
    quantifying the difference in a standard, relatable metric:
    how many standard deviations separate the two means.
    As it is not influenced by the sample size, we can rely on it for Big Data.
    The beauty of Cohen's d is its standardized nature.
    It tells you the magnitude of the difference regardless of the scale
    of your metric (seconds, dollars, clicks).

    Cohen's d measures the difference between the means in units of
    standard deviation (the normal variation).

    0.0–0.2 | Small effect (Trivial) | The difference is so small that you can barely taste it.
    0.2–0.5 | Medium Effect | You may notice the difference in sweetness, but it's not a drastic change.
    0.5–0.8 | Medium to Large Effect | The difference is clearly perceptible to most people.
    0.8+    | Large Effect | Brand B is much sweeter than Brand A; it's an obvious and substantial change.
    """
    # Convert lists to numpy arrays if needed
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    # 1. Calculate Group Means
    mean1, mean2 = np.mean(group1), np.mean(group2)

    # 2. Calculate Sample Sizes
    n1, n2 = len(group1), len(group2)

    if n1 < 1e3 or n2 < 1e3:
        warnings.warn(
            f"Warning: Small sample sizes (n1={n1}, n2={n2}). "
            "Cohen's d may be unreliable. Consider: "
            "1) Using non-parametric tests (e.g., Mann-Whitney U), "
            "2) Reporting confidence intervals for effect size, "
            "3) Being cautious with interpretation"
        )

    # Safety check for insufficient data to calculate variance
    if n1 < 2 or n2 < 2:
        return 0.0

    # 3. Calculate Biased-Corrected Standard Deviations (ddof=1)
    # For mathematical purposes, use `ddof` to adjust for bias.
    std1_sq = np.var(group1, ddof=1)  # Variance of group 1
    std2_sq = np.var(group2, ddof=1)  # Variance of group 2

    # 4. Calculate Pooled Standard Deviation (the denominator)
    # This formula is for two groups with unequal sizes
    pooled_std = np.sqrt(((n1 - 1) * std1_sq + (n2 - 1) * std2_sq) / (n1 + n2 - 2))

    # 5. Calculate Cohen's d
    if pooled_std == 0:
        return 0.0  # Avoid division by zero for identical data

    d = (mean1 - mean2) / pooled_std

    return float(d)


def _fft_numpy(
    data: Union[List[float], np.ndarray], inverse: bool = False
) -> np.ndarray:
    """
    Standard NumPy implementation of Fast Fourier Transform (FFT).

    The discrete Fourier transform (DFT) converts a finite sequence of
    equally-spaced samples of a function into a same-length sequence of
    equally-spaced samples of the discrete-time Fourier transform (DTFT),
    which is a complex-valued function of frequency.

    Parameters
    ----------
    data : array-like
        Input signal.
    inverse : bool
        If True, calculates the Inverse FFT (reconstructs signal from spectrum).

    Returns
    -------
    np.ndarray
        The complex frequency spectrum (or reconstructed signal).
    """
    data = np.asarray(data)

    if inverse:
        return np.fft.ifft(data)
    else:
        return np.fft.fft(data)


if not NUMBA_AVAILABLE:
    _mse_numba = _mse_numpy
    _mse_grad_numba = _mse_grad_numpy
    _mae_numba = _mae_numpy
    _mae_grad_numba = _mae_grad_numpy
    _huber_numba = _huber_numpy
    _huber_grad_numba = _huber_grad_numpy
    _quantile_numba = _quantile_numpy
    _quantile_grad_numba = _quantile_grad_numpy
    _hinge_numba = _hinge_numpy
    _hinge_grad_numba = _hinge_grad_numpy
    _logloss_numba = _logloss_numpy
    _logloss_grad_numba = _logloss_grad_numpy
    _focal_loss_numba = _focal_loss_numpy
    _focal_loss_grad_numba = _focal_loss_grad_numpy
    _contrastive_loss_numba = _contrastive_loss_numpy
    _contrastive_loss_grad_numba = _contrastive_loss_grad_numpy
    _cohens_d_numba = _cohens_d_numpy
    _fft_numba = _fft_numpy

if not CUPY_AVAILABLE:
    _mse_cupy = _mse_numpy
    _mse_grad_cupy = _mse_grad_numpy
    _mae_cupy = _mae_numpy
    _mae_grad_cupy = _mae_grad_numpy
    _huber_cupy = _huber_numpy
    _huber_grad_cupy = _huber_grad_numpy
    _quantile_cupy = _quantile_numpy
    _quantile_grad_cupy = _quantile_grad_numpy
    _hinge_cupy = _hinge_numpy
    _hinge_grad_cupy = _hinge_grad_numpy
    _logloss_cupy = _logloss_numpy
    _logloss_grad_cupy = _logloss_grad_numpy
    _focal_loss_cupy = _focal_loss_numpy
    _focal_loss_grad_cupy = _focal_loss_grad_numpy
    _contrastive_loss_cupy = _contrastive_loss_numpy
    _contrastive_loss_grad_cupy = _contrastive_loss_grad_numpy
    _mmd_cupy = _mmd_numpy
    _cohens_d_cupy = _cohens_d_numpy
    _fft_cupy = _fft_numpy

# ============================================================================
# PUBLIC METRICS CLASS
# ============================================================================


class Metrics:
    def __init__(self, gpu_threshold: int = 100_000):
        self.gpu_threshold = gpu_threshold
        self.numba_available = NUMBA_AVAILABLE
        self.gpu_available = CUPY_AVAILABLE
        self.history: Dict[str, List] = {}

        print(
            f"Metrics initialized. GPU Available: {self.gpu_available}, Numba Available: {self.numba_available}"
        )

    def start_monitoring(self):
        self.history = {}
        print("Metrics monitoring started.")

    def update(self, epoch: int, logs: Dict[str, float]):
        if "epoch" not in self.history:
            self.history["epoch"] = []
        self.history["epoch"].append(epoch)
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def get_history_df(self) -> Optional[pd.DataFrame]:
        if not self.history:
            return None
        return pd.DataFrame(self.history)

    def plot_history(self, metrics: Optional[List[str]] = None):
        """For visualizing the history of network learning: accuracy, loss
        in graphs you need to run this code after your training."""
        import matplotlib.pyplot as plt

        df = self.get_history_df()
        if df is None:
            print("No history to plot.")
            return
        if metrics is None:
            metrics = [col for col in df.columns if col != "epoch"]
        num_metrics = len(metrics)
        fig, axes = plt.subplots(
            num_metrics, 1, figsize=(10, num_metrics * 4), sharex=True
        )
        if num_metrics == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            if metric in df.columns:
                ax.plot(df["epoch"], df[metric], marker="o", linestyle="-")
                ax.set_ylabel(metric)
                ax.grid(True)
                ax.set_title(f"Training History: {metric}")
        axes[-1].set_xlabel("Epoch")
        plt.tight_layout()
        plt.show()

    def _dispatch(
        self, func_name: str, y_true: np.ndarray, y_pred: np.ndarray, force_cpu: bool
    ):
        backend_choice = "numpy"
        use_gpu = (
            self.gpu_available and not force_cpu and y_true.size >= self.gpu_threshold
        )
        if use_gpu:
            backend_choice = "cupy"
        elif self.numba_available:
            backend_choice = "numba"
        func = globals().get(f"_{func_name}_{backend_choice}") or globals().get(
            f"_{func_name}_numpy"
        )
        y_true_conv = y_true.astype(np.float32)
        y_pred_conv = y_pred.astype(np.float32)
        if use_gpu:
            y_true_conv = cp.asarray(y_true_conv)
            y_pred_conv = cp.asarray(y_pred_conv)
        return func, y_true_conv, y_pred_conv

    def _validate_inputs(self, y_true, y_pred):
        if y_true.shape[0] == 0:
            return 0.0, None
        if y_true.shape != y_pred.shape:
            raise ValueError("Input arrays must have the same shape.")
        return None, None

    def _execute(
        self,
        name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        return_grad: bool,
        force_cpu: bool,
        **kwargs,
    ):
        val, grad = self._validate_inputs(y_true, y_pred)
        if val is not None:
            return (val, grad) if return_grad else val
        loss_func, y_true_c, y_pred_c = self._dispatch(name, y_true, y_pred, force_cpu)
        loss = loss_func(y_true_c, y_pred_c, **kwargs)
        if return_grad:
            grad_func, _, _ = self._dispatch(f"{name}_grad", y_true, y_pred, force_cpu)
            grad = grad_func(y_true_c, y_pred_c, **kwargs)
            if CUPY_AVAILABLE and isinstance(grad, cp.ndarray):
                grad = cp.asnumpy(grad)
            return loss, grad
        return loss

    def mse(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        return_grad: bool = False,
        force_cpu: bool = False,
    ):
        """Mean Squared Error is the average of the squared differences
        between the predicted and actual values."""
        return self._execute("mse", y_true, y_pred, return_grad, force_cpu)

    def mae(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        return_grad: bool = False,
        force_cpu: bool = False,
    ):
        """Mean absolute error (MAE) is a measure of errors between
        paired observations expressing the same phenomenon."""
        return self._execute("mae", y_true, y_pred, return_grad, force_cpu)

    def rmse(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        return_grad: bool = False,
        force_cpu: bool = False,
    ):
        """RMSE (Root Mean Square Error) is a common metric in machine learning
        and statistics that measures the average magnitude of errors between
        predicted values and actual observed values, essentially showing
        how spread out the prediction errors (residuals) are from
        a model's prediction line or curve."""
        if return_grad:
            mse_val, mse_grad = self.mse(
                y_true, y_pred, return_grad=True, force_cpu=force_cpu
            )
            if mse_val == 0:
                return 0.0, mse_grad * 0.0
            rmse_val = np.sqrt(mse_val)
            rmse_grad = mse_grad / (2 * rmse_val)
            return rmse_val, rmse_grad
        return np.sqrt(self.mse(y_true, y_pred, return_grad=False, force_cpu=force_cpu))

    def huber_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        delta: float = 1.0,
        return_grad: bool = False,
        force_cpu: bool = False,
    ):
        """Huber loss is a loss function used in robust regression,
        that is less sensitive to outliers in data than the squared error loss."""
        return self._execute(
            "huber", y_true, y_pred, return_grad, force_cpu, delta=delta
        )

    def quantile_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        quantile: float = 0.5,
        return_grad: bool = False,
        force_cpu: bool = False,
    ):
        """Quantile loss, also known as pinball loss, is an asymmetric loss function
        used in machine learning (especially quantile regression) to predict
        specific quantiles (e.g., median, P90) of a target variable, penalizing over-
        and under-predictions differently based on the chosen quantile level (q)."""
        if not 0 < quantile < 1:
            raise ValueError("Quantile must be between 0 and 1.")
        return self._execute(
            "quantile", y_true, y_pred, return_grad, force_cpu, quantile=quantile
        )

    def hinge_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        return_grad: bool = False,
        force_cpu: bool = False,
    ):
        """Hinge loss, also known as max-margin loss, is a loss function
        that is particularly useful for training models in binary classification problems.
        """
        # Hinge loss expects labels to be -1 or 1
        return self._execute("hinge", y_true, y_pred, return_grad, force_cpu)

    def log_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        return_grad: bool = False,
        force_cpu: bool = False,
    ):
        """Log Loss is a logarithmic transformation of the likelihood function,
        primarily used to evaluate the performance of probabilistic classifiers."""
        # Log loss expects labels to be 0 or 1
        return self._execute("logloss", y_true, y_pred, return_grad, force_cpu)

    def cross_entropy_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        return_grad: bool = False,
        force_cpu: bool = False,
    ):
        """It measures the average number of bits required to identify an event
        from one probability distribution, p, using the optimal code for another
        probability distribution, q. In other words, cross-entropy measures
        the difference between the discovered probability distribution
        of a classification model and the predicted values."""
        # Alias for log_loss in the binary case
        return self.log_loss(y_true, y_pred, return_grad, force_cpu)

    def triplet_loss(
        self,
        anchor: np.ndarray,
        positive: np.ndarray,
        negative: np.ndarray,
        margin: float = 0.5,
        return_grad: bool = False,
        force_cpu: bool = False,
    ):
        """Triplet loss is a deep learning technique for learning useful data embeddings
        by training on triplets (anchor, positive, negative) to pull similar items closer
        and push dissimilar ones apart in a feature space, ensuring the anchor-positive distance
        is less than the anchor-negative distance by a set margin, crucial for tasks
        like face recognition and verification."""
        val, _ = self._validate_inputs(anchor, positive)
        if val is not None:
            return (val, (None, None, None)) if return_grad else val
        if anchor.shape != negative.shape:
            raise ValueError("Input arrays must have the same shape.")

        # Dispatching manually as the signature is different
        use_gpu = (
            self.gpu_available and not force_cpu and anchor.size >= self.gpu_threshold
        )

        # Convert to float32
        anchor_c = anchor.astype(np.float32)
        positive_c = positive.astype(np.float32)
        negative_c = negative.astype(np.float32)

        if use_gpu:
            anchor_c, positive_c, negative_c = (
                cp.asarray(anchor_c),
                cp.asarray(positive_c),
                cp.asarray(negative_c),
            )

        # Using L2 distance squared for simplicity and performance
        pos_dist = (
            cp.sum((anchor_c - positive_c) ** 2, axis=-1)
            if use_gpu
            else np.sum((anchor_c - positive_c) ** 2, axis=-1)
        )
        neg_dist = (
            cp.sum((anchor_c - negative_c) ** 2, axis=-1)
            if use_gpu
            else np.sum((anchor_c - negative_c) ** 2, axis=-1)
        )

        loss = np.maximum(0, pos_dist - neg_dist + margin)
        loss_val = cp.mean(loss) if use_gpu else np.mean(loss)

        if return_grad:
            mask = (loss > 0).astype(np.float32)
            if use_gpu:
                mask = cp.asarray(mask)

            grad_anchor = 2 * (negative_c - positive_c) * mask[..., np.newaxis]
            grad_positive = 2 * (positive_c - anchor_c) * mask[..., np.newaxis]
            grad_negative = 2 * (anchor_c - negative_c) * mask[..., np.newaxis]

            grad_anchor /= anchor.shape[0]
            grad_positive /= anchor.shape[0]
            grad_negative /= anchor.shape[0]

            if use_gpu:
                return loss_val, (
                    cp.asnumpy(grad_anchor),
                    cp.asnumpy(grad_positive),
                    cp.asnumpy(grad_negative),
                )
            return loss_val, (grad_anchor, grad_positive, grad_negative)

        return loss_val

    def focal_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        alpha: float = 0.25,
        gamma: float = 2.0,
        return_grad: bool = False,
        force_cpu: bool = False,
    ):
        """
        Calculates the Binary Focal Loss.
        Focuses training on hard examples.
        """
        return self._execute(
            "focal_loss",
            y_true,
            y_pred,
            return_grad,
            force_cpu,
            alpha=alpha,
            gamma=gamma,
        )

    def categorical_crossentropy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        from_logits: bool = False,
        force_cpu: bool = False,
    ) -> float:
        """
        Calculates Categorical Cross-Entropy (Softmax Loss).
        Expects one-hot encoded targets or integer labels.
        """
        # Note: This implementation currently supports NumPy/CuPy via direct calls
        # to avoid complex Numba implementation for multi-class softmax.

        use_gpu = (
            self.gpu_available and not force_cpu and y_true.size >= self.gpu_threshold
        )
        xp = cp if use_gpu else np

        y_true_c = y_true
        y_pred_c = y_pred

        if use_gpu:
            y_true_c = cp.asarray(y_true)
            y_pred_c = cp.asarray(y_pred)

        if from_logits:
            # Stable Softmax
            exps = xp.exp(y_pred_c - xp.max(y_pred_c, axis=-1, keepdims=True))
            y_pred_c = exps / xp.sum(exps, axis=-1, keepdims=True)

        y_pred_c = xp.clip(y_pred_c, EPSILON, 1.0 - EPSILON)

        # Support for both one-hot and integer labels
        if y_true_c.ndim == 1 or y_true_c.shape[1] == 1:  # Integer labels
            # Advanced indexing to pick probabilities of true classes
            # Creating one-hot logic on the fly efficiently
            n = y_pred_c.shape[0]
            # Flatten if needed
            if y_true_c.ndim > 1:
                y_true_c = y_true_c.flatten()
            log_p = -xp.log(y_pred_c[xp.arange(n), y_true_c.astype(int)])
            return float(xp.mean(log_p))
        else:  # One-hot
            return float(-xp.mean(xp.sum(y_true_c * xp.log(y_pred_c), axis=-1)))

    def contrastive_loss(
        self,
        y_true: np.ndarray,
        embedding_1: np.ndarray,
        embedding_2: np.ndarray,
        margin: float = 1.0,
        return_grad: bool = False,
        force_cpu: bool = False,
    ):
        """
        Calculates Contrastive Loss for pairs of embeddings.
        y_true: 1 for similar pairs, 0 for dissimilar.
        """
        val, _ = self._validate_inputs(embedding_1, embedding_2)

        if val is not None:
            if return_grad:
                return val, (None, None)
            else:
                return val

        use_gpu = (
            self.gpu_available
            and not force_cpu
            and embedding_1.size >= self.gpu_threshold
        )
        xp = cp if use_gpu else np

        e1 = embedding_1.astype(np.float32)
        e2 = embedding_2.astype(np.float32)
        y = y_true.astype(np.float32)

        if use_gpu:
            e1, e2, y = cp.asarray(e1), cp.asarray(e2), cp.asarray(y)

        # L2 Distance Squared
        diff = e1 - e2
        dist_sq = xp.sum(diff**2, axis=-1)

        # Dispatch to compiled kernels for loss calculation
        if use_gpu:
            loss = _contrastive_loss_cupy(y, dist_sq, margin)
        elif self.numba_available and not force_cpu:
            # Numba needs numpy arrays, conversion happens inside dispatch if we used it
            # But here we called manual dispatch.
            loss = _contrastive_loss_numba(y, dist_sq, margin)
        else:
            loss = _contrastive_loss_numpy(y, dist_sq, margin)

        if return_grad:
            # Calculate gradient w.r.t embeddings
            # dL/de1 = dL/d(dist_sq) * d(dist_sq)/de1
            # d(dist_sq)/de1 = 2 * (e1 - e2)

            if use_gpu:
                grad_dist = _contrastive_loss_grad_cupy(y, dist_sq, margin, None)
            elif self.numba_available and not force_cpu:
                grad_dist = _contrastive_loss_grad_numba(y, dist_sq, margin, dist_sq)
            else:
                grad_dist = _contrastive_loss_grad_numpy(y, dist_sq, margin, None)

            grad_e1 = grad_dist[:, np.newaxis] * 2 * diff
            grad_e2 = -grad_e1

            if use_gpu:
                return float(loss), (cp.asnumpy(grad_e1), cp.asnumpy(grad_e2))
            return float(loss), (grad_e1, grad_e2)

        return float(loss)

    def mmd_loss(
        self, X: np.ndarray, Y: np.ndarray, gamma: float = 1.0, force_cpu: bool = False
    ) -> float:
        """
        Calculates Maximum Mean Discrepancy (MMD) using a Gaussian kernel.
        Measures the distance between two distributions P (samples X) and Q (samples Y).
        """
        use_gpu = self.gpu_available and not force_cpu and X.size >= self.gpu_threshold

        X_c = X.astype(np.float32)
        Y_c = Y.astype(np.float32)

        if use_gpu:
            X_c = cp.asarray(X_c)
            Y_c = cp.asarray(Y_c)
            return float(_mmd_cupy(X_c, Y_c, gamma))

        return float(_mmd_numpy(X_c, Y_c, gamma))

    def wasserstein_approx(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Approximates Wasserstein distance (Earth Mover's Distance) for 1D distributions.
        For 1D, this is exactly the L1 distance between cumulative distribution functions (CDFs).
        """
        # This is a fast approximation valid for 1D arrays
        # Scipy's wasserstein_distance essentially sorts and takes L1 diff

        # Sort both arrays
        y_true_sorted = np.sort(y_true)
        y_pred_sorted = np.sort(y_pred)

        return float(np.mean(np.abs(y_true_sorted - y_pred_sorted)))

    def cohens_d(
        self,
        group1: Union[List[float], np.ndarray],
        group2: Union[List[float], np.ndarray],
        force_cpu: bool = False,
    ):
        """Cohen's d calculates the standardized difference between two group means,
        essentially measuring effect size by dividing the difference in means by
        the pooled standard deviation"""
        use_gpu = (
            self.gpu_available and not force_cpu and group1.size >= self.gpu_threshold
        )

        X_c = group1.astype(np.float32)
        Y_c = group2.astype(np.float32)

        if use_gpu:
            X_c = cp.asarray(X_c)
            Y_c = cp.asarray(Y_c)
            return float(_cohens_d_cupy(X_c, Y_c))

        return float(_cohens_d_numpy(X_c, Y_c))

    def fft(
        self,
        data: Union[List[float], np.ndarray],
        inverse: bool = False,
        force_cpu: bool = False,
        engine: str = "auto",
    ) -> Union[np.ndarray, List[complex]]:
        """
        Compute the Fast Fourier Transform (FFT) of a 1D signal.
        FFT with zero-padding to the next power of 2.
        """
        data_arr = np.asarray(data)
        if inverse or np.iscomplexobj(data_arr):
            target_dtype = np.complex128
        else:
            target_dtype = np.float64  # Use float64 for better precision than float32

        use_gpu = (
            self.gpu_available and not force_cpu and data_arr.size >= self.gpu_threshold
        )

        # Resolve 'auto' engine
        if engine == "auto":
            if use_gpu and CUPY_AVAILABLE:
                engine = "cupy"
            elif self.numba_available and not force_cpu:
                engine = "numba"
            else:
                engine = "numpy"

        # 3. Prepare Data & Execute
        if engine == "cupy" and CUPY_AVAILABLE:
            input_data = cp.asarray(data_arr, dtype=target_dtype)
            # Padding logic for CuPy if desired (usually CuPy FFT handles non-pow2, but for consistency):
            # n_orig = input_data.shape[0]
            # n_padded = 1 << (n_orig - 1).bit_length()
            # if n_orig != n_padded:
            #     input_data = cp.pad(input_data, (0, n_padded - n_orig), 'constant')

            spectrum = _fft_cupy(input_data, inverse=inverse)
            return spectrum

        elif engine == "numba" and self.numba_available:
            # Numba implementation handles padding internally or we do it here.
            # Our _fft_numba handles padding internally.
            input_data = np.asarray(data_arr, dtype=target_dtype)
            return _fft_numba(input_data, inverse=inverse)

        else:
            # NumPy / Fallback
            input_data = np.asarray(data_arr, dtype=target_dtype)
            # Standard NumPy FFT handles any size
            return _fft_numpy(input_data, inverse=inverse)

    def list_metrics(self) -> pd.DataFrame:
        """
        Returns a DataFrame listing all available metric functions, their
        descriptions, and usage signatures.

        Returns:
            pd.DataFrame: A table with columns 'Metric', 'Description', and 'Usage'.
        """
        import inspect

        methods_data = []

        for name, func in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith("_") or name == "list_metrics":
                continue

            # Skip update/start/plot/get_history helper methods
            if name in ["start_monitoring", "update", "get_history_df", "plot_history"]:
                continue

            doc = inspect.getdoc(func)
            description = doc.split("\n")[0] if doc else "No description available."
            try:
                signature = str(inspect.signature(func))
            except ValueError:
                signature = "(...)"

            methods_data.append(
                {
                    "Metric": name,
                    "Description": description,
                    "Usage": f"{name}{signature}",
                }
            )

        df = pd.DataFrame(methods_data)
        if not df.empty:
            df = df.sort_values(by="Metric").reset_index(drop=True)
        return df
