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
import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil

# --- Environment Checks for Optional Dependencies ---

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not installed. CPU parallel implementations will be unavailable.")

try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() > 0:
        CUPY_AVAILABLE = True
        cp.cuda.Device(0).use()
    else:
        CUPY_AVAILABLE = False
        warnings.warn("CuPy is installed, but no compatible CUDA GPU was found.")
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not installed. GPU acceleration will be unavailable.")


# ============================================================================
# PRIVATE IMPLEMENTATIONS (The actual computation engines)
# ============================================================================

# --- MSE Implementations ---
if NUMBA_AVAILABLE:
    @jit("float64(float32[:], float32[:])", nopython=True, parallel=True, fastmath=True)
    def _mse_numba(y_true, y_pred):
        n = y_true.shape[0]
        error = 0.0
        for i in prange(n):
            error += (y_true[i] - y_pred[i]) ** 2
        return error / n

    @jit("float32[:](float32[:], float32[:])", nopython=True, parallel=True, fastmath=True)
    def _mse_grad_numba(y_true, y_pred):
        n = y_true.shape[0]
        grad = np.empty_like(y_pred)
        for i in prange(n):
            grad[i] = 2.0 / n * (y_pred[i] - y_true[i])
        return grad

if CUPY_AVAILABLE:
    def _mse_cupy(y_true, y_pred): return cp.mean((y_true - y_pred) ** 2)
    def _mse_grad_cupy(y_true, y_pred): return 2.0 / y_true.shape[0] * (y_pred - y_true)

def _mse_numpy(y_true, y_pred): return np.mean((y_true - y_pred) ** 2)
def _mse_grad_numpy(y_true, y_pred): return 2.0 / y_true.shape[0] * (y_pred - y_true)


# --- MAE Implementations ---
if NUMBA_AVAILABLE:
    @jit("float64(float32[:], float32[:])", nopython=True, parallel=True, fastmath=True)
    def _mae_numba(y_true, y_pred):
        n = y_true.shape[0]
        error = 0.0
        for i in prange(n):
            error += abs(y_true[i] - y_pred[i])
        return error / n

    @jit("float32[:](float32[:], float32[:])", nopython=True, parallel=True, fastmath=True)
    def _mae_grad_numba(y_true, y_pred):
        n = y_true.shape[0]
        grad = np.empty_like(y_pred)
        for i in prange(n):
            diff = y_pred[i] - y_true[i]
            grad[i] = 1.0 if diff > 0 else -1.0 if diff < 0 else 0.0
        return grad / n

if CUPY_AVAILABLE:
    def _mae_cupy(y_true, y_pred): return cp.mean(cp.abs(y_true - y_pred))
    def _mae_grad_cupy(y_true, y_pred): return cp.sign(y_pred - y_true) / y_true.shape[0]

def _mae_numpy(y_true, y_pred): return np.mean(np.abs(y_true - y_pred))
def _mae_grad_numpy(y_true, y_pred): return np.sign(y_pred - y_true) / y_true.shape[0]


# --- Huber Loss Implementations ---
if NUMBA_AVAILABLE:
    @jit("float64(float32[:], float32[:], float32)", nopython=True, parallel=True, fastmath=True)
    def _huber_numba(y_true, y_pred, delta):
        n = y_true.shape[0]
        error = 0.0
        for i in prange(n):
            abs_err = abs(y_true[i] - y_pred[i])
            if abs_err <= delta:
                error += 0.5 * abs_err**2
            else:
                error += delta * (abs_err - 0.5 * delta)
        return error / n

    @jit("float32[:](float32[:], float32[:], float32)", nopython=True, parallel=True, fastmath=True)
    def _huber_grad_numba(y_true, y_pred, delta):
        n = y_true.shape[0]
        grad = np.empty_like(y_pred)
        for i in prange(n):
            err = y_pred[i] - y_true[i]
            if abs(err) <= delta:
                grad[i] = err
            else:
                grad[i] = delta if err > 0 else -delta
        return grad / n

if CUPY_AVAILABLE:
    def _huber_cupy(y_true, y_pred, delta):
        err = y_true - y_pred
        abs_err = cp.abs(err)
        return cp.mean(cp.where(abs_err <= delta, 0.5 * abs_err**2, delta * (abs_err - 0.5 * delta)))
    def _huber_grad_cupy(y_true, y_pred, delta):
        err = y_pred - y_true
        return cp.where(cp.abs(err) <= delta, err, cp.sign(err) * delta) / y_true.shape[0]

def _huber_numpy(y_true, y_pred, delta):
    err = y_true - y_pred
    abs_err = np.abs(err)
    return np.mean(np.where(abs_err <= delta, 0.5 * abs_err**2, delta * (abs_err - 0.5 * delta)))
def _huber_grad_numpy(y_true, y_pred, delta):
    err = y_pred - y_true
    return np.where(np.abs(err) <= delta, err, np.sign(err) * delta) / y_true.shape[0]


# --- Quantile Loss Implementations ---
if NUMBA_AVAILABLE:
    @jit("float64(float32[:], float32[:], float32)", nopython=True, parallel=True, fastmath=True)
    def _quantile_numba(y_true, y_pred, quantile):
        n = y_true.shape[0]
        error = 0.0
        for i in prange(n):
            err = y_true[i] - y_pred[i]
            if err > 0:
                error += quantile * err
            else:
                error += (quantile - 1.0) * err
        return error / n
    @jit("float32[:](float32[:], float32[:], float32)", nopython=True, parallel=True, fastmath=True)
    def _quantile_grad_numba(y_true, y_pred, quantile):
        n = y_true.shape[0]
        grad = np.empty_like(y_pred)
        for i in prange(n):
            err = y_true[i] - y_pred[i]
            grad[i] = quantile if err <= 0 else quantile - 1.0
        return grad / n

if CUPY_AVAILABLE:
    def _quantile_cupy(y_true, y_pred, quantile):
        err = y_true - y_pred
        return cp.mean(cp.maximum(quantile * err, (quantile - 1) * err))
    def _quantile_grad_cupy(y_true, y_pred, quantile):
        err = y_true - y_pred
        return cp.where(err <= 0, quantile, quantile - 1.0) / y_true.shape[0]

def _quantile_numpy(y_true, y_pred, quantile):
    err = y_true - y_pred
    return np.mean(np.maximum(quantile * err, (quantile - 1) * err))
def _quantile_grad_numpy(y_true, y_pred, quantile):
    err = y_true - y_pred
    return np.where(err <= 0, quantile, quantile - 1.0) / y_true.shape[0]


# --- Hinge Loss Implementations ---
if NUMBA_AVAILABLE:
    @jit("float64(float32[:], float32[:])", nopython=True, parallel=True, fastmath=True)
    def _hinge_numba(y_true, y_pred):
        n = y_true.shape[0]
        loss = 0.0
        for i in prange(n):
            loss += max(0.0, 1.0 - y_true[i] * y_pred[i])
        return loss / n
    @jit("float32[:](float32[:], float32[:])", nopython=True, parallel=True, fastmath=True)
    def _hinge_grad_numba(y_true, y_pred):
        n = y_true.shape[0]
        grad = np.empty_like(y_pred)
        for i in prange(n):
            grad[i] = -y_true[i] if 1.0 - y_true[i] * y_pred[i] > 0 else 0.0
        return grad / n

if CUPY_AVAILABLE:
    def _hinge_cupy(y_true, y_pred): return cp.mean(cp.maximum(0.0, 1.0 - y_true * y_pred))
    def _hinge_grad_cupy(y_true, y_pred): return cp.where(1.0 - y_true * y_pred > 0, -y_true, 0.0) / y_true.shape[0]

def _hinge_numpy(y_true, y_pred): return np.mean(np.maximum(0.0, 1.0 - y_true * y_pred))
def _hinge_grad_numpy(y_true, y_pred): return np.where(1.0 - y_true * y_pred > 0, -y_true, 0.0) / y_true.shape[0]


# --- LogLoss / Cross-Entropy Implementations ---
EPSILON = 1e-15 # for numerical stability
if NUMBA_AVAILABLE:
    @jit("float64(float32[:], float32[:])", nopython=True, parallel=True, fastmath=True)
    def _logloss_numba(y_true, y_pred):
        n = y_true.shape[0]
        loss = 0.0
        for i in prange(n):
            pred_clipped = max(EPSILON, min(1.0 - EPSILON, y_pred[i]))
            loss -= y_true[i] * np.log(pred_clipped) + (1.0 - y_true[i]) * np.log(1.0 - pred_clipped)
        return loss / n
    @jit("float32[:](float32[:], float32[:])", nopython=True, parallel=True, fastmath=True)
    def _logloss_grad_numba(y_true, y_pred):
        n = y_true.shape[0]
        grad = np.empty_like(y_pred)
        for i in prange(n):
            pred_clipped = max(EPSILON, min(1.0 - EPSILON, y_pred[i]))
            grad[i] = (pred_clipped - y_true[i]) / (pred_clipped * (1.0 - pred_clipped))
        return grad / n

if CUPY_AVAILABLE:
    def _logloss_cupy(y_true, y_pred):
        pred_clipped = cp.clip(y_pred, EPSILON, 1.0 - EPSILON)
        return -cp.mean(y_true * cp.log(pred_clipped) + (1.0 - y_true) * cp.log(1.0 - pred_clipped))
    def _logloss_grad_cupy(y_true, y_pred):
        pred_clipped = cp.clip(y_pred, EPSILON, 1.0 - EPSILON)
        return (pred_clipped - y_true) / (pred_clipped * (1.0 - pred_clipped)) / y_true.shape[0]

def _logloss_numpy(y_true, y_pred):
    pred_clipped = np.clip(y_pred, EPSILON, 1.0 - EPSILON)
    return -np.mean(y_true * np.log(pred_clipped) + (1.0 - y_true) * np.log(1.0 - pred_clipped))
def _logloss_grad_numpy(y_true, y_pred):
    pred_clipped = np.clip(y_pred, EPSILON, 1.0 - EPSILON)
    return (pred_clipped - y_true) / (pred_clipped * (1.0 - pred_clipped)) / y_true.shape[0]


# ============================================================================
# PUBLIC METRICS CLASS
# ============================================================================

class Metrics:
    def __init__(self, gpu_threshold: int = 100_000):
        self.gpu_threshold = gpu_threshold
        self.numba_available = NUMBA_AVAILABLE
        self.gpu_available = CUPY_AVAILABLE
        self.history: Dict[str, List] = {}
        if self.numba_available:
            try:
                num_threads = psutil.cpu_count(logical=True)
                os.environ['NUMBA_NUM_THREADS'] = str(num_threads)
            except Exception as e:
                warnings.warn(f"Could not set Numba threads with psutil: {e}")
        print(f"Metrics initialized. GPU Available: {self.gpu_available}, Numba Available: {self.numba_available}")

    def start_monitoring(self):
        self.history = {}
        print("Metrics monitoring started.")

    def update(self, epoch: int, logs: Dict[str, float]):
        if "epoch" not in self.history: self.history["epoch"] = []
        self.history["epoch"].append(epoch)
        for key, value in logs.items():
            if key not in self.history: self.history[key] = []
            self.history[key].append(value)

    def get_history_df(self) -> Optional[pd.DataFrame]:
        if not self.history: return None
        return pd.DataFrame(self.history)

    def plot_history(self, metrics: Optional[List[str]] = None):
        import matplotlib.pyplot as plt
        df = self.get_history_df()
        if df is None:
            print("No history to plot.")
            return
        if metrics is None: metrics = [col for col in df.columns if col != "epoch"]
        num_metrics = len(metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, num_metrics * 4), sharex=True)
        if num_metrics == 1: axes = [axes]
        for ax, metric in zip(axes, metrics):
            if metric in df.columns:
                ax.plot(df["epoch"], df[metric], marker='o', linestyle='-')
                ax.set_ylabel(metric)
                ax.grid(True)
                ax.set_title(f"Training History: {metric}")
        axes[-1].set_xlabel("Epoch")
        plt.tight_layout()
        plt.show()

    def _dispatch(self, func_name: str, y_true: np.ndarray, y_pred: np.ndarray, force_cpu: bool):
        backend_choice = "numpy"
        use_gpu = self.gpu_available and not force_cpu and y_true.size >= self.gpu_threshold
        if use_gpu: backend_choice = "cupy"
        elif self.numba_available: backend_choice = "numba"
        func = globals().get(f"_{func_name}_{backend_choice}") or globals().get(f"_{func_name}_numpy")
        y_true_conv = y_true.astype(np.float32)
        y_pred_conv = y_pred.astype(np.float32)
        if use_gpu:
            y_true_conv = cp.asarray(y_true_conv)
            y_pred_conv = cp.asarray(y_pred_conv)
        return func, y_true_conv, y_pred_conv

    def _validate_inputs(self, y_true, y_pred):
        if y_true.shape[0] == 0: return 0.0, None
        if y_true.shape != y_pred.shape: raise ValueError("Input arrays must have the same shape.")
        return None, None

    def _execute(self, name: str, y_true: np.ndarray, y_pred: np.ndarray, return_grad: bool, force_cpu: bool, **kwargs):
        val, grad = self._validate_inputs(y_true, y_pred)
        if val is not None: return (val, grad) if return_grad else val
        loss_func, y_true_c, y_pred_c = self._dispatch(name, y_true, y_pred, force_cpu)
        loss = loss_func(y_true_c, y_pred_c, **kwargs)
        if return_grad:
            grad_func, _, _ = self._dispatch(f"{name}_grad", y_true, y_pred, force_cpu)
            grad = grad_func(y_true_c, y_pred_c, **kwargs)
            if CUPY_AVAILABLE and isinstance(grad, cp.ndarray): grad = cp.asnumpy(grad)
            return loss, grad
        return loss

    def mse(self, y_true: np.ndarray, y_pred: np.ndarray, return_grad: bool = False, force_cpu: bool = False):
        return self._execute("mse", y_true, y_pred, return_grad, force_cpu)
    
    def mae(self, y_true: np.ndarray, y_pred: np.ndarray, return_grad: bool = False, force_cpu: bool = False):
        return self._execute("mae", y_true, y_pred, return_grad, force_cpu)

    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray, return_grad: bool = False, force_cpu: bool = False):
        if return_grad:
            mse_val, mse_grad = self.mse(y_true, y_pred, return_grad=True, force_cpu=force_cpu)
            if mse_val == 0: return 0.0, mse_grad * 0.0
            rmse_val = np.sqrt(mse_val)
            rmse_grad = mse_grad / (2 * rmse_val)
            return rmse_val, rmse_grad
        return np.sqrt(self.mse(y_true, y_pred, return_grad=False, force_cpu=force_cpu))

    def huber_loss(self, y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0, return_grad: bool = False, force_cpu: bool = False):
        return self._execute("huber", y_true, y_pred, return_grad, force_cpu, delta=delta)

    def quantile_loss(self, y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.5, return_grad: bool = False, force_cpu: bool = False):
        if not 0 < quantile < 1: raise ValueError("Quantile must be between 0 and 1.")
        return self._execute("quantile", y_true, y_pred, return_grad, force_cpu, quantile=quantile)

    def hinge_loss(self, y_true: np.ndarray, y_pred: np.ndarray, return_grad: bool = False, force_cpu: bool = False):
        # Hinge loss expects labels to be -1 or 1
        return self._execute("hinge", y_true, y_pred, return_grad, force_cpu)

    def log_loss(self, y_true: np.ndarray, y_pred: np.ndarray, return_grad: bool = False, force_cpu: bool = False):
        # Log loss expects labels to be 0 or 1
        return self._execute("logloss", y_true, y_pred, return_grad, force_cpu)

    def cross_entropy_loss(self, y_true: np.ndarray, y_pred: np.ndarray, return_grad: bool = False, force_cpu: bool = False):
        # Alias for log_loss in the binary case
        return self.log_loss(y_true, y_pred, return_grad, force_cpu)
    
    def triplet_loss(self, anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray, margin: float = 0.5, return_grad: bool = False, force_cpu: bool = False):
        val, _ = self._validate_inputs(anchor, positive)
        if val is not None: return (val, (None, None, None)) if return_grad else val
        if anchor.shape != negative.shape: raise ValueError("Input arrays must have the same shape.")
        
        # Dispatching manually as the signature is different
        backend_choice = "numpy"
        use_gpu = self.gpu_available and not force_cpu and anchor.size >= self.gpu_threshold
        if use_gpu: backend_choice = "cupy"
        elif self.numba_available: backend_choice = "numba"

        # Convert to float32
        anchor_c = anchor.astype(np.float32)
        positive_c = positive.astype(np.float32)
        negative_c = negative.astype(np.float32)

        if use_gpu:
            anchor_c, positive_c, negative_c = cp.asarray(anchor_c), cp.asarray(positive_c), cp.asarray(negative_c)

        # Using L2 distance squared for simplicity and performance
        pos_dist = cp.sum((anchor_c - positive_c)**2, axis=-1) if use_gpu else np.sum((anchor_c - positive_c)**2, axis=-1)
        neg_dist = cp.sum((anchor_c - negative_c)**2, axis=-1) if use_gpu else np.sum((anchor_c - negative_c)**2, axis=-1)

        loss = np.maximum(0, pos_dist - neg_dist + margin)
        loss_val = cp.mean(loss) if use_gpu else np.mean(loss)

        if return_grad:
            mask = (loss > 0).astype(np.float32)
            if use_gpu: mask = cp.asarray(mask)

            grad_anchor = 2 * (negative_c - positive_c) * mask[..., np.newaxis]
            grad_positive = 2 * (positive_c - anchor_c) * mask[..., np.newaxis]
            grad_negative = 2 * (anchor_c - negative_c) * mask[..., np.newaxis]
            
            grad_anchor /= anchor.shape[0]
            grad_positive /= anchor.shape[0]
            grad_negative /= anchor.shape[0]

            if use_gpu:
                return loss_val, (cp.asnumpy(grad_anchor), cp.asnumpy(grad_positive), cp.asnumpy(grad_negative))
            return loss_val, (grad_anchor, grad_positive, grad_negative)

        return loss_val
