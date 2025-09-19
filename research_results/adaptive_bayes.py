# adaptive_bayes.py
import math

import numpy as np

# Optional GPU
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False

from numba import njit, prange


# ---------------------------
# Numba CPU kernels
# ---------------------------
@njit(cache=True, fastmath=True)
def _log1p_clip(x):
    return math.log1p(x)


@njit(cache=True, fastmath=True)
def _sigmoid(z):
    # Stable logistic
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


@njit(parallel=True, cache=True, fastmath=True)
def _score_batch_numba(X, w, eps):
    n, d = X.shape
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        s = 0.0
        for j in range(d):
            x = X[i, j]
            if abs(x) > eps:
                s += w[j] * _log1p_clip(x)
        out[i] = s
    return out


@njit(parallel=True, cache=True, fastmath=True)
def _update_batch_numba(X, y, w, base_lr, eps):
    n, d = X.shape
    for i in prange(n):
        # compute prob
        s = 0.0
        for j in range(d):
            x = X[i, j]
            if abs(x) > eps:
                s += w[j] * _log1p_clip(x)
        p = _sigmoid(s)
        err = y[i] - p
        lr = base_lr * abs(err) * (1.0 - abs(p - 0.5))
        # update
        for j in range(d):
            x = X[i, j]
            if abs(x) > eps:
                w[j] += lr * err * x
    # in-place update of w


class AdaptiveBayes:
    def __init__(self, base_lr=1e-2, eps=1e-10, device=None):
        """
        base_lr: initial base learning rate
        eps: threshold for selective feature updates (|x| > eps)
        device: 'cpu' or 'gpu' (auto if None)
        """
        self.base_lr = float(base_lr)
        self.eps = float(eps)
        self.device = device
        self._is_gpu = False
        self.w = None
        self._xp = np  # default
        self._backend = "cpu"

        if device is None:
            # auto
            if CUPY_AVAILABLE:
                self._is_gpu = True
                self._xp = cp
                self._backend = "gpu"
        elif device == "gpu" and CUPY_AVAILABLE:
            self._is_gpu = True
            self._xp = cp
            self._backend = "gpu"
        else:
            self._is_gpu = False
            self._xp = np
            self._backend = "cpu"

    def _ensure_weights(self, d):
        if self.w is None:
            if self._is_gpu:
                self.w = self._xp.zeros(d, dtype=self._xp.float64)
            else:
                self.w = np.zeros(d, dtype=np.float64)
        else:
            if (self._is_gpu and self.w.shape[0] != d) or (
                (not self._is_gpu) and self.w.shape[0] != d
            ):
                if self._is_gpu:
                    self.w = self._xp.zeros(d, dtype=self._xp.float64)
                else:
                    self.w = np.zeros(d, dtype=np.float64)

    def _score_cpu(self, X):
        return _score_batch_numba(X, self.w, self.eps)

    def _update_cpu(self, X, y):
        _update_batch_numba(X, y, self.w, self.base_lr, self.eps)

    def _score_gpu(self, X):
        # X, w on GPU
        # s = (w * log1p(X_masked)).sum(axis=1)
        X_mask = self._xp.abs(X) > self.eps
        X_log = self._xp.where(X_mask, self._xp.log1p(X), 0.0)
        return X_log.dot(self.w)

    def _update_gpu(self, X, y):
        # vectorized per-batch update on GPU
        X_mask = self._xp.abs(X) > self.eps
        X_log = self._xp.where(X_mask, self._xp.log1p(X), 0.0)
        s = X_log.dot(self.w)
        p = 1.0 / (1.0 + self._xp.exp(-s))
        err = y - p
        lr = self.base_lr * self._xp.abs(err) * (1.0 - self._xp.abs(p - 0.5))
        # Broadcasted update: w += sum_i (lr_i * err_i) * x_i
        # Using raw X for update term as per rule; mask to apply selective updates
        coeff = (lr * err)[:, None]
        delta = (coeff * self._xp.where(X_mask, X, 0.0)).sum(axis=0)
        self.w += delta

    def fit(self, X, y, epochs=1, batch_size=65536, shuffle=True):
        """
        X: numpy or cupy array (if device='gpu', cupy preferred)
        y: 0/1 labels
        """
        xp = self._xp
        if not self._is_gpu and isinstance(X, np.ndarray) is False:
            X = np.asarray(X, dtype=np.float64)
        if self._is_gpu and isinstance(X, cp.ndarray) is False:
            X = cp.asarray(X, dtype=cp.float64)

        n, d = X.shape
        self._ensure_weights(d)

        if not self._is_gpu:
            # Move weights to CPU numpy if needed
            if not isinstance(self.w, np.ndarray):
                self.w = np.asarray(self.w.get(), dtype=np.float64)

        for _ in range(epochs):
            idx = xp.arange(n)
            if shuffle:
                if self._is_gpu:
                    rp = cp.random.permutation(n)
                else:
                    rp = np.random.permutation(n)
                idx = rp
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                bidx = idx[start:end]
                Xb = X[bidx]
                yb = y[bidx]
                if self._is_gpu:
                    self._update_gpu(Xb, yb)
                else:
                    # ensure CPU contiguous arrays for Numba
                    Xb_cpu = np.ascontiguousarray(Xb)
                    yb_cpu = np.ascontiguousarray(yb)
                    self._update_cpu(Xb_cpu, yb_cpu)
        return self

    def predict_proba(self, X, batch_size=262144):
        xp = self._xp
        if self._is_gpu and not isinstance(X, cp.ndarray):
            X = cp.asarray(X, dtype=cp.float64)
        if (not self._is_gpu) and not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float64)

        n = X.shape[0]
        probs = []
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            Xb = X[start:end]
            if self._is_gpu:
                s = self._score_gpu(Xb)
                pb = 1.0 / (1.0 + self._xp.exp(-s))
                probs.append(pb)
            else:
                s = self._score_cpu(np.ascontiguousarray(Xb))
                pb = 1.0 / (1.0 + np.exp(-s))
                probs.append(pb)
        out = xp.concatenate(probs, axis=0)
        if self._is_gpu:
            return cp.asnumpy(out)
        return out

    def decision_function(self, X, batch_size=262144):
        xp = self._xp
        if self._is_gpu and not isinstance(X, cp.ndarray):
            X = cp.asarray(X, dtype=cp.float64)
        if (not self._is_gpu) and not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float64)

        n = X.shape[0]
        scores = []
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            Xb = X[start:end]
            if self._is_gpu:
                s = self._score_gpu(Xb)
                scores.append(s)
            else:
                s = self._score_cpu(np.ascontiguousarray(Xb))
                scores.append(s)
        out = xp.concatenate(scores, axis=0)
        if self._is_gpu:
            return cp.asnumpy(out)
        return out

    def predict(self, X, threshold=0.5):
        p = self.predict_proba(X)
        return (p >= threshold).astype(np.int32)
