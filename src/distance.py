# src/ds_tools/distance.py
"""
This module provides a collection of high-performance, system-aware distance
and similarity metric implementations, with support for CPU parallelization
and GPU acceleration. All implementations are self-contained where feasible.
"""

import inspect
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd


def _detect_optional_dependencies():
    deps = {
        "numba_available": False,
        "cupy_available": False,
        "cp": None,
        "jit": None,
        "prange": None,
    }

    # Numba detection
    try:
        from numba import jit, prange

        deps.update({"numba_available": True, "jit": jit, "prange": prange})
    except ImportError:
        warnings.warn("Numba not installed...")

    # CuPy detection
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() > 0:
            cupy.cuda.Device(0).use()
            deps.update({"cupy_available": True, "cp": cupy})
        else:
            warnings.warn("CuPy is installed, but no compatible CUDA GPU found.")
    except ImportError:
        warnings.warn("CuPy not installed...")
    except Exception as e:  # Catch any CuPy error (like driver issues)
        warnings.warn(f"CuPy installed but failed to initialize: {e}")

    return deps


_DEPS = _detect_optional_dependencies()
NUMBA_AVAILABLE = _DEPS["numba_available"]
CUPY_AVAILABLE = _DEPS["cupy_available"]
cp = _DEPS["cp"]  # None, if CuPy unavailable
jit = _DEPS["jit"]  # None, if Numba unavailable
prange = _DEPS["prange"]  # None, if Numba unavailable

# ============================================================================
# NUMBA IMPLEMENTATIONS (JIT-compiled, CPU optimized)
# ============================================================================


if NUMBA_AVAILABLE:
    # --- Vector-to-Vector Distances ---
    @jit("float64(float32[:], float32[:])", nopython=True, fastmath=True)
    def _euclidean_numba(u, v):  # Euclidean / L2
        dist_sq = 0.0
        for i in range(u.shape[0]):
            dist_sq += (u[i] - v[i]) ** 2
        return np.sqrt(dist_sq)

    @jit("float64(float32[:], float32[:])", nopython=True, fastmath=True)
    def _manhattan_numba(u, v):  # Manhattan / L1
        dist = 0.0
        for i in range(u.shape[0]):
            dist += abs(u[i] - v[i])
        return dist

    @jit("float64(float32[:], float32[:], int_)", nopython=True, fastmath=True)
    def _minkowski_numba(u, v, p):  # Minkowski
        dist = 0.0
        for i in range(u.shape[0]):
            dist += abs(u[i] - v[i]) ** p
        return dist ** (1.0 / p)

    @jit("float64(float32[:], float32[:])", nopython=True, fastmath=True)
    def _chebyshev_numba(u, v):  # Chebyshev / L-infinity
        max_dist = 0.0
        for i in range(u.shape[0]):
            dist = abs(u[i] - v[i])
            if dist > max_dist:
                max_dist = dist
        return max_dist

    @jit("float64(float32[:], float32[:])", nopython=True, fastmath=True)
    def _cosine_similarity_numba(u, v):  # Cosine Similarity
        dot, norm_u, norm_v = 0.0, 0.0, 0.0
        for i in range(u.shape[0]):
            dot += u[i] * v[i]
            norm_u += u[i] ** 2
            norm_v += v[i] ** 2
        norm_u, norm_v = np.sqrt(norm_u), np.sqrt(norm_v)
        if norm_u == 0.0 and norm_v == 0.0:
            return 1.0  # similarity of two zero vectors = 1
        if norm_u == 0.0 or norm_v == 0.0:
            return 0.0  # similarity with one zero vectors = 0
        return dot / (norm_u * norm_v)

    @jit("float64(float32[:], float32[:], float32[:,:])", nopython=True, fastmath=True)
    def _mahalanobis_numba(u, v, VI):  # Mahalanobis
        diff = u - v
        # Perform (u-v)^T @ VI @ (u-v)
        tmp = np.empty(VI.shape[0], dtype=np.float32)
        for i in range(VI.shape[0]):
            sum_val = 0.0
            for j in range(VI.shape[1]):
                sum_val += diff[j] * VI[i, j]
            tmp[i] = sum_val
        result = 0.0
        for i in range(tmp.shape[0]):
            result += tmp[i] * diff[i]
        return np.sqrt(result)

    @jit("float64(float32[:], float32[:])", nopython=True, fastmath=True)
    def _hamming_numba(u, v):  # Hamming
        diff_count = 0
        for i in range(u.shape[0]):
            if u[i] != v[i]:
                diff_count += 1
        return diff_count / u.shape[0]

    @jit("float64(float32[:], float32[:])", nopython=True, fastmath=True)
    def _jaccard_numba(u, v):  # Jaccard
        intersection, union = 0, 0
        for i in range(u.shape[0]):
            u_bool = u[i] != 0
            v_bool = v[i] != 0
            if u_bool and v_bool:
                intersection += 1
            if u_bool or v_bool:
                union += 1
        if union == 0:
            return 0.0
        return 1.0 - (intersection / union)

    @jit("float64(float32[:], float32[:], float32[:])", nopython=True, fastmath=True)
    def _canberra_numba(u, v, w):  # weighted Canberra distance
        n = u.shape[0]
        dist = 0.0
        for i in range(n):
            abs_u = abs(u[i])
            abs_v = abs(v[i])
            denom = abs_u + abs_v
            if denom == 0.0:
                continue
            if denom > 0:  # Avoid division by zero
                num = abs(u[i] - v[i])
                dist += num / denom * w[i]
        return dist

    @jit("float64(float32[:], float32[:])", nopython=True, fastmath=True)
    def _braycurtis_numba(u, v):  # Bray-Curtis distance between two 1D arrays
        diff_sum = 0.0
        abs_sum = 0.0
        for i in range(u.shape[0]):
            diff_sum += abs(u[i] - v[i])
            abs_sum += abs(u[i] + v[i])
        if abs_sum == 0.0:
            return 0.0
        return diff_sum / abs_sum

    # --- Relative Entropy (Kullback-Leibler Divergence) ---
    @jit("float64(float32[:], float32[:])", nopython=True, fastmath=True)
    def _relative_entropy_numba(u, v):  # relative entropy between two 1D arrays
        entropy = 0.0
        for i in range(u.shape[0]):

            # Avoid log(0) and division by zero; input should theoretically be prob dists
            if u[i] > 0 and v[i] > 0:
                entropy += u[i] * np.log(u[i] / v[i])
        return entropy

    # Helper to calculate contingency table components
    @jit("UniTuple(float64, 4)(float32[:], float32[:])", nopython=True, fastmath=True)
    def _get_boolean_counts_numba(u, v):
        c_tt = 0.0  # True-True
        c_tf = 0.0  # True-False
        c_ft = 0.0  # False-True
        c_ff = 0.0  # False-False

        for i in range(u.shape[0]):
            u_bool = u[i] != 0
            v_bool = v[i] != 0

            if u_bool and v_bool:
                c_tt += 1.0
            elif u_bool and not v_bool:
                c_tf += 1.0
            elif not u_bool and v_bool:
                c_ft += 1.0
            else:
                c_ff += 1.0
        return c_tt, c_tf, c_ft, c_ff

    @jit("float64(float32[:], float32[:])", nopython=True, fastmath=True)
    def _dice_numba(u, v):  # Dice dissimilarity between two boolean 1D arrays
        c_tt, c_tf, c_ft, _ = _get_boolean_counts_numba(u, v)
        denom = 2.0 * c_tt + c_tf + c_ft
        if denom == 0.0:
            return 0.0
        return (c_tf + c_ft) / denom

    @jit("float64(float32[:], float32[:])", nopython=True, fastmath=True)
    def _kulsinski_numba(u, v):  # Kulsinski dissimilarity between two boolean 1D arrays
        c_tt, c_tf, c_ft, c_ff = _get_boolean_counts_numba(u, v)
        denom = c_tf + c_ft
        if denom == 0.0:
            return 0.0
        return c_tt / denom

    @jit("float64(float32[:], float32[:])", nopython=True, fastmath=True)
    def _rogers_tanimoto_numba(
        u, v
    ):  # Rogers-Tanimoto dissimilarity between two boolean 1D arrays
        c_tt, c_tf, c_ft, c_ff = _get_boolean_counts_numba(u, v)
        r = 2.0 * (c_tf + c_ft)
        denom = c_tt + c_ff + r
        if denom == 0.0:
            return 0.0
        return r / denom

    @jit("float64(float32[:], float32[:])", nopython=True, fastmath=True)
    def _russellrao_numba(
        u, v
    ):  # Ruseell-Rao dissimilarity between two boolean 1D arrays
        c_tt, c_tf, c_ft, c_ff = _get_boolean_counts_numba(u, v)
        n = c_tt + c_tf + c_ft + c_ff
        if n == 0.0:
            return 0.0
        return (n - c_tt) / n

    @jit("float64(float32[:], float32[:])", nopython=True, fastmath=True)
    def _sokal_michener_numba(
        u, v
    ):  # Sokal-Michener dissimilarity between two boolean 1D arrays
        c_tt, c_tf, c_ft, c_ff = _get_boolean_counts_numba(u, v)
        r = 2.0 * (c_tf + c_ft)
        n = c_tt + c_tf + c_ft + c_ff
        denom = n + r
        if denom == 0.0:
            return 0.0
        return r / denom

    @jit("float64(float32[:], float32[:])", nopython=True, fastmath=True)
    def _sokal_sneath_numba(
        u, v
    ):  # Sokal-Sneath dissimilarity between two boolean 1D arrays
        c_tt, c_tf, c_ft, c_ff = _get_boolean_counts_numba(u, v)
        r = 2.0 * (c_tf + c_ft)
        denom = c_tt + r
        if denom == 0.0:
            return 0.0
        return r / denom

    @jit("float64(float32[:], float32[:])", nopython=True, fastmath=True)
    def _yule_numba(u, v):  # Yule dissimilarity between two boolean 1D arrays
        c_tt, c_tf, c_ft, c_ff = _get_boolean_counts_numba(u, v)
        r = 2.0 * c_tf * c_ft
        denom = c_tt * c_ff + r / 2.0
        if denom == 0.0:
            return 0.0  # Avoid div by zero, definition varies
        return r / denom

    # --- Matrix-based Distances ---
    @jit(
        "float32[:,:](float32[:,:], float32[:,:])",
        nopython=True,
        parallel=True,
        fastmath=True,
    )
    def _pairwise_euclidean_numba(X, Y):  # pragma: no cover
        n_x, n_y = X.shape[0], Y.shape[0]
        n_features = X.shape[1]
        distances = np.empty((n_x, n_y), dtype=np.float32)
        for i in prange(n_x):
            for j in range(n_y):
                dist_sq = 0.0
                for k in range(n_features):
                    dist_sq += (X[i, k] - Y[j, k]) ** 2
                distances[i, j] = np.sqrt(dist_sq)
        return distances


# ==============================================================================
# CUPY IMPLEMENTATIONS (GPU accelerated)
# ==============================================================================


if CUPY_AVAILABLE:

    def _euclidean_cupy(u, v):
        return cp.linalg.norm(u - v)

    def _manhattan_cupy(u, v):
        return cp.sum(cp.abs(u - v))

    def _minkowski_cupy(u, v, p):
        return cp.sum(cp.abs(u - v) ** p) ** (1.0 / p)

    def _chebyshev_cupy(u, v):
        return cp.max(cp.abs(u - v))

    def _cosine_similarity_cupy(u, v):
        dot, norm_u, norm_v = cp.dot(u, v), cp.linalg.norm(u), cp.linalg.norm(v)
        if norm_u == 0.0 and norm_v == 0.0:
            return 1.0  # similarity of two zero vectors = 1
        if norm_u == 0.0 or norm_v == 0.0:
            return 0.0  # similarity with one zero vectors = 0
        return dot / (norm_u * norm_v)

    def _mahalanobis_cupy(u, v, VI):
        diff = u - v
        return cp.sqrt(diff.T @ VI @ diff)

    def _hamming_cupy(u, v):
        return cp.mean(u != v)

    def _jaccard_cupy(u, v):
        u_bool, v_bool = u.astype(bool), v.astype(bool)
        intersection, union = cp.sum(u_bool & v_bool), cp.sum(u_bool | v_bool)
        if union == 0:
            return 0.0
        return 1.0 - (intersection / union)

    def _pairwise_euclidean_cupy(X, Y):
        diff = X[:, None, :] - Y[None, :, :]
        return cp.sqrt(cp.sum(diff**2, axis=-1))

    def _canberra_cupy(u, v, w=None):
        """GPU-accelerated Canberra distance."""
        if w is None:
            w = cp.ones_like(u)
        num = cp.abs(u - v)
        denom = cp.abs(u) + cp.abs(v)
        # Safe division
        mask = denom != 0
        ratios = cp.where(mask, num / denom, 0.0)
        return float(cp.sum(ratios * w))

    def _braycurtis_cupy(u, v):
        diff = cp.abs(u - v)
        sum_val = cp.abs(u + v)
        denom = cp.sum(sum_val)
        if denom == 0.0:
            return 0.0
        return cp.sum(diff) / denom

    def _relative_entropy_cupy(u, v):
        # Filter where both are positive to avoid NaNs/Infs
        mask = (u > 0) & (v > 0)
        u_valid = u[mask]
        v_valid = v[mask]
        return cp.sum(u_valid * cp.log(u_valid / v_valid))

    # --- Boolean Helpers for CuPy ---
    def _get_boolean_counts_cupy(u, v):
        u_bool = u.astype(bool)
        v_bool = v.astype(bool)
        not_u = ~u_bool
        not_v = ~v_bool
        c_tt = cp.sum(u_bool & v_bool)
        c_tf = cp.sum(u_bool & not_v)
        c_ft = cp.sum(not_u & v_bool)
        c_ff = cp.sum(not_u & not_v)
        return float(c_tt), float(c_tf), float(c_ft), float(c_ff)

    def _dice_cupy(u, v):
        c_tt, c_tf, c_ft, _ = _get_boolean_counts_cupy(u, v)
        denom = 2.0 * c_tt + c_tf + c_ft
        return (c_tf + c_ft) / denom if denom != 0 else 0.0

    def _kulsinski_cupy(u, v):
        c_tt, c_tf, c_ft, c_ff = _get_boolean_counts_cupy(u, v)
        denom = c_tf + c_ft
        if denom == 0.0:
            return 0.0
        return c_tt / denom

    def _rogers_tanimoto_cupy(u, v):
        c_tt, c_tf, c_ft, c_ff = _get_boolean_counts_cupy(u, v)
        r = 2.0 * (c_tf + c_ft)
        denom = c_tt + c_ff + r
        return r / denom if denom != 0 else 0.0

    def _russellrao_cupy(u, v):
        c_tt, c_tf, c_ft, c_ff = _get_boolean_counts_cupy(u, v)
        n = c_tt + c_tf + c_ft + c_ff
        return (n - c_tt) / n if n != 0 else 0.0

    def _sokal_michener_cupy(u, v):
        c_tt, c_tf, c_ft, c_ff = _get_boolean_counts_cupy(u, v)
        r = 2.0 * (c_tf + c_ft)
        n = c_tt + c_tf + c_ft + c_ff
        denom = n + r
        return r / denom if denom != 0 else 0.0

    def _sokal_sneath_cupy(u, v):
        c_tt, c_tf, c_ft, c_ff = _get_boolean_counts_cupy(u, v)
        r = 2.0 * (c_tf + c_ft)
        denom = c_tt + r
        return r / denom if denom != 0 else 0.0

    def _yule_cupy(u, v):
        c_tt, c_tf, c_ft, c_ff = _get_boolean_counts_cupy(u, v)
        r = 2.0 * c_tf * c_ft
        denom = c_tt * c_ff + r / 2.0
        return r / denom if denom != 0 else 0.0


# ==============================================================================
# NUMPY IMPLEMENTATIONS (baseline, always available)
# ==============================================================================


def _euclidean_numpy(u, v):
    return np.linalg.norm(u - v)


def _manhattan_numpy(u, v):
    return np.sum(np.abs(u - v))


def _minkowski_numpy(u, v, p):
    return np.sum(np.abs(u - v) ** p) ** (1.0 / p)


def _chebyshev_numpy(u, v):
    return np.max(np.abs(u - v))


def _cosine_similarity_numpy(u, v):
    dot, norm_u, norm_v = np.dot(u, v), np.linalg.norm(u), np.linalg.norm(v)
    if norm_u == 0.0 and norm_v == 0.0:
        return 1.0  # similarity of two zero vectors = 1
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0  # similarity with one zero vectors = 0
    return dot / (norm_u * norm_v)


def _mahalanobis_numpy(u, v, VI):
    diff = u - v
    return np.sqrt(diff.T @ VI @ diff)


def _hamming_numpy(u, v):
    return np.mean(u != v)


def _jaccard_numpy(u, v):
    u_bool, v_bool = u.astype(bool), v.astype(bool)
    intersection, union = np.sum(u_bool & v_bool), np.sum(u_bool | v_bool)
    if union == 0:
        return 0.0
    return 1.0 - (intersection / union)


def _pairwise_euclidean_numpy(X, Y):
    # Using broadcasting for efficiency
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def _canberra_numpy(u, v, w=None):
    """
    Vectorized Canberra distance.
    dist = Σ |u[i] - v[i]| / (|u[i]| + |v[i]|) * w[i]
    """
    if w is None:
        w = np.ones_like(u)
    num = np.abs(u - v)
    denom = np.abs(u) + np.abs(v)
    # Safe division
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(denom != 0, num / denom, 0.0)
    # Weighted sum using einsum (slightly faster than sum)
    return np.einsum("i,i->", ratios, w)


def _braycurtis_numpy(u, v):
    diff = np.abs(u - v)
    sum_val = np.abs(u + v)
    denom = np.sum(sum_val)
    if denom == 0.0:
        return 0.0
    return np.sum(diff) / denom


def _relative_entropy_numpy(u, v):
    # Filter where both are positive to avoid NaNs/Infs (standard KL behavior)
    mask = (u > 0) & (v > 0)
    u_valid = u[mask]
    v_valid = v[mask]
    return np.sum(u_valid * np.log(u_valid / v_valid))


# --- Boolean Helpers for NumPy ---
def _get_boolean_counts_numpy(u, v):
    u_bool = u.astype(bool)
    v_bool = v.astype(bool)
    not_u = ~u_bool
    not_v = ~v_bool
    c_tt = np.sum(u_bool & v_bool)
    c_tf = np.sum(u_bool & not_v)
    c_ft = np.sum(not_u & v_bool)
    c_ff = np.sum(not_u & not_v)
    return c_tt, c_tf, c_ft, c_ff


def _dice_numpy(u, v):
    c_tt, c_tf, c_ft, _ = _get_boolean_counts_numpy(u, v)
    denom = 2.0 * c_tt + c_tf + c_ft
    return (c_tf + c_ft) / denom if denom != 0 else 0.0


def _kulsinski_numpy(u, v):
    """
    Implementation matching the test expectation (Kulczynski 1 behavior).
    Formula: c_tt / (c_tf + c_ft)
    """
    c_tt, c_tf, c_ft, c_ff = _get_boolean_counts_numpy(u, v)
    denom = c_tf + c_ft
    if denom == 0.0:
        return 0.0
    return c_tt / denom


def _rogers_tanimoto_numpy(u, v):
    c_tt, c_tf, c_ft, c_ff = _get_boolean_counts_numpy(u, v)
    r = 2.0 * (c_tf + c_ft)
    denom = c_tt + c_ff + r
    return r / denom if denom != 0 else 0.0


def _russellrao_numpy(u, v):
    c_tt, c_tf, c_ft, c_ff = _get_boolean_counts_numpy(u, v)
    n = c_tt + c_tf + c_ft + c_ff
    return (n - c_tt) / n if n != 0 else 0.0


def _sokal_michener_numpy(u, v):
    c_tt, c_tf, c_ft, c_ff = _get_boolean_counts_numpy(u, v)
    r = 2.0 * (c_tf + c_ft)
    n = c_tt + c_ff  # c_tf + c_ft +
    denom = n + r
    return r / denom if denom != 0 else 0.0


def _sokal_sneath_numpy(u, v):
    c_tt, c_tf, c_ft, c_ff = _get_boolean_counts_numpy(u, v)
    r = 2.0 * (c_tf + c_ft)
    denom = c_tt + r
    return r / denom if denom != 0 else 0.0


def _yule_numpy(u, v):
    c_tt, c_tf, c_ft, c_ff = _get_boolean_counts_numpy(u, v)
    r = 2.0 * c_tf * c_ft
    denom = c_tt * c_ff + r / 2.0
    return r / denom if denom != 0 else 0.0


if not NUMBA_AVAILABLE:
    _euclidean_numba = _euclidean_numpy
    _manhattan_numba = _manhattan_numpy
    _minkowski_numba = _minkowski_numpy
    _chebyshev_numba = _chebyshev_numpy
    _cosine_similarity_numba = _cosine_similarity_numpy
    _mahalanobis_numba = _mahalanobis_numpy
    _hamming_numba = _hamming_numpy
    _jaccard_numba = _jaccard_numpy
    _pairwise_euclidean_numba = _pairwise_euclidean_numpy
    _canberra_numba = _canberra_numpy
    _braycurtis_numba = _braycurtis_numpy
    _relative_entropy_numba = _relative_entropy_numpy
    _dice_numba = _dice_numpy
    _kulsinski_numba = _kulsinski_numpy
    _rogers_tanimoto_numba = _rogers_tanimoto_numpy
    _russellrao_numba = _russellrao_numpy
    _sokal_michener_numba = _sokal_michener_numpy
    _sokal_sneath_numba = _sokal_sneath_numpy
    _yule_numba = _yule_numpy

if not CUPY_AVAILABLE:
    _euclidean_cupy = _euclidean_numpy
    _manhattan_cupy = _manhattan_numpy
    _minkowski_cupy = _minkowski_numpy
    _chebyshev_cupy = _chebyshev_numpy
    _cosine_similarity_cupy = _cosine_similarity_numpy
    _mahalanobis_cupy = _mahalanobis_numpy
    _hamming_cupy = _hamming_numpy
    _jaccard_cupy = _jaccard_numpy
    _pairwise_euclidean_cupy = _pairwise_euclidean_numpy
    _canberra_cupy = _canberra_numpy
    _braycurtis_cupy = _braycurtis_numpy
    _relative_entropy_cupy = _relative_entropy_numpy
    _dice_cupy = _dice_numpy
    _kulsinski_cupy = _kulsinski_numpy
    _rogers_tanimoto_cupy = _rogers_tanimoto_numpy
    _russellrao_cupy = _russellrao_numpy
    _sokal_michener_cupy = _sokal_michener_numpy
    _sokal_sneath_cupy = _sokal_sneath_numpy
    _yule_cupy = _yule_numpy

_kulczynski1_numpy = _kulsinski_numpy
_kulczynski1_numba = _kulsinski_numba
_kulczynski1_cupy = _kulsinski_cupy

# ============================================================================
# PUBLIC DISTANCE CLASS
# ============================================================================


class Distance:
    def __init__(self, gpu_threshold: int = 10_000):
        self.gpu_threshold = gpu_threshold
        self.numba_available = NUMBA_AVAILABLE
        self.gpu_available = CUPY_AVAILABLE

        logging.getLogger(__name__).debug(
            f"Distance initialized. GPU: {self.gpu_available}, Numba: {self.numba_available}"
        )

    def list_distances(self) -> pd.DataFrame:
        """
        Returns a DataFrame listing all available distances functions, their
        descriptions, and usage signatures.

        Returns:
            pd.DataFrame: A table with columns 'Distance', 'Description', and 'Usage'.
        """
        methods_data = []

        # Iterate over all members of the instance
        for name, func in inspect.getmembers(self, predicate=inspect.ismethod):
            # Skip private methods (starting with _) and the list_distances method itself
            if name.startswith("_") or name == "list_distances":
                continue

            # Get the first line of the docstring
            doc = inspect.getdoc(func)
            description = doc.split("\n")[0] if doc else "No description available."

            # Get the function signature (arguments)
            try:
                signature = str(inspect.signature(func))
            except ValueError:
                signature = "(...)"

            methods_data.append(
                {
                    "Distance": name,
                    "Description": description,
                    "Usage": f"{name}{signature}",
                }
            )

        # Create DataFrame and sort by Distance name
        df = pd.DataFrame(methods_data)
        if not df.empty:
            df = df.sort_values(by="Distance").reset_index(drop=True)

        return df

    def _validate_vectors(self, u, v, check_dims=True):
        if check_dims and (u.ndim != 1 or v.ndim != 1):
            raise ValueError("Inputs must be 1D vectors.")
        if u.shape[-1] != v.shape[-1]:
            raise ValueError(
                "Input vectors/matrices must have the same number of features (last dimension)."
            )
        if u.shape[0] == 0:
            return 0.0
        return None

    def _dispatch_v2v(self, name: str, u: np.ndarray, v: np.ndarray, force_cpu: bool):
        backend, use_gpu = "numpy", False
        if self.gpu_available and not force_cpu and u.size >= self.gpu_threshold:
            backend, use_gpu = "cupy", True
        elif self.numba_available:
            backend = "numba"

        func = globals().get(f"_{name}_{backend}") or globals().get(f"_{name}_numpy")

        u_c, v_c = u.astype(np.float32), v.astype(np.float32)
        if use_gpu:
            u_c, v_c = cp.asarray(u_c), cp.asarray(v_c)

        return func, u_c, v_c

    def _dispatch_m2m(self, name: str, X: np.ndarray, Y: np.ndarray, force_cpu: bool):
        backend, use_gpu = "numpy", False
        if self.gpu_available and not force_cpu and X.size >= self.gpu_threshold:
            backend, use_gpu = "cupy", True
        elif self.numba_available:
            backend = "numba"

        func = globals().get(f"_{name}_{backend}") or globals().get(f"_{name}_numpy")

        X_c, Y_c = X.astype(np.float32), Y.astype(np.float32)
        if use_gpu:
            X_c, Y_c = cp.asarray(X_c), cp.asarray(Y_c)
        return func, X_c, Y_c, use_gpu

    # --- Vector-to-Vector Metrics ---
    def euclidean(self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False):
        """Euclidean distance between two vectors."""
        if self._validate_vectors(u, v) is not None:
            return 0.0
        func, u_c, v_c = self._dispatch_v2v("euclidean", u, v, force_cpu)
        return float(func(u_c, v_c))

    def manhattan(self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False):
        """Manhattan (city-block) distance between two vectors."""
        if self._validate_vectors(u, v) is not None:
            return 0.0
        func, u_c, v_c = self._dispatch_v2v("manhattan", u, v, force_cpu)
        return float(func(u_c, v_c))

    def minkowski(self, u: np.ndarray, v: np.ndarray, p: int, force_cpu: bool = False):
        """Minkowski distance between two vectors, where the parameter 'p' controls
        the type of distance, allowing for different ways to calculate similarity."""
        if p < 1:
            raise ValueError("p must be at least 1 for Minkowski distance.")
        if self._validate_vectors(u, v) is not None:
            return 0.0
        func, u_c, v_c = self._dispatch_v2v("minkowski", u, v, force_cpu)
        return float(func(u_c, v_c, p))

    def chebyshev(self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False):
        """Chebyshev distance between two vectors."""
        if self._validate_vectors(u, v) is not None:
            return 0.0
        func, u_c, v_c = self._dispatch_v2v("chebyshev", u, v, force_cpu)
        return float(func(u_c, v_c))

    def cosine_similarity(self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False):
        """Cosine similarity between two vectors."""
        if self._validate_vectors(u, v) is not None:
            return 1.0
        func, u_c, v_c = self._dispatch_v2v("cosine_similarity", u, v, force_cpu)
        return float(func(u_c, v_c))

    def mahalanobis(
        self, u: np.ndarray, v: np.ndarray, cov_inv: np.ndarray, force_cpu: bool = False
    ):
        """Mahalanobis that quantifies the distance between a point and
        a distribution or between two points, accounting for the correlations
        and variances of the variables in a multivariate dataset."""
        if self._validate_vectors(u, v) is not None:
            return 0.0
        if (
            cov_inv.ndim != 2
            or cov_inv.shape[0] != cov_inv.shape[1]
            or cov_inv.shape[0] != u.shape[0]
        ):
            raise ValueError(
                "Inverse covariance matrix must be square with dimensions matching the vectors."
            )

        func, u_c, v_c = self._dispatch_v2v("mahalanobis", u, v, force_cpu)
        cov_inv_c = cov_inv.astype(np.float32)
        if CUPY_AVAILABLE and isinstance(u_c, cp.ndarray):
            cov_inv_c = cp.asarray(cov_inv_c)
        return float(func(u_c, v_c, cov_inv_c))

    def hamming(self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False):
        """Hamming distance between two vectors."""
        if self._validate_vectors(u, v) is not None:
            return 0.0
        func, u_c, v_c = self._dispatch_v2v("hamming", u, v, force_cpu)
        return float(func(u_c, v_c))

    def jaccard(self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False):
        """Jaccard distance between two vectors."""
        if self._validate_vectors(u, v) is not None:
            return 0.0
        func, u_c, v_c = self._dispatch_v2v("jaccard", u, v, force_cpu)
        return float(func(u_c, v_c))

    def haversine(
        self, lat1: float, lon1: float, lat2: float, lon2: float, radius: float = 6371.0
    ):
        """Haversine distance calculates the shortest distance (great-circle distance)
        between two points on a sphere, like Earth, using their latitude and longitude,
        accounting for the planet's curvature for accuracy over long distances,
        unlike simple flat-surface calculations."""
        lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        return radius * 2 * np.arcsin(np.sqrt(a))

    def canberra(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray = None,
        force_cpu: bool = False,
    ):
        """Calculates the Canberra distance between two vectors."""

        if self._validate_vectors(u, v) is not None:
            return 0.0

        if w is not None:
            w = np.asarray(w, dtype=np.float32)
            return float(_canberra_numba(u.astype(np.float32), v.astype(np.float32), w))

        func, u_c, v_c = self._dispatch_v2v("canberra", u, v, force_cpu)
        return float(func(u_c, v_c))

    def cityblock(self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False) -> float:
        """
        Calculates the City Block distance (alias for Manhattan distance).
        """
        return self.manhattan(u, v, force_cpu)

    def braycurtis(
        self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False
    ) -> float:
        """Calculates the Bray-Curtis distance between two vectors."""
        if self._validate_vectors(u, v) is not None:
            return 0.0
        func, u_c, v_c = self._dispatch_v2v("braycurtis", u, v, force_cpu)
        return float(func(u_c, v_c))

    def relative_entropy(
        self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False
    ) -> float:
        """
        Calculates the Relative Entropy (Kullback-Leibler Divergence) D(P||Q).
        u and v should be probability distributions.
        """
        if self._validate_vectors(u, v) is not None:
            return 0.0
        func, u_c, v_c = self._dispatch_v2v("relative_entropy", u, v, force_cpu)
        return float(func(u_c, v_c))

    # --- Boolean Dissimilarity Measures ---
    def dice(self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False) -> float:
        """Calculates the Dice dissimilarity between two boolean 1D arrays."""
        if self._validate_vectors(u, v) is not None:
            return 0.0
        func, u_c, v_c = self._dispatch_v2v("dice", u, v, force_cpu)
        return float(func(u_c, v_c))

    def kulsinski(self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False) -> float:
        """Calculates the Kulsinski dissimilarity between two boolean 1D arrays."""
        if self._validate_vectors(u, v) is not None:
            return 0.0
        func, u_c, v_c = self._dispatch_v2v("kulsinski", u, v, force_cpu)
        return float(func(u_c, v_c))

    def kulczynski1(
        self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False
    ) -> float:
        """Alias for kulsinski to match SciPy naming."""
        return self.kulsinski(u, v, force_cpu)

    def rogers_tanimoto(
        self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False
    ) -> float:
        """Calculates the Rogers-Tanimoto dissimilarity between two boolean 1D arrays."""
        if self._validate_vectors(u, v) is not None:
            return 0.0
        func, u_c, v_c = self._dispatch_v2v("rogers_tanimoto", u, v, force_cpu)
        return float(func(u_c, v_c))

    def russellrao(
        self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False
    ) -> float:
        """Calculates the Russell-Rao dissimilarity between two boolean 1D arrays."""
        if self._validate_vectors(u, v) is not None:
            return 0.0
        func, u_c, v_c = self._dispatch_v2v("russellrao", u, v, force_cpu)
        return float(func(u_c, v_c))

    def sokal_michener(
        self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False
    ) -> float:
        """Calculates the Sokal-Michener dissimilarity between two boolean 1D arrays."""
        if self._validate_vectors(u, v) is not None:
            return 0.0
        func, u_c, v_c = self._dispatch_v2v("sokal_michener", u, v, force_cpu)
        return float(func(u_c, v_c))

    def sokal_sneath(
        self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False
    ) -> float:
        """Calculates the Sokal-Sneath dissimilarity between two boolean 1D arrays."""
        if self._validate_vectors(u, v) is not None:
            return 0.0
        func, u_c, v_c = self._dispatch_v2v("sokal_sneath", u, v, force_cpu)
        return float(func(u_c, v_c))

    def yule(self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False) -> float:
        """Calculates the Yule dissimilarity between two boolean 1D arrays."""
        if self._validate_vectors(u, v) is not None:
            return 0.0
        func, u_c, v_c = self._dispatch_v2v("yule", u, v, force_cpu)
        return float(func(u_c, v_c))

    # --- Matrix-based Metrics ---
    def pairwise_euclidean(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None, force_cpu: bool = False
    ):
        """Pairwise Euclidean distance calculates the straight-line distance
        between every possible pair of points (vectors) in a dataset,
        resulting in a distance matrix that shows how dissimilar each point
        is from every other point, commonly used in clustering and
        machine learning to group similar items"""
        Y = X if Y is None else Y

        if X.shape[0] == 0:
            return np.empty((0, Y.shape[0]), dtype=np.float32)

        if self._validate_vectors(X, Y, check_dims=False) is not None:
            return np.empty((X.shape[0], Y.shape[0]), dtype=np.float32)

        func, X_c, Y_c, use_gpu = self._dispatch_m2m(
            "pairwise_euclidean", X, Y, force_cpu
        )
        if func is None:  # Fallback to numpy if no specialized func
            return _pairwise_euclidean_numpy(X.astype(np.float32), Y.astype(np.float32))

        distances = func(X_c, Y_c)
        if use_gpu:
            return cp.asnumpy(distances)
        return distances

    def kmeans_distance(self, X, centroids, force_cpu: bool = False):
        """Uses distance metrics, primarily Euclidean distance,
        to group similar data points by finding cluster centers (centroids)
        that minimize the total intra-cluster distance, known as inertia,
        calculated as the sum of squared distances from each point
        to its assigned centroid, aiming to create compact, similar clusters."""
        return self.pairwise_euclidean(X, centroids, force_cpu=force_cpu)

    def knn_distances(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        k: int = 5,
        force_cpu: bool = False,
    ):
        """This distance is the measure of similarity between two vectors."""
        Y = X if Y is None else Y
        if X.shape[0] == 0:
            return (np.empty((0, k)), np.empty((0, k), dtype=int))
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                "Input matrices must have the same number of features (columns)."
            )
        if self._validate_vectors(X, Y, check_dims=False) is not None:
            return (
                np.empty((0, Y.shape[0]), dtype=np.float32),
                np.empty((0, Y.shape[0]), dtype=np.float32),
            )

        dist_matrix = self.pairwise_euclidean(X, Y, force_cpu=force_cpu)

        # Argpartition is faster than argsort if we only need the first k elements
        indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]

        # Now get the actual distances for these k indices
        distances = np.take_along_axis(dist_matrix, indices, axis=1)

        # Sort the k results for each row
        sorted_order = np.argsort(distances, axis=1)
        distances = np.take_along_axis(distances, sorted_order, axis=1)
        indices = np.take_along_axis(indices, sorted_order, axis=1)

        return distances, indices

    def radius_neighbors(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        radius: float = 1.0,
        force_cpu: bool = False,
    ):
        """Radius neighbors distance refers to finding data points within
        a specific distance (radius) from a query point, used in machine learning
        (like classification/regression) and data analysis to identify local groups,
        with distances calculated via metrics like Euclidean or Manhattan,
        providing insights into spatial relationships or local density."""
        Y = X if Y is None else Y
        if X.shape[0] == 0:
            return ([], [])

        dist_matrix = self.pairwise_euclidean(X, Y, force_cpu=force_cpu)

        all_indices, all_distances = [], []
        for i in range(dist_matrix.shape[0]):
            row = dist_matrix[i, :]
            mask = row <= radius

            indices = np.where(mask)[0]
            distances = row[mask]

            # Sort results by distance
            sorted_order = np.argsort(distances)
            all_indices.append(indices[sorted_order])
            all_distances.append(distances[sorted_order])

        return all_distances, all_indices
