# src/ds_tools/distance.py
"""
This module provides a collection of high-performance, system-aware distance
and similarity metric implementations, with support for CPU parallelization
and GPU acceleration. All implementations are self-contained where feasible.
"""
import warnings
from typing import Optional

import numpy as np

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
# PRIVATE IMPLEMENTATIONS (The actual computation engines)
# ============================================================================

# --- Vector-to-Vector Distances ---

# Euclidean / L2
if NUMBA_AVAILABLE:

    @jit("float64(float32[:], float32[:])", nopython=True, fastmath=True)
    def _euclidean_numba(u, v):
        dist_sq = 0.0
        for i in range(u.shape[0]):
            dist_sq += (u[i] - v[i]) ** 2
        return np.sqrt(dist_sq)


if CUPY_AVAILABLE:

    def _euclidean_cupy(u, v):
        return cp.linalg.norm(u - v)


def _euclidean_numpy(u, v):
    return np.linalg.norm(u - v)


# Manhattan / L1
if NUMBA_AVAILABLE:

    @jit("float64(float32[:], float32[:])", nopython=True, fastmath=True)
    def _manhattan_numba(u, v):
        dist = 0.0
        for i in range(u.shape[0]):
            dist += abs(u[i] - v[i])
        return dist


if CUPY_AVAILABLE:

    def _manhattan_cupy(u, v):
        return cp.sum(cp.abs(u - v))


def _manhattan_numpy(u, v):
    return np.sum(np.abs(u - v))


# Minkowski
if NUMBA_AVAILABLE:

    @jit("float64(float32[:], float32[:], int_)", nopython=True, fastmath=True)
    def _minkowski_numba(u, v, p):
        dist = 0.0
        for i in range(u.shape[0]):
            dist += abs(u[i] - v[i]) ** p
        return dist ** (1.0 / p)


if CUPY_AVAILABLE:

    def _minkowski_cupy(u, v, p):
        return cp.sum(cp.abs(u - v) ** p) ** (1.0 / p)


def _minkowski_numpy(u, v, p):
    return np.sum(np.abs(u - v) ** p) ** (1.0 / p)


# Chebyshev / L-infinity
if NUMBA_AVAILABLE:

    @jit("float64(float32[:], float32[:])", nopython=True, fastmath=True)
    def _chebyshev_numba(u, v):
        max_dist = 0.0
        for i in range(u.shape[0]):
            dist = abs(u[i] - v[i])
            if dist > max_dist:
                max_dist = dist
        return max_dist


if CUPY_AVAILABLE:

    def _chebyshev_cupy(u, v):
        return cp.max(cp.abs(u - v))


def _chebyshev_numpy(u, v):
    return np.max(np.abs(u - v))


# Cosine Similarity
if NUMBA_AVAILABLE:

    @jit("float64(float32[:], float32[:])", nopython=True, fastmath=True)
    def _cosine_similarity_numba(u, v):
        dot, norm_u, norm_v = 0.0, 0.0, 0.0
        for i in range(u.shape[0]):
            dot += u[i] * v[i]
            norm_u += u[i] ** 2
            norm_v += v[i] ** 2
        norm_u, norm_v = np.sqrt(norm_u), np.sqrt(norm_v)
        if norm_u == 0.0 or norm_v == 0.0:
            return 0.0
        if norm_u == 0.0 and norm_v == 0.0:
            return 1.0
        return 1.0 - (dot / (norm_u * norm_v))


if CUPY_AVAILABLE:

    def _cosine_similarity_cupy(u, v):
        dot, norm_u, norm_v = cp.dot(u, v), cp.linalg.norm(u), cp.linalg.norm(v)
        if norm_u == 0.0 or norm_v == 0.0:
            return 0.0
        return 1.0 - (dot / (norm_u * norm_v))


def _cosine_similarity_numpy(u, v):
    dot, norm_u, norm_v = np.dot(u, v), np.linalg.norm(u), np.linalg.norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return 1.0 - (dot / (norm_u * norm_v))


# Mahalanobis
if NUMBA_AVAILABLE:

    @jit("float64(float32[:], float32[:], float32[:,:])", nopython=True, fastmath=True)
    def _mahalanobis_numba(u, v, VI):
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


if CUPY_AVAILABLE:

    def _mahalanobis_cupy(u, v, VI):
        diff = u - v
        return cp.sqrt(diff.T @ VI @ diff)


def _mahalanobis_numpy(u, v, VI):
    diff = u - v
    return np.sqrt(diff.T @ VI @ diff)


# Hamming
if NUMBA_AVAILABLE:

    @jit("float64(float32[:], float32[:])", nopython=True, fastmath=True)
    def _hamming_numba(u, v):
        diff_count = 0
        for i in range(u.shape[0]):
            if u[i] != v[i]:
                diff_count += 1
        return diff_count / u.shape[0]


if CUPY_AVAILABLE:

    def _hamming_cupy(u, v):
        return cp.mean(u != v)


def _hamming_numpy(u, v):
    return np.mean(u != v)


# Jaccard
if NUMBA_AVAILABLE:

    @jit("float64(float32[:], float32[:])", nopython=True, fastmath=True)
    def _jaccard_numba(u, v):
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


if CUPY_AVAILABLE:

    def _jaccard_cupy(u, v):
        u_bool, v_bool = u.astype(bool), v.astype(bool)
        intersection, union = cp.sum(u_bool & v_bool), cp.sum(u_bool | v_bool)
        if union == 0:
            return 0.0
        return 1.0 - (intersection / union)


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


# --- Matrix-based Distances ---
if NUMBA_AVAILABLE:

    @jit(
        "float32[:,:](float32[:,:], float32[:,:])",
        nopython=True,
        parallel=True,
        fastmath=True,
    )
    def _pairwise_euclidean_numba(X, Y):
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


if CUPY_AVAILABLE:

    def _pairwise_euclidean_cupy(X, Y):
        diff = X[:, None, :] - Y[None, :, :]
        return cp.sqrt(cp.sum(diff**2, axis=-1))


# ============================================================================
# PUBLIC DISTANCE CLASS
# ============================================================================
class Distance:
    def __init__(self, gpu_threshold: int = 10_000):
        self.gpu_threshold = gpu_threshold
        self.numba_available = NUMBA_AVAILABLE
        self.gpu_available = CUPY_AVAILABLE
        print(
            f"Distance initialized. GPU: {self.gpu_available}, Numba: {self.numba_available}"
        )

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

        func = globals().get(f"_{name}_{backend}")

        X_c, Y_c = X.astype(np.float32), Y.astype(np.float32)
        if use_gpu:
            X_c, Y_c = cp.asarray(X_c), cp.asarray(Y_c)
        return func, X_c, Y_c, use_gpu

    # --- Vector-to-Vector Metrics ---
    def euclidean(self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False):
        if self._validate_vectors(u, v) is not None:
            return 0.0
        func, u_c, v_c = self._dispatch_v2v("euclidean", u, v, force_cpu)
        return float(func(u_c, v_c))

    def manhattan(self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False):
        if self._validate_vectors(u, v) is not None:
            return 0.0
        func, u_c, v_c = self._dispatch_v2v("manhattan", u, v, force_cpu)
        return float(func(u_c, v_c))

    def minkowski(self, u: np.ndarray, v: np.ndarray, p: int, force_cpu: bool = False):
        if p < 1:
            raise ValueError("p must be at least 1 for Minkowski distance.")
        if self._validate_vectors(u, v) is not None:
            return 0.0
        func, u_c, v_c = self._dispatch_v2v("minkowski", u, v, force_cpu)
        return float(func(u_c, v_c, p))

    def chebyshev(self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False):
        if self._validate_vectors(u, v) is not None:
            return 0.0
        func, u_c, v_c = self._dispatch_v2v("chebyshev", u, v, force_cpu)
        return float(func(u_c, v_c))

    def cosine_similarity(self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False):
        if self._validate_vectors(u, v) is not None:
            return 1.0
        func, u_c, v_c = self._dispatch_v2v("cosine_similarity", u, v, force_cpu)
        return float(func(u_c, v_c))

    def mahalanobis(
        self, u: np.ndarray, v: np.ndarray, cov_inv: np.ndarray, force_cpu: bool = False
    ):
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
        if self._validate_vectors(u, v) is not None:
            return 0.0
        func, u_c, v_c = self._dispatch_v2v("hamming", u, v, force_cpu)
        return float(func(u_c, v_c))

    def jaccard(self, u: np.ndarray, v: np.ndarray, force_cpu: bool = False):
        if self._validate_vectors(u, v) is not None:
            return 0.0
        func, u_c, v_c = self._dispatch_v2v("jaccard", u, v, force_cpu)
        return float(func(u_c, v_c))

    def haversine(
        self, lat1: float, lon1: float, lat2: float, lon2: float, radius: float = 6371.0
    ):
        lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        return radius * 2 * np.arcsin(np.sqrt(a))

    # --- Matrix-based Metrics ---
    def pairwise_euclidean(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None, force_cpu: bool = False
    ):
        Y = X if Y is None else Y

        if X.shape[0] == 0:
            return np.empty((0, Y.shape[0]), dtype=np.float32)

        if self._validate_vectors(X, Y, check_dims=False) is not None:
            return np.empty((X.shape[0], Y.shape[0]), dtype=np.float32)

        func, X_c, Y_c, use_gpu = self._dispatch_m2m(
            "pairwise_euclidean", X, Y, force_cpu
        )
        if func is None:  # Fallback to numpy if no specialized func
            return self._dispatch_m2m("pairwise_euclidean", X, Y, force_cpu=True)[0](
                X_c, Y_c
            )

        distances = func(X_c, Y_c)
        if use_gpu:
            return cp.asnumpy(distances)
        return distances

    def kmeans_distance(self, X, centroids, force_cpu: bool = False):
        return self.pairwise_euclidean(X, centroids, force_cpu=force_cpu)

    def knn_distances(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        k: int = 5,
        force_cpu: bool = False,
    ):
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
        Y = X if Y is None else Y
        if self._validate_vectors(X, Y, check_dims=False) is not None:
            return np.empty((0, Y.shape[0]), dtype=np.float32)

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
