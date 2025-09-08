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
"""

# tests/test_distance.py
"""
Comprehensive tests for the Distance class in ds_tools.
This suite covers initialization, backend dispatching (NumPy, Numba, CuPy),
and the correctness of all implemented distance and similarity metrics.

This test suite covers almost 100% of the distance.py functionality, including:
All 12 metrics, including their correctness compared to reference implementations.
All backends (NumPy, Numba, CuPy) and the logic behind their selection (gpu_threshold, force_cpu).
All major code paths, including matrix (pairwise, knn, radius) and vector functions.
Error and edge case handling (incorrect sizes, empty arrays, incorrect parameters).
"""
import numpy as np
import pytest
from scipy.spatial.distance import cdist  # Used as a trusted reference

# --- Import the necessary components from the library ---

# --- Attempt to import optional dependencies for marking tests ---
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def small_sample_vectors():
    """Provides a pair of small vectors that won't trigger the GPU threshold."""
    np.random.seed(42)
    u = np.random.rand(100).astype(np.float32)
    v = np.random.rand(100).astype(np.float32)
    return u, v


@pytest.fixture(scope="module")
def large_sample_vectors():
    """Provides a pair of large vectors that WILL trigger the GPU threshold."""
    np.random.seed(42)
    size = 15_000  # Larger than the default 10k threshold
    u = np.random.rand(size).astype(np.float32)
    v = np.random.rand(size).astype(np.float32)
    return u, v


@pytest.fixture(scope="module")
def sample_matrices():
    """Provides two matrices for pairwise and neighbor calculations."""
    np.random.seed(42)
    X = np.random.rand(50, 10).astype(np.float32)
    Y = np.random.rand(60, 10).astype(np.float32)
    return X, Y


@pytest.fixture(scope="module")
def inverse_covariance_matrix():
    """Provides a sample inverse covariance matrix for Mahalanobis distance."""
    np.random.seed(42)
    # Generate a random symmetric positive-definite matrix
    A = np.random.rand(100, 100)
    cov = np.dot(A, A.T)
    # Add a small value to the diagonal for numerical stability
    cov += np.eye(100) * 1e-6
    return np.linalg.inv(cov).astype(np.float32)


# ============================================================================
# Tests for Initialization and Dispatching
# ============================================================================


def test_distance_initialization(tools):
    """Tests that the Distance class initializes correctly."""
    # 'tools' fixture provides a DSTools instance, so we access .distance
    dist = tools.distance
    assert dist.gpu_available == CUPY_AVAILABLE
    assert dist.numba_available == NUMBA_AVAILABLE


def test_backend_dispatching_logic(
    tools, mocker, small_sample_vectors, large_sample_vectors
):
    """Tests that the correct backend is chosen based on data size and flags."""
    if not (CUPY_AVAILABLE and NUMBA_AVAILABLE):
        pytest.skip("This test requires both CuPy and Numba.")

    dist = tools.distance
    mock_numba = mocker.patch("ds_tool.distance._euclidean_numba", return_value=1.0)
    mock_cupy = mocker.patch("ds_tool.distance._euclidean_cupy", return_value=2.0)

    # 1. Small data -> Numba should be used
    u_small, v_small = small_sample_vectors
    dist.euclidean(u_small, v_small, force_cpu=False)
    mock_numba.assert_called_once()
    mock_cupy.assert_not_called()
    mock_numba.reset_mock()

    # 2. Large data -> CuPy should be used
    u_large, v_large = large_sample_vectors
    dist.euclidean(u_large, v_large, force_cpu=False)
    mock_numba.assert_not_called()
    mock_cupy.assert_called_once()
    mock_cupy.reset_mock()

    # 3. Large data with force_cpu=True -> Numba should be used
    dist.euclidean(u_large, v_large, force_cpu=True)
    mock_numba.assert_called_once()
    mock_cupy.assert_not_called()


# ============================================================================
# Tests for Vector-to-Vector Distances
# ============================================================================

# --- Parametrized test for simple distances ---
VECTOR_METRICS = [
    ("euclidean", {}, lambda u, v: np.linalg.norm(u - v)),
    ("manhattan", {}, lambda u, v: np.sum(np.abs(u - v))),
    ("minkowski", {"p": 3}, lambda u, v, p: np.sum(np.abs(u - v) ** p) ** (1 / p)),
    ("chebyshev", {}, lambda u, v: np.max(np.abs(u - v))),
    (
        "cosine_similarity",
        {},
        lambda u, v: 1 - cdist(u.reshape(1, -1), v.reshape(1, -1), "cosine")[0, 0],
    ),
    ("hamming", {}, lambda u, v: np.mean(u != v)),
    (
        "jaccard",
        {},
        lambda u, v: 1.0
        - (
            np.sum(u.astype(bool) & v.astype(bool))
            / np.sum(u.astype(bool) | v.astype(bool))
        ),
    ),
]


@pytest.mark.parametrize("method_name, kwargs, trusted_func", VECTOR_METRICS)
def test_vector_distances_correctness(
    tools, method_name, kwargs, trusted_func, large_sample_vectors
):
    """Tests numerical correctness of vector-to-vector distances against a trusted source."""
    u, v = large_sample_vectors

    # Get the result from our library (testing GPU path)
    method = getattr(tools.distance, method_name)
    result = method(u, v, **kwargs)

    # Get the expected result from the trusted source (NumPy/SciPy)
    expected = trusted_func(u, v, **kwargs)

    assert np.isclose(result, expected, rtol=1e-5)


def test_mahalanobis_correctness(
    tools, small_sample_vectors, inverse_covariance_matrix
):
    """Separate test for Mahalanobis due to its unique signature."""
    u, v = small_sample_vectors
    VI = inverse_covariance_matrix

    # SciPy's implementation is a trusted source
    from scipy.spatial.distance import mahalanobis as scipy_mahalanobis

    expected = scipy_mahalanobis(u, v, VI)

    result = tools.distance.mahalanobis(u, v, VI)
    assert np.isclose(result, expected)


def test_haversine_correctness(tools):
    """Tests Haversine distance with a known example (Paris to London)."""
    # Coordinates for Paris and London
    lat1, lon1 = 48.8566, 2.3522  # Paris
    lat2, lon2 = 51.5074, -0.1278  # London

    expected_distance_km = 343.5  # Approximate distance

    result = tools.distance.haversine(lat1, lon1, lat2, lon2)
    assert np.isclose(result, expected_distance_km, atol=1)  # Allow 1km tolerance


# ============================================================================
# Tests for Matrix-based Functions
# ============================================================================


def test_pairwise_euclidean_correctness(tools, sample_matrices):
    """Tests pairwise Euclidean distance against SciPy's cdist."""
    X, Y = sample_matrices

    # Test within a single matrix
    expected_within = cdist(X, X, "euclidean")
    result_within = tools.distance.pairwise_euclidean(X)
    assert np.allclose(result_within, expected_within)

    # Test between two matrices
    expected_between = cdist(X, Y, "euclidean")
    result_between = tools.distance.pairwise_euclidean(X, Y)
    assert np.allclose(result_between, expected_between)


def test_kmeans_distance_is_alias(tools, mocker, sample_matrices):
    """Tests that kmeans_distance is a correct alias for pairwise_euclidean."""
    X, centroids = sample_matrices
    mock_pairwise = mocker.patch.object(
        tools.distance, "pairwise_euclidean", return_value=np.array([])
    )

    tools.distance.kmeans_distance(X, centroids)

    mock_pairwise.assert_called_once_with(X, centroids, force_cpu=False)


def test_knn_distances_correctness(tools, sample_matrices):
    """Tests k-NN distance calculation for correct shapes and properties."""
    X, Y = sample_matrices
    k = 5

    distances, indices = tools.distance.knn_distances(X, Y, k=k)

    assert distances.shape == (X.shape[0], k)
    assert indices.shape == (X.shape[0], k)
    assert np.all(distances >= 0)
    # Check that distances in each row are sorted
    assert np.all(np.diff(distances, axis=1) >= 0)


def test_radius_neighbors_correctness(tools, sample_matrices):
    """Tests radius neighbors search for correct properties."""
    X, Y = sample_matrices
    radius = 0.7

    distances, indices = tools.distance.radius_neighbors(X, Y, radius=radius)

    assert isinstance(distances, list) and isinstance(indices, list)
    assert len(distances) == X.shape[0]

    # Check that all returned distances are within the radius
    for dist_array in distances:
        if len(dist_array) > 0:
            assert np.all(dist_array <= radius)


# ============================================================================
# Tests for Edge Cases and Error Handling
# ============================================================================


def test_empty_vector_input(tools):
    """Tests graceful handling of empty vector inputs."""
    u, v = np.array([]), np.array([])
    assert tools.distance.euclidean(u, v) == 0.0
    assert tools.distance.cosine_similarity(u, v) == 1.0  # Perfect similarity


def test_vector_shape_mismatch_raises_error(tools):
    """Tests that a shape mismatch in vectors raises ValueError."""
    u = np.array([1, 2, 3])
    v = np.array([1, 2])
    with pytest.raises(
        ValueError, match="Input vectors/matrices must have the same number of features"
    ):
        tools.distance.euclidean(u, v)


def test_invalid_minkowski_p_raises_error(tools, small_sample_vectors):
    """Tests that p < 1 in Minkowski distance raises ValueError."""
    u, v = small_sample_vectors
    with pytest.raises(ValueError, match="p must be at least 1"):
        tools.distance.minkowski(u, v, p=0)


def test_invalid_mahalanobis_vi_raises_error(tools, small_sample_vectors):
    """Tests that an invalid inverse covariance matrix raises ValueError."""
    u, v = small_sample_vectors
    VI_bad_shape = np.eye(u.shape[0] - 1)  # Wrong dimension
    with pytest.raises(ValueError, match="Inverse covariance matrix must be square"):
        tools.distance.mahalanobis(u, v, VI_bad_shape)
