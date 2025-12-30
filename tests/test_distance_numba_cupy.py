# tests/test_distance_numba_cupy.py
import numpy as np
import pytest
from ds_tools import distance

NUMBA_AVAILABLE = distance.NUMBA_AVAILABLE
CUPY_AVAILABLE = distance.CUPY_AVAILABLE
cp = distance.cp

# ---------------- NUMBA TESTS ----------------


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_euclidean_numba():
    u = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    res = distance._euclidean_numba(u, v)
    assert np.isclose(res, np.linalg.norm(u - v))


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_manhattan_numba():
    u = np.array([1.0, 2.0], dtype=np.float32)
    v = np.array([4.0, 6.0], dtype=np.float32)
    res = distance._manhattan_numba(u, v)
    assert np.isclose(res, np.sum(np.abs(u - v)))


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_minkowski_numba():
    u = np.array([1.0, 2.0], dtype=np.float32)
    v = np.array([4.0, 6.0], dtype=np.float32)
    res = distance._minkowski_numba(u, v, p=3)
    expected = np.power(np.sum(np.abs(u - v) ** 3), 1 / 3)
    assert np.isclose(res, expected)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_chebyshev_numba():
    u = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v = np.array([2.0, 5.0, 7.0], dtype=np.float32)
    res = distance._chebyshev_numba(u, v)
    assert np.isclose(res, np.max(np.abs(u - v)))


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_cosine_similarity_numba():
    u = np.array([1.0, 0.0], dtype=np.float32)
    v = np.array([0.0, 1.0], dtype=np.float32)
    res = distance._cosine_similarity_numba(u, v)
    # Ортогональные векторы → 0
    assert np.isclose(res, 0.0)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_mahalanobis_numba():
    u = np.array([1.0, 2.0], dtype=np.float32)
    v = np.array([2.0, 4.0], dtype=np.float32)
    VI = np.eye(2, dtype=np.float32)  # inverse covariance = identity
    res = distance._mahalanobis_numba(u, v, VI)
    assert np.isclose(res, np.linalg.norm(u - v))


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_hamming_numba():
    u = np.array([1, 0, 1, 1], dtype=np.float32)
    v = np.array([1, 1, 0, 1], dtype=np.float32)
    res = distance._hamming_numba(u, v)
    expected = np.mean(u != v)
    assert np.isclose(res, expected)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_jaccard_numba_nonempty_and_empty():
    u = np.array([1, 0, 1], dtype=np.float32)
    v = np.array([1, 1, 0], dtype=np.float32)
    res = distance._jaccard_numba(u, v)
    intersection = np.sum(u.astype(bool) & v.astype(bool))
    union = np.sum(u.astype(bool) | v.astype(bool))
    expected = 1.0 - (intersection / union)
    assert np.isclose(res, expected)

    # edge-case union==0
    u0 = np.array([0, 0, 0], dtype=np.float32)
    v0 = np.array([0, 0, 0], dtype=np.float32)
    res0 = distance._jaccard_numba(u0, v0)
    assert res0 == 0.0


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
def test_pairwise_euclidean_numba():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    Y = np.array([[5.0, 6.0]], dtype=np.float32)
    res = distance._pairwise_euclidean_numba(X, Y)
    assert res.shape == (2, 1)


# ---------------- CUPY TESTS ----------------


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_euclidean_cupy():
    u = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
    v = cp.array([4.0, 5.0, 6.0], dtype=cp.float32)
    res = distance._euclidean_cupy(u, v)
    assert cp.allclose(res, cp.linalg.norm(u - v))


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_manhattan_cupy():
    u = cp.array([1.0, 2.0], dtype=cp.float32)
    v = cp.array([4.0, 6.0], dtype=cp.float32)
    res = distance._manhattan_cupy(u, v)
    assert cp.allclose(res, cp.sum(cp.abs(u - v)))


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_minkowski_cupy():
    u = cp.array([1.0, 2.0], dtype=cp.float32)
    v = cp.array([4.0, 6.0], dtype=cp.float32)
    res = distance._minkowski_cupy(u, v, p=3)
    expected = cp.power(cp.sum(cp.abs(u - v) ** 3), 1 / 3)
    assert cp.allclose(res, expected)


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_chebyshev_cupy():
    u = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
    v = cp.array([2.0, 5.0, 7.0], dtype=cp.float32)
    res = distance._chebyshev_cupy(u, v)
    assert cp.allclose(res, cp.max(cp.abs(u - v)))


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_cosine_similarity_cupy():
    u = cp.array([1.0, 0.0], dtype=cp.float32)
    v = cp.array([0.0, 1.0], dtype=cp.float32)
    res = distance._cosine_similarity_cupy(u, v)
    assert cp.allclose(res, 0.0)


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_mahalanobis_cupy():
    u = cp.array([1.0, 2.0], dtype=cp.float32)
    v = cp.array([2.0, 4.0], dtype=cp.float32)
    VI = cp.eye(2, dtype=cp.float32)
    res = distance._mahalanobis_cupy(u, v, VI)
    assert cp.allclose(res, cp.linalg.norm(u - v))


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_hamming_cupy():
    u = cp.array([1, 0, 1, 1], dtype=cp.int32)
    v = cp.array([1, 1, 0, 1], dtype=cp.int32)
    res = distance._hamming_cupy(u, v)
    expected = cp.mean(u != v)
    assert cp.allclose(res, expected)


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_jaccard_cupy_nonempty_and_empty():
    u = cp.array([1, 0, 1], dtype=cp.int32)
    v = cp.array([1, 1, 0], dtype=cp.int32)
    res = distance._jaccard_cupy(u, v)
    expected = cp.sum(u & v) / cp.sum(u | v)
    assert cp.allclose(res, expected)

    # edge-case union==0
    u0 = cp.array([0, 0, 0], dtype=cp.int32)
    v0 = cp.array([0, 0, 0], dtype=cp.int32)
    res0 = distance._jaccard_cupy(u0, v0)
    assert res0 == 0.0


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_pairwise_euclidean_cupy():
    X = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
    Y = cp.array([[5.0, 6.0]], dtype=cp.float32)
    res = distance._pairwise_euclidean_cupy(X, Y)
    assert res.shape == (2, 1)
