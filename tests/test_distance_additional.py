# tests/test_distance_additional.py
"""
* Copyright (c) [2025] [Sergii Kavun]
*
* This software is dual-licensed:
* - PolyForm Noncommercial 1.0.0 (default)
* - Commercial license available
*
* See LICENSE for details
*
"""

import types

import numpy as np
import pytest

import distance
from distance import CUPY_AVAILABLE, NUMBA_AVAILABLE

# --- Helpers ---------------------------------------------------------------


def make_random_matrix(n_rows=8, n_cols=3):
    return np.random.RandomState(0).rand(n_rows, n_cols).astype(np.float32)


# --- 1) Jaccard edge: union == 0 for numba and cupy implementations -----------
@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is not available")
def test_jaccard_numba_union_zero():
    u = np.array([0, 0, 0], dtype=np.float32)
    v = np.array([0, 0, 0], dtype=np.float32)
    assert distance._jaccard_numba(u, v) == 0.0


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy is not available")
def test_jaccard_cupy_union_zero():
    cp = getattr(distance, "cp", None)
    assert cp is not None  # safety
    u = cp.array([0, 0, 0], dtype=cp.float32)
    v = cp.array([0, 0, 0], dtype=cp.float32)
    assert distance._jaccard_cupy(u, v) == 0.0


# --- 2) _dispatch_v2v fallback: if backend function missing, fallback to numpy --
def test_dispatch_v2v_fallback_to_numpy(monkeypatch):
    # Let's establish that numba is available, but the _manhattan_numba function has been removed:
    dist = distance.Distance(gpu_threshold=10_000)
    monkeypatch.setattr(dist, "gpu_available", False)
    monkeypatch.setattr(dist, "numba_available", True)

    # Remove the numba version from the global namespace (if any)
    monkeypatch.delitem(distance.__dict__, "_manhattan_numba", raising=False)

    # We check that the manhattan call uses the numpy implementation as a fallback.
    u = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    expected = np.sum(np.abs(u - v))
    assert np.isclose(dist.manhattan(u, v), expected)


# --- 3) pairwise_euclidean: cover a branch when the cupy version is missing --
def test_pairwise_euclidean_fallback_when_cupy_missing(monkeypatch, mocker):
    dist = distance.Distance(gpu_threshold=1)
    X = make_random_matrix(10, 4)
    Y = make_random_matrix(12, 4)

    # Force gpu selection on first dispatch
    monkeypatch.setattr(dist, "gpu_available", True)

    fake_cp = types.SimpleNamespace(
        asarray=lambda x: x, asnumpy=lambda x: x, ndarray=np.ndarray
    )
    mocker.patch.dict(distance.__dict__, {"cp": fake_cp})

    monkeypatch.delitem(distance.__dict__, "_pairwise_euclidean_cupy", raising=False)

    mock_numpy = mocker.patch(
        "distance._pairwise_euclidean_numpy", return_value=np.full((10, 12), 777.0)
    )

    res = dist.pairwise_euclidean(X, Y, force_cpu=False)
    mock_numpy.assert_called_once()
    assert np.allclose(res, 777.0)


# --- 4) _dispatch_m2m returns use_gpu True when X.size >= threshold ------------
def test_dispatch_m2m_use_gpu_flag(monkeypatch):
    dist = distance.Distance(gpu_threshold=4)  # very small threshold
    X = np.ones((2, 4), dtype=np.float32)  # size = 8 >= threshold
    Y = np.ones((3, 4), dtype=np.float32)

    # Fake cp (to prevent cp.asarray from breaking)
    fake_cp = types.SimpleNamespace(asarray=lambda x: x)
    monkeypatch.setattr(distance, "cp", fake_cp, raising=False)

    # We will suppress GPU = True
    monkeypatch.setattr(dist, "gpu_available", True)

    func, Xc, Yc, use_gpu = dist._dispatch_m2m(
        "pairwise_euclidean", X, Y, force_cpu=False
    )
    assert use_gpu is True


# --- 5) haversine edge cases ------------------------------------------------
def test_haversine_same_point_zero():
    dist = distance.Distance()
    res = dist.haversine(0.0, 0.0, 0.0, 0.0)
    assert pytest.approx(res, abs=1e-6) == 0.0


def test_haversine_antipodal():
    dist = distance.Distance()
    # (0,0) and (0,180) are antipodes -> distance = pi * R
    R = 6371.0
    res = dist.haversine(0.0, 0.0, 0.0, 180.0, radius=R)
    assert pytest.approx(res, rel=1e-3) == np.pi * R
