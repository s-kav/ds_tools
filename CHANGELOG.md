# Changelog

All notable changes to this project will be documented in this file.

---
## [2.0.1] - 2025-09-17

### 🐞 Fixed

-   **Fixed a critical packaging bug** that caused the `v2.0.0` wheel on PyPI to be empty (containing only metadata). This resulted in a `ModuleNotFoundError` when trying to import the library after installation. The package now correctly includes all source code. (Fixes #4)

### Internal

-   Significantly improved test coverage for the `metrics` and `distance` modules, adding tests for all backend implementations (NumPy, Numba, CuPy) and numerous edge cases.
-   Refactored the CI pipeline for more robust and reliable test execution.


## [2.0.0] - 2025-09-08

### Version 2.0.0 - Modular Architecture and Performance

This major update marks the transition of the library to a more structured, modular, and high-performance architecture. The main `DSTools` class now acts as an entry point to specialized modules (`metrics`, `distance`, etc.), and calculations have been optimized for CPU (Numba) and GPU (CuPy) use.

## 💥 BREAKING CHANGES (API)

The primary way of calling functions has changed. Methods are now grouped into logical namespaces.

* **Was (v1.x.x):** All methods were called directly from `tools`.
* **Now (v2.0.0):** Methods are called from child modules, such as `tools.metrics` and `tools.distance`, etc.

**Example migration:**
```python
# Old code (v1.x.x)
loss = tools.mse(y_true, y_pred) # Example, such method might not exist

# New code (v2.0.0)
from ds_tools import DSTools
tools = DSTools()
loss = tools.metrics.mse(y_true, y_pred)
dist = tools.distance.euclidean(u, v)

```
## ✨ New Features and Improvements

1. New Metrics module (tools.metrics)
A full-fledged framework for calculating metrics and loss functions has been added:
Implemented functions: mae, mse, rmse, huber_loss, quantile_loss, hinge_loss, log_loss, cross_entropy_loss, triplet_loss.
Gradient calculation: All functions support optional gradient return (return_grad=True), which is useful for custom ML model training cycles.
Real-time monitoring: A system for tracking metrics during training has been added (start_monitoring, update, plot_history).

2. New Distance module (tools.distance)
Added a module for high-performance distance and similarity calculations:
Implemented metrics: euclidean, manhattan, cosine_similarity, minkowski, chebyshev, mahalanobis, haversine, hamming, jaccard.
Matrix calculations: Added pairwise_euclidean, kmeans_distance, knn_distances, radius_neighbors with pure NumPy/CuPy/Numba implementations.

3. Performance optimizations
GPU support (CuPy): Functions in metrics and distance automatically use the GPU for calculations on large data arrays, which significantly speeds up the work.
CPU parallelization (Numba): For systems without a GPU, calculations are automatically parallelized across all CPU cores using Numba.
Smart dispatching: Added force_cpu=True flag and gpu_threshold threshold to flexibly control the choice of the computing backend.

## 🏛️ Architecture and Code Quality
Model refactoring: All Pydantic models (MetricsConfig, CorrelationConfig, etc.) are moved to a separate, clean ds_tools/models.py file.
Modular structure: The main code is divided into logical modules (metrics.py, distance.py, models.py), which simplifies support and further expansion.
Increased test coverage: Added comprehensive tests for new modules, covering all functions, backends, and error handling.

## [1.0.9] - 2025-07-22

### Fixed some issues