# Changelog

All notable changes to this project will be documented in this file.

## [2.3.2] - 2025-12-31

This release introduces advanced mathematical capabilities with hardware acceleration support, fixes critical metric definitions to align with scientific standards, and improves library stability.

### ✨ **New Features**

#### 1. **Cohen's d (Effect Size)**
Added a high-performance implementation of Cohen's d to the `Metrics` class. This metric is essential for quantifying the standardized difference between two group means.

- **Three Backends:**
    - `numpy`: Baseline implementation (universally compatible).
    - `numba`: JIT-compiled CPU optimization for high-speed execution.
    - `cupy`: GPU-accelerated implementation for massive datasets.
- **Features:** Uses pooled standard deviation with bias correction (`ddof=1`) and includes automatic warnings for small sample sizes (`N < 1000`).

#### 2. **Fast Fourier Transform (FFT)**
Added a flexible `fft()` method to the `Metrics` class for 1D signal processing.

- **Custom Numba Algorithm:** Implemented a Radix-2 Recursive Cooley-Tukey algorithm for JIT-compiled execution, including automatic zero-padding to the nearest power of 2.
- **GPU Support:** Full support for CuPy based FFT for heavy workloads.
- **Inverse Transform:** Supports `inverse=True` for IFFT signal reconstruction.
- **Engine Control:** New `engine` parameter allows explicit selection of `'numpy'`, `'numba'`, or `'cupy'` backends.

### 🐛 **Bug Fixes**

- **Kulsinski Metric Parity:** Fixed the calculation logic for the `kulsinski` boolean distance metric in `distance.py` to match the standard definition (Kulczynski 1) used by `scipy.spatial.distance`.
- **Numba Threading Crash:** Removed unsafe modification of `NUMBA_NUM_THREADS` during `Metrics` initialization, resolving a `RuntimeError` when Numba was already initialized elsewhere.
- **FFT Data Types:** Fixed an issue where complex spectra were incorrectly cast to `float32` during Inverse FFT, preventing data loss in the imaginary part and ensuring accurate signal reconstruction.

### ⚡ **Improvements & Refactoring**

- **Code Style:** Refactored `distance.py` and `metrics.py` to resolve multiple PEP 8 compliance issues (specifically E701 multi-statement lines), improving readability.
- **Type Safety:** Enhanced type handling for complex number operations across all backends.
- **Testing:** Added comprehensive test suites for FFT (peak detection, round-trip accuracy) and Cohen's d (magnitude verification).

### 🐞 Critical Fixes (Import & Packaging)

-   **Fixed `ModuleNotFoundError` on Import:** Resolved a critical issue where the library could not be imported after installation (`from ds_tools import DSTools` failed). This was achieved by correcting internal absolute imports to relative imports within the package structure.
-   **Corrected Package Discovery:** Fixed `setup.cfg` and `pyproject.toml` configurations to ensure the `src` directory is correctly mapped to the `ds_tools` package during the build process.

### 📦 Dependencies & Environment

-   **Updated Python Support:**
    -   Added official support for **Python 3.12**.
    -   Dropped support for **Python 3.8** (EOL).
    -   Supported range is now: **3.9, 3.10, 3.11, 3.12**.
-   **New Dependency:** Added `psutil` as a core dependency for CPU core detection in parallel processing.
-   **Streamlined Optional Dependencies:** Clarified installation extras. Use `pip install dscience-tools[performance]` to install `numba` and `cupy` for hardware acceleration.
-   **PEP 621 Adoption:** Centralized all project metadata and dependencies into `pyproject.toml`, making it the single source of truth and removing conflicts with `setup.cfg`.

### ⚙️ CI/CD & Testing Overhaul

-   **Matrix Testing:** Implemented a robust GitHub Actions testing matrix that validates the library across all supported Python versions and all dependency combinations (No Optional Deps, Numba only, CuPy only, All).
-   **Accurate Code Coverage:** Fixed coverage reporting for hardware-dependent code. The CI now correctly merges reports from different environments, reflecting a true test coverage of >81% (covering 100% of CPU-accessible code).
-   **Documentation:** Fixed documentation deployment workflow (GitHub Pages now correctly serves the MkDocs site).

---

## [2.0.1] - 2025-09-17

### 🐞 Fixed

-   **Fixed a critical packaging bug** that caused the `v2.0.0` wheel on PyPI to be empty (containing only metadata). This resulted in a `ModuleNotFoundError` when trying to import the library after installation. The package now correctly includes all source code. (Fixes #4)

### Internal

-   Significantly improved test coverage for the `metrics` and `distance` modules, adding tests for all backend implementations (NumPy, Numba, CuPy) and numerous edge cases.
-   Refactored the CI pipeline for more robust and reliable test execution.

## 🚀 Supported Python Versions Update

To leverage modern language features, improve performance, and ensure long-term maintainability, the range of supported Python versions has been updated.

-   **Dropped Support for Python 3.8:** Python 3.8 has reached its end-of-life for bugfixes and is no longer actively supported. Dropping it allows the library to use newer, more efficient APIs.
-   **Added Support for Python 3.12:** The library is now fully tested and compatible with the latest stable version, Python 3.12.

The officially supported Python versions are now **3.9, 3.10, 3.11, and 3.12**. Users on Python 3.8 are encouraged to upgrade their environment to continue receiving updates for this library.


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