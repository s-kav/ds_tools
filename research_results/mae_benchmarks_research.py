import time
import warnings
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# Imports for different libraries
try:
    import torch

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not installed")

try:
    import tensorflow as tf

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not installed")

try:
    from sklearn.metrics import mean_absolute_error

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not installed")

try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not installed")

# ============================================================================
# MAE IMPLEMENTATIONS IN DIFFERENT FRAMEWORKS
# ============================================================================


# 1. PyTorch implementation
def mae_pytorch_cpu(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """MAE in PyTorch on CPU"""
    return torch.mean(torch.abs(y_true - y_pred)).item()


def mae_pytorch_gpu(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """MAE in PyTorch on GPU"""
    if torch.cuda.is_available():
        y_true_gpu = y_true.cuda()
        y_pred_gpu = y_pred.cuda()
        result = torch.mean(torch.abs(y_true_gpu - y_pred_gpu))
        return result.cpu().item()
    else:
        return mae_pytorch_cpu(y_true, y_pred)


# 2. TensorFlow implementation
def mae_tensorflow_cpu(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """MAE in TensorFlow on CPU"""
    with tf.device("/CPU:0"):
        return tf.reduce_mean(tf.abs(y_true - y_pred)).numpy()


def mae_tensorflow_gpu(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """MAE in TensorFlow on GPU"""
    if tf.config.list_physical_devices("GPU"):
        with tf.device("/GPU:0"):
            return tf.reduce_mean(tf.abs(y_true - y_pred)).numpy()
    else:
        return mae_tensorflow_cpu(y_true, y_pred)


# 3. Scikit-learn implementation
def mae_sklearn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE using sklearn"""
    return mean_absolute_error(y_true, y_pred)


# 4. Numba implementations
if NUMBA_AVAILABLE:

    @jit(nopython=True, fastmath=True)
    def mae_numba_basic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Basic Numba implementation MAE"""
        return np.mean(np.abs(y_true - y_pred))

    @jit(nopython=True, parallel=True, fastmath=True)
    def mae_numba_parallel(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Parallel Numba implementation MAE"""
        n = len(y_true)
        total = 0.0
        for i in prange(n):
            total += abs(y_true[i] - y_pred[i])
        return total / n

    @jit(nopython=True, fastmath=True)
    def mae_numba_vectorized(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Vectorized Numba implementation"""
        diff = y_true - y_pred
        abs_diff = np.abs(diff)
        return np.mean(abs_diff)


# 5. Pure NumPy for comparison
def mae_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Basic NumPy implementation"""
    return np.mean(np.abs(y_true - y_pred))


def mae_numpy_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Manual implementation without built-in functions"""
    return np.sum(np.abs(y_true - y_pred)) / len(y_true)


# ============================================================================
# BENCHMARKING FUNCTIONS
# ============================================================================


def generate_data(size: int, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
    """Generating test data"""
    np.random.seed(42)  # For reproducibility
    y_true = np.random.randn(size).astype(dtype)
    y_pred = y_true + 0.1 * np.random.randn(size).astype(dtype)
    return y_true, y_pred


def benchmark_function(func, y_true, y_pred, n_runs: int = 100) -> Tuple[float, float]:
    """Benchmark functions with time measurement"""
    times = []

    # Warm-up
    for _ in range(3):
        try:
            func(y_true, y_pred)
        except Exception:
            return float("inf"), 0.0

    # Measurements
    for _ in range(n_runs):
        start_time = time.perf_counter()
        try:
            func(y_true, y_pred)
        except Exception:
            return float("inf"), 0.0
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    return np.mean(times), np.std(times)


def run_comprehensive_benchmark():
    """Comprehensive benchmark of all implementations"""
    # sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000]
    results = {}

    print("🚀 COMPREHENSIVE BENCHMARK OF MAE IMPLEMENTATIONS")
    print("=" * 80)

    for size in sizes:
        print(f"\n📊 Data size: {size:,} elements")
        print("-" * 50)

        y_true_np, y_pred_np = generate_data(size)
        size_results = {}

        # NumPy baseline
        mean_time, std_time = benchmark_function(mae_numpy, y_true_np, y_pred_np)
        size_results["NumPy"] = mean_time
        print(f"NumPy:           {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms")

        # NumPy manual
        mean_time, std_time = benchmark_function(mae_numpy_manual, y_true_np, y_pred_np)
        size_results["NumPy Manual"] = mean_time
        print(f"NumPy Manual:    {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms")

        # Sklearn
        if SKLEARN_AVAILABLE:
            mean_time, std_time = benchmark_function(mae_sklearn, y_true_np, y_pred_np)
            size_results["Sklearn"] = mean_time
            print(f"Sklearn:         {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms")

        # Numba implementation
        if NUMBA_AVAILABLE:
            mean_time, std_time = benchmark_function(
                mae_numba_basic, y_true_np, y_pred_np
            )
            size_results["Numba Basic"] = mean_time
            print(f"Numba Basic:     {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms")

            mean_time, std_time = benchmark_function(
                mae_numba_parallel, y_true_np, y_pred_np
            )
            size_results["Numba Parallel"] = mean_time
            print(f"Numba Parallel:  {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms")

            mean_time, std_time = benchmark_function(
                mae_numba_vectorized, y_true_np, y_pred_np
            )
            size_results["Numba Vector"] = mean_time
            print(f"Numba Vector:    {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms")

        # PyTorch implementations
        if PYTORCH_AVAILABLE:
            y_true_torch = torch.from_numpy(y_true_np)
            y_pred_torch = torch.from_numpy(y_pred_np)

            mean_time, std_time = benchmark_function(
                mae_pytorch_cpu, y_true_torch, y_pred_torch
            )
            size_results["PyTorch CPU"] = mean_time
            print(f"PyTorch CPU:     {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms")

            if torch.cuda.is_available():
                mean_time, std_time = benchmark_function(
                    mae_pytorch_gpu, y_true_torch, y_pred_torch
                )
                size_results["PyTorch GPU"] = mean_time
                print(
                    f"PyTorch GPU:     {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms"
                )

        # TensorFlow implementation
        if TENSORFLOW_AVAILABLE:
            y_true_tf = tf.constant(y_true_np)
            y_pred_tf = tf.constant(y_pred_np)

            mean_time, std_time = benchmark_function(
                mae_tensorflow_cpu, y_true_tf, y_pred_tf
            )
            size_results["TensorFlow CPU"] = mean_time
            print(f"TensorFlow CPU:  {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms")

            if tf.config.list_physical_devices("GPU"):
                mean_time, std_time = benchmark_function(
                    mae_tensorflow_gpu, y_true_tf, y_pred_tf
                )
                size_results["TensorFlow GPU"] = mean_time
                print(
                    f"TensorFlow GPU:  {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms"
                )

        results[size] = size_results

        # We show the best result for this size
        if size_results:
            best_method = min(size_results, key=size_results.get)
            best_time = size_results[best_method]
            print(f"\n🏆 Best result: {best_method} ({best_time*1000:.3f}ms)")

    return results


def plot_results(results: Dict):
    """Plot benchmark results"""
    if not results:
        return

    sizes = list(results.keys())
    methods = set()
    for size_results in results.values():
        methods.update(size_results.keys())

    methods = sorted(list(methods))

    plt.figure(figsize=(15, 10))

    # Plot 1: Execution time
    plt.subplot(2, 2, 1)
    for method in methods:
        times = []
        valid_sizes = []
        for size in sizes:
            if method in results[size] and results[size][method] != float("inf"):
                times.append(results[size][method] * 1000)  # in milliseconds
                valid_sizes.append(size)

        if times:
            plt.loglog(valid_sizes, times, marker="o", label=method, linewidth=2)

    plt.xlabel("Data Size")
    plt.ylabel("Execution Time (ms)")
    plt.title("MAE Execution Time vs Data Size")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Relative performance to NumPy
    plt.subplot(2, 2, 2)
    for method in methods:
        if method == "NumPy":
            continue
        speedups = []
        valid_sizes = []
        for size in sizes:
            if (
                method in results[size]
                and "NumPy" in results[size]
                and results[size][method] != float("inf")
                and results[size]["NumPy"] != float("inf")
            ):
                speedup = results[size]["NumPy"] / results[size][method]
                speedups.append(speedup)
                valid_sizes.append(size)

        if speedups:
            plt.semilogx(valid_sizes, speedups, marker="s", label=method, linewidth=2)

    plt.axhline(y=1, color="black", linestyle="--", alpha=0.5, label="NumPy baseline")
    plt.xlabel("Data Size")
    plt.ylabel("Speedup vs NumPy")
    plt.title("Relative Performance")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Performance for large data (>100k)
    plt.subplot(2, 2, 3)
    large_sizes = [s for s in sizes if s >= 100000]
    if large_sizes:
        method_times = {}
        for method in methods:
            times = []
            for size in large_sizes:
                if method in results[size] and results[size][method] != float("inf"):
                    times.append(results[size][method] * 1000)
            if times:
                method_times[method] = np.mean(times)

        if method_times:
            methods_sorted = sorted(method_times.keys(), key=lambda x: method_times[x])
            times_sorted = [method_times[m] for m in methods_sorted]

            bars = plt.bar(range(len(methods_sorted)), times_sorted)
            plt.xticks(range(len(methods_sorted)), methods_sorted, rotation=45)
            plt.ylabel("Average Time (ms)")
            plt.title("Performance on Large Data (>100K)")

            # Color bars
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)

    # Plot 4: Memory efficiency analysis (theoretical)
    plt.subplot(2, 2, 4)
    memory_efficiency = {
        "NumPy": 1.0,
        "NumPy Manual": 1.0,
        "Sklearn": 1.1,  # small overhead
        "Numba Basic": 0.8,  # more efficient
        "Numba Parallel": 0.9,  # parallelism requires overhead
        "Numba Vector": 0.8,
        "PyTorch CPU": 1.3,  # tensor overhead
        "PyTorch GPU": 0.6,  # GPU memory is faster
        "TensorFlow CPU": 1.5,  # larger overhead
        "TensorFlow GPU": 0.7,
    }

    available_methods = [m for m in memory_efficiency.keys() if m in methods]
    efficiency_values = [memory_efficiency[m] for m in available_methods]

    bars = plt.bar(range(len(available_methods)), efficiency_values)
    plt.xticks(range(len(available_methods)), available_methods, rotation=45)
    plt.ylabel("Relative Memory Efficiency")
    plt.title("Theoretical Memory Efficiency")
    plt.axhline(y=1, color="red", linestyle="--", alpha=0.7, label="NumPy baseline")

    # Color bars by efficiency
    colors = [
        "green" if v < 1 else "orange" if v < 1.2 else "red" for v in efficiency_values
    ]
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    plt.tight_layout()
    plt.show()


def detailed_analysis():
    """Detailed performance analysis"""
    print("\n" + "=" * 80)
    print("📈 DETAILED PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Analysis for different data types
    dtypes = [np.float32, np.float64]
    size = 1000000

    print(f"\n🔍 Data type analysis (size: {size:,})")
    print("-" * 50)

    for dtype in dtypes:
        print(f"\nData type: {dtype}")
        y_true, y_pred = generate_data(size, dtype)

        # NumPy
        time_np, _ = benchmark_function(mae_numpy, y_true, y_pred, n_runs=50)
        print(f"  NumPy:     {time_np*1000:.3f}ms")

        # Numba
        if NUMBA_AVAILABLE:
            time_numba, _ = benchmark_function(
                mae_numba_basic, y_true, y_pred, n_runs=50
            )
            speedup = time_np / time_numba
            print(f"  Numba:     {time_numba*1000:.3f}ms (speedup: {speedup:.2f}x)")

    # Numba compilation overhead analysis
    if NUMBA_AVAILABLE:
        print("\n⚡ Numba compilation overhead analysis")
        print("-" * 50)

        y_true, y_pred = generate_data(10000)

        # First call (with compilation)
        start_time = time.perf_counter()
        mae_numba_basic(y_true, y_pred)
        compile_time = time.perf_counter() - start_time

        # Subsequent calls
        start_time = time.perf_counter()
        mae_numba_basic(y_true, y_pred)
        runtime = time.perf_counter() - start_time

        print(f"Compilation time: {compile_time*1000:.3f}ms")
        print(f"Execution time: {runtime*1000:.3f}ms")
        print(f"Compilation overhead: {compile_time/runtime:.1f}x")

    # GPU analysis
    if PYTORCH_AVAILABLE and torch.cuda.is_available():
        print("\n🖥️  GPU analysis")
        print("-" * 50)

        sizes_gpu = sizes[-3:]

        for size in sizes_gpu:
            y_true, y_pred = generate_data(size)
            y_true_torch = torch.from_numpy(y_true)
            y_pred_torch = torch.from_numpy(y_pred)

            # CPU time
            time_cpu, _ = benchmark_function(
                mae_pytorch_cpu, y_true_torch, y_pred_torch, n_runs=20
            )

            # GPU time
            time_gpu, _ = benchmark_function(
                mae_pytorch_gpu, y_true_torch, y_pred_torch, n_runs=20
            )

            speedup = time_cpu / time_gpu if time_gpu > 0 else 0
            print(
                f"Size {size:,}: CPU {time_cpu*1000:.3f}ms, GPU {time_gpu*1000:.3f}ms (speedup: {speedup:.2f}x)"
            )


def system_info():
    """System information for benchmark"""
    print("🖥️  SYSTEM INFORMATION")
    print("=" * 50)

    import platform

    import psutil

    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    print(
        f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical"
    )
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

    if PYTORCH_AVAILABLE and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
        )

    print("\nLibrary versions:")
    print(f"  NumPy: {np.__version__}")

    if PYTORCH_AVAILABLE:
        print(f"  PyTorch: {torch.__version__}")
    if TENSORFLOW_AVAILABLE:
        print(f"  TensorFlow: {tf.__version__}")
    if SKLEARN_AVAILABLE:
        import sklearn

        print(f"  Scikit-learn: {sklearn.__version__}")
    if NUMBA_AVAILABLE:
        import numba

        print(f"  Numba: {numba.__version__}")


def main():
    """Main function to run benchmarks"""
    system_info()

    # Main benchmark
    results = run_comprehensive_benchmark()

    # Detailed analysis
    detailed_analysis()

    # Plot results
    try:
        plot_results(results)
    except ImportError:
        print("\n⚠️ Matplotlib is not installed, plots will not be generated")

    # Final recommendations
    print("\n" + "=" * 80)
    print("🎯 FINAL RECOMMENDATIONS")
    print("=" * 80)

    recommendations = """
    📊 GENERAL MAE PERFORMANCE RECOMMENDATIONS:
    
    🥇 FOR MAXIMUM SPEED:
       • Numba (jit) - best choice for CPU computations
       • PyTorch GPU - for very large arrays with GPU
       • Parallel Numba - for multi-core systems
    
    ⚖️ FOR SPEED/SIMPLICITY BALANCE:
       • Sklearn - optimized C implementation
       • NumPy - simplicity and reliability
       • PyTorch CPU - if already using PyTorch
    
    🔄 FOR DIFFERENT DATA SIZES:
       • < 10K elements: Sklearn or NumPy
       • 10K - 1M elements: Numba
       • > 1M elements: Numba Parallel or PyTorch GPU
    
    ⚠️ AVOID:
       • TensorFlow for simple computations (large overhead)
       • GPU for small data (transfer overhead)
       • First Numba call in production (compile overhead)
    
    💡 OPTIMIZATIONS:
       • Use float32 instead of float64 when possible
       • Pre-compile Numba functions in advance
       • Batching for GPU operations
       • Avoid copying data between devices
    """

    print(recommendations)


if __name__ == "__main__":
    sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000]
    main()
