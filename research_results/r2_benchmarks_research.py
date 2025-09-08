import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

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
    from sklearn.metrics import r2_score as sklearn_r2_score
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
# R² SCORE IMPLEMENTATIONS IN DIFFERENT FRAMEWORKS
# ============================================================================

# 1. PyTorch implementation
def r2_pytorch_cpu(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """R² score in PyTorch on CPU"""
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return (1 - ss_res / ss_tot).item()

def r2_pytorch_gpu(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """R² score in PyTorch on GPU"""
    if torch.cuda.is_available():
        y_true_gpu = y_true.cuda()
        y_pred_gpu = y_pred.cuda()
        ss_res = torch.sum((y_true_gpu - y_pred_gpu) ** 2)
        ss_tot = torch.sum((y_true_gpu - torch.mean(y_true_gpu)) ** 2)
        result = 1 - ss_res / ss_tot
        return result.cpu().item()
    else:
        return r2_pytorch_cpu(y_true, y_pred)

# 2. TensorFlow implementation
def r2_tensorflow_cpu(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """R² score in TensorFlow on CPU"""
    with tf.device('/CPU:0'):
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
        ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        return (1 - ss_res / ss_tot).numpy()

def r2_tensorflow_gpu(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """R² score in TensorFlow on GPU"""
    if tf.config.list_physical_devices('GPU'):
        with tf.device('/GPU:0'):
            ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
            ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
            return (1 - ss_res / ss_tot).numpy()
    else:
        return r2_tensorflow_cpu(y_true, y_pred)

# 3. Scikit-learn implementation
def r2_sklearn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² score using sklearn"""
    return sklearn_r2_score(y_true, y_pred)

# 4. Numba implementations
if NUMBA_AVAILABLE:
    @jit(nopython=True, fastmath=True)
    def r2_numba_basic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Basic Numba implementation R² score"""
        n = len(y_true)
        ss_res = 0.0
        ss_tot = 0.0
        mean_true = 0.0
        
        # Calculate mean
        for i in range(n):
            mean_true += y_true[i]
        mean_true /= n
        
        # Calculate sums of squares
        for i in range(n):
            diff_res = y_true[i] - y_pred[i]
            diff_tot = y_true[i] - mean_true
            ss_res += diff_res * diff_res
            ss_tot += diff_tot * diff_tot
        
        return 1.0 - ss_res / ss_tot
    
    @jit(nopython=True, parallel=True, fastmath=True)
    def r2_numba_parallel(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Parallel Numba implementation R² score"""
        n = len(y_true)
        
        # Calculate mean (first pass)
        mean_true = 0.0
        for i in range(n):
            mean_true += y_true[i]
        mean_true /= n
        
        # Calculate sums of squares (parallel)
        ss_res = 0.0
        ss_tot = 0.0
        for i in prange(n):
            diff_res = y_true[i] - y_pred[i]
            diff_tot = y_true[i] - mean_true
            ss_res += diff_res * diff_res
            ss_tot += diff_tot * diff_tot
        
        return 1.0 - ss_res / ss_tot
    
    @jit(nopython=True, fastmath=True)
    def r2_numba_vectorized(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Vectorized Numba implementation"""
        mean_true = np.mean(y_true)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - mean_true) ** 2)
        return 1.0 - ss_res / ss_tot

# 5. Pure NumPy for comparison
def r2_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Basic NumPy implementation"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def r2_numpy_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Manual implementation without built-in functions"""
    n = len(y_true)
    mean_true = np.sum(y_true) / n
    
    ss_res = 0.0
    ss_tot = 0.0
    for i in range(n):
        ss_res += (y_true[i] - y_pred[i]) ** 2
        ss_tot += (y_true[i] - mean_true) ** 2
    
    return 1 - ss_res / ss_tot

# 6. Optimized NumPy implementation
def r2_numpy_optimized(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Optimized NumPy implementation with fewer operations"""
    mean_true = y_true.mean()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - mean_true) ** 2).sum()
    return 1 - ss_res / ss_tot

# ============================================================================
# BENCHMARKING FUNCTIONS
# ============================================================================

def generate_data(size: int, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
    """Generating test data with correlation for meaningful R²"""
    np.random.seed(42)  # For reproducibility
    x = np.linspace(0, 10, size).astype(dtype)
    y_true = (2 * x + 1 + 0.5 * np.random.randn(size)).astype(dtype)
    y_pred = (2 * x + 1 + 0.1 * np.random.randn(size)).astype(dtype)
    return y_true, y_pred

def benchmark_function(func, y_true, y_pred, n_runs: int = 100) -> Tuple[float, float]:
    """Benchmark functions with time measurement"""
    times = []
    result_value = None
    
    # Warm-up
    for _ in range(3):
        try:
            result_value = func(y_true, y_pred)
        except Exception as e:
            return float('inf'), 0.0, 0.0
    
    # Measurements
    for _ in range(n_runs):
        start_time = time.perf_counter()
        try:
            result = func(y_true, y_pred)
        except Exception as e:
            return float('inf'), 0.0, 0.0
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return np.mean(times), np.std(times), result_value

def run_comprehensive_benchmark():
    """Comprehensive benchmark of all implementations"""
    sizes = [1000, 10000, 100000, 1000000, 10000000]
    results = {}
    accuracy_results = {}
    
    print("🚀 COMPREHENSIVE BENCHMARK OF R² SCORE IMPLEMENTATIONS")
    print("=" * 80)
    
    for size in sizes:
        print(f"\n📊 Data size: {size:,} elements")
        print("-" * 50)
        
        y_true_np, y_pred_np = generate_data(size)
        size_results = {}
        accuracy_info = {}
        
        # NumPy baseline
        mean_time, std_time, result = benchmark_function(r2_numpy, y_true_np, y_pred_np)
        size_results['NumPy'] = mean_time
        accuracy_info['NumPy'] = result
        print(f"NumPy:           {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms → R²: {result:.6f}")
        
        # NumPy manual
        mean_time, std_time, result = benchmark_function(r2_numpy_manual, y_true_np, y_pred_np)
        size_results['NumPy Manual'] = mean_time
        accuracy_info['NumPy Manual'] = result
        print(f"NumPy Manual:    {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms → R²: {result:.6f}")
        
        # NumPy optimized
        mean_time, std_time, result = benchmark_function(r2_numpy_optimized, y_true_np, y_pred_np)
        size_results['NumPy Optimized'] = mean_time
        accuracy_info['NumPy Optimized'] = result
        print(f"NumPy Optimized: {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms → R²: {result:.6f}")
        
        # Sklearn
        if SKLEARN_AVAILABLE:
            mean_time, std_time, result = benchmark_function(r2_sklearn, y_true_np, y_pred_np)
            size_results['Sklearn'] = mean_time
            accuracy_info['Sklearn'] = result
            print(f"Sklearn:         {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms → R²: {result:.6f}")
        
        # Numba implementation
        if NUMBA_AVAILABLE:
            mean_time, std_time, result = benchmark_function(r2_numba_basic, y_true_np, y_pred_np)
            size_results['Numba Basic'] = mean_time
            accuracy_info['Numba Basic'] = result
            print(f"Numba Basic:     {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms → R²: {result:.6f}")
            
            mean_time, std_time, result = benchmark_function(r2_numba_parallel, y_true_np, y_pred_np)
            size_results['Numba Parallel'] = mean_time
            accuracy_info['Numba Parallel'] = result
            print(f"Numba Parallel:  {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms → R²: {result:.6f}")
            
            mean_time, std_time, result = benchmark_function(r2_numba_vectorized, y_true_np, y_pred_np)
            size_results['Numba Vector'] = mean_time
            accuracy_info['Numba Vector'] = result
            print(f"Numba Vector:    {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms → R²: {result:.6f}")
        
        # PyTorch implementations
        if PYTORCH_AVAILABLE:
            y_true_torch = torch.from_numpy(y_true_np)
            y_pred_torch = torch.from_numpy(y_pred_np)
            
            mean_time, std_time, result = benchmark_function(r2_pytorch_cpu, y_true_torch, y_pred_torch)
            size_results['PyTorch CPU'] = mean_time
            accuracy_info['PyTorch CPU'] = result
            print(f"PyTorch CPU:     {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms → R²: {result:.6f}")
            
            if torch.cuda.is_available():
                mean_time, std_time, result = benchmark_function(r2_pytorch_gpu, y_true_torch, y_pred_torch)
                size_results['PyTorch GPU'] = mean_time
                accuracy_info['PyTorch GPU'] = result
                print(f"PyTorch GPU:     {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms → R²: {result:.6f}")
        
        # TensorFlow implementation
        if TENSORFLOW_AVAILABLE:
            y_true_tf = tf.constant(y_true_np)
            y_pred_tf = tf.constant(y_pred_np)
            
            mean_time, std_time, result = benchmark_function(r2_tensorflow_cpu, y_true_tf, y_pred_tf)
            size_results['TensorFlow CPU'] = mean_time
            accuracy_info['TensorFlow CPU'] = result
            print(f"TensorFlow CPU:  {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms → R²: {result:.6f}")
            
            if tf.config.list_physical_devices('GPU'):
                mean_time, std_time, result = benchmark_function(r2_tensorflow_gpu, y_true_tf, y_pred_tf)
                size_results['TensorFlow GPU'] = mean_time
                accuracy_info['TensorFlow GPU'] = result
                print(f"TensorFlow GPU:  {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms → R²: {result:.6f}")
        
        results[size] = size_results
        accuracy_results[size] = accuracy_info
        
        # We show the best result for this size
        if size_results:
            best_method = min(size_results, key=size_results.get)
            best_time = size_results[best_method]
            print(f"\n🏆 Best result: {best_method} ({best_time*1000:.3f}ms)")
    
    return results, accuracy_results

def plot_results(results: Dict, accuracy_results: Dict):
    """Plot benchmark results"""
    if not results:
        return
        
    sizes = list(results.keys())
    methods = set()
    for size_results in results.values():
        methods.update(size_results.keys())
    
    methods = sorted(list(methods))
    
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Execution time
    plt.subplot(2, 2, 1)
    for method in methods:
        times = []
        valid_sizes = []
        for size in sizes:
            if method in results[size] and results[size][method] != float('inf'):
                times.append(results[size][method] * 1000)  # in milliseconds
                valid_sizes.append(size)
        
        if times:
            plt.loglog(valid_sizes, times, marker='o', label=method, linewidth=2)
    
    plt.xlabel('Data Size')
    plt.ylabel('Execution Time (ms)')
    plt.title('R² Score Execution Time vs Data Size')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Relative performance to NumPy
    plt.subplot(2, 2, 2)
    for method in methods:
        if method == 'NumPy':
            continue
        speedups = []
        valid_sizes = []
        for size in sizes:
            if (method in results[size] and 'NumPy' in results[size] and 
                results[size][method] != float('inf') and results[size]['NumPy'] != float('inf')):
                speedup = results[size]['NumPy'] / results[size][method]
                speedups.append(speedup)
                valid_sizes.append(size)
        
        if speedups:
            plt.semilogx(valid_sizes, speedups, marker='s', label=method, linewidth=2)
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='NumPy baseline')
    plt.xlabel('Data Size')
    plt.ylabel('Speedup vs NumPy')
    plt.title('Relative Performance')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy comparison
    plt.subplot(2, 2, 3)

    if accuracy_results and any(accuracy_results.values()):
        # Выбираем референсный метод (Sklearn)
        ref_method = 'Sklearn' if 'Sklearn' in methods else 'NumPy'
        
        print(f"Using {ref_method} as reference for accuracy comparison")
        
        # Собираем данные об ошибках
        all_accuracy_data = {}
        for method in methods:
            if method == ref_method:
                continue  # Пропускаем референсный метод
                
            method_errors = []
            method_sizes = []
            for size in sizes:
                if (size in accuracy_results and 
                    method in accuracy_results[size] and 
                    ref_method in accuracy_results[size] and
                    accuracy_results[size][method] is not None and
                    accuracy_results[size][ref_method] is not None):
                    
                    error = abs(accuracy_results[size][method] - accuracy_results[size][ref_method])
                    # Заменяем нулевые ошибки на очень маленькое значение для логарифмической шкалы
                    if error == 0.0:
                        error = 1e-20  # Минимальное значение для отображения
                    method_errors.append(error)
                    method_sizes.append(size)
            
            if method_errors:
                all_accuracy_data[method] = (method_sizes, method_errors)
                print(f"Method {method}: {len(method_errors)} data points, min error: {min(method_errors):.2e}")
        
        # Отображаем все методы с данными
        if all_accuracy_data:
            colors = plt.cm.tab10(np.linspace(0, 1, len(all_accuracy_data)))
            
            # Создаем обычный plot вместо semilogy для лучшего отображения
            for i, (method, (method_sizes, method_errors)) in enumerate(all_accuracy_data.items()):
                plt.plot(method_sizes, method_errors, 
                      marker='^', label=method, 
                      color=colors[i % len(colors)],
                      linewidth=2, markersize=6)
            
            plt.xlabel('Data Size')
            plt.ylabel(f'Absolute Error vs {ref_method}')
            plt.title('Numerical Accuracy Comparison')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # Устанавливаем логарифмическую шкалу вручную
            
            # Добавляем информацию о минимальной/максимальной ошибке
            all_errors = []
            for method_sizes, method_errors in all_accuracy_data.values():
                all_errors.extend(method_errors)
            
            if all_errors:
                min_error = min(all_errors)
                max_error = max(all_errors)
                plt.text(0.02, 0.98, f'Error range: {min_error:.2e} - {max_error:.2e}',
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                # Устанавливаем разумные пределы для оси Y
                plt.ylim(min_error * 0.1, max_error * 10)
        else:
            plt.text(0.5, 0.5, 'No error data available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Numerical Accuracy Comparison')
    else:
        plt.text(0.5, 0.5, 'No accuracy data available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Numerical Accuracy Comparison')
    
    # Plot 4: Performance for large data (>100k) - ИСПРАВЛЕННЫЙ КОД
    plt.subplot(2, 2, 4)
    large_sizes = [s for s in sizes if s >= 100000]
    
    if large_sizes:
        # Создаем основную ось
        ax1 = plt.gca()
        ax2 = ax1.twinx()  # Создаем вторую ось для NumPy Manual
        
        method_times = {}
        numpy_manual_times = {}
        
        for method in methods:
            times = []
            for size in large_sizes:
                if (method in results[size] and results[size][method] != float('inf') and
                    results[size][method] > 0):
                    times.append(results[size][method] * 1000)
            
            if times:
                if method == 'NumPy Manual':
                    numpy_manual_times[method] = np.mean(times)
                else:
                    method_times[method] = np.mean(times)
        
        # Сортируем методы по времени (исключая NumPy Manual)
        if method_times:
            methods_sorted = sorted(method_times.keys(), key=lambda x: method_times[x])
            times_sorted = [method_times[m] for m in methods_sorted]
            
            # Создаем барплот для всех методов кроме NumPy Manual
            bars = ax1.bar(range(len(methods_sorted)), times_sorted, alpha=0.8)
            ax1.set_xticks(range(len(methods_sorted)))
            ax1.set_xticklabels(methods_sorted, rotation=45, ha='right')
            ax1.set_ylabel('Average Time (ms) - Main Methods', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # Цвета для баров
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        # Добавляем NumPy Manual на вторую ось
        if numpy_manual_times:
            numpy_manual_methods = list(numpy_manual_times.keys())
            numpy_manual_times_list = [numpy_manual_times[m] for m in numpy_manual_methods]
            
            # Добавляем точку для NumPy Manual
            ax2.plot(len(methods_sorted) + 0.5, numpy_manual_times_list[0], 
                    'ro', markersize=10, label='NumPy Manual')
            ax2.set_ylabel('NumPy Manual Time (ms)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Устанавливаем лимиты для второй оси
            if numpy_manual_times_list:
                ax2.set_ylim(0, numpy_manual_times_list[0] * 1.1)
        
        plt.title('Performance on Large Data (>100K)\nNumPy Manual shown on right axis')
        ax1.grid(True, alpha=0.3)
        
        # Добавляем легенду
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def detailed_analysis():
    """Detailed performance analysis"""
    print("\n" + "="*80)
    print("📈 DETAILED PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Analysis for different data types
    dtypes = [np.float32, np.float64]
    size = 1000000
    
    print(f"\n🔍 Data type analysis (size: {size:,})")
    print("-" * 50)
    
    for dtype in dtypes:
        print(f"\nData type: {dtype}")
        y_true, y_pred = generate_data(size, dtype)
        
        # NumPy
        time_np, _, result_np = benchmark_function(r2_numpy, y_true, y_pred, n_runs=50)
        print(f"  NumPy:     {time_np*1000:.3f}ms → R²: {result_np:.6f}")
        
        # Numba
        if NUMBA_AVAILABLE:
            time_numba, _, result_numba = benchmark_function(r2_numba_basic, y_true, y_pred, n_runs=50)
            speedup = time_np / time_numba
            print(f"  Numba:     {time_numba*1000:.3f}ms (speedup: {speedup:.2f}x) → R²: {result_numba:.6f}")
    
    # Numba compilation overhead analysis
    if NUMBA_AVAILABLE:
        print(f"\n⚡ Numba compilation overhead analysis")
        print("-" * 50)
        
        y_true, y_pred = generate_data(10000)
        
        # First call (with compilation)
        start_time = time.perf_counter()
        r2_numba_basic(y_true, y_pred)
        compile_time = time.perf_counter() - start_time
        
        # Subsequent calls
        start_time = time.perf_counter()
        r2_numba_basic(y_true, y_pred)
        runtime = time.perf_counter() - start_time
        
        print(f"Compilation time: {compile_time*1000:.3f}ms")
        print(f"Execution time: {runtime*1000:.3f}ms")
        print(f"Compilation overhead: {compile_time/runtime:.1f}x")
    
    # GPU analysis
    if PYTORCH_AVAILABLE and torch.cuda.is_available():
        print(f"\n🖥️  GPU analysis")
        print("-" * 50)
        
        sizes_gpu = [1000000, 10000000]
        
        for size in sizes_gpu:
            y_true, y_pred = generate_data(size)
            y_true_torch = torch.from_numpy(y_true)
            y_pred_torch = torch.from_numpy(y_pred)
            
            # CPU time
            time_cpu, _, result_cpu = benchmark_function(r2_pytorch_cpu, y_true_torch, y_pred_torch, n_runs=20)
            
            # GPU time
            time_gpu, _, result_gpu = benchmark_function(r2_pytorch_gpu, y_true_torch, y_pred_torch, n_runs=20)
            
            speedup = time_cpu / time_gpu if time_gpu > 0 else 0
            print(f"Size {size:,}: CPU {time_cpu*1000:.3f}ms, GPU {time_gpu*1000:.3f}ms (speedup: {speedup:.2f}x)")
            print(f"  CPU R²: {result_cpu:.6f}, GPU R²: {result_gpu:.6f}")

def system_info():
    """System information for benchmark"""
    print("🖥️  SYSTEM INFORMATION")
    print("="*50)
    
    import platform
    import psutil
    
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    if PYTORCH_AVAILABLE and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    print(f"\nLibrary versions:")
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
    results, accuracy_results = run_comprehensive_benchmark()
    
    # Detailed analysis
    detailed_analysis()
    
    # Plot results
    try:
        plot_results(results, accuracy_results)
    except ImportError:
        print("\n⚠️ Matplotlib is not installed, plots will not be generated")
    
if __name__ == "__main__":
    main()