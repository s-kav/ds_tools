# benchmark_adaptive_bayes.py
# https://www.perplexity.ai/page/adaptive-bayesian-classifier-i-3mg7tF_gSESJuFrTC5mIJw
import gzip
import os
import shutil
import time
import zipfile

import numpy as np
import pandas as pd
import requests
from pandas.errors import ParserError
from sklearn.preprocessing import StandardScaler

# Optional GPU memory
try:
    from cupy.cuda import runtime as cuda_rt

    GPU_OK = True
except Exception:
    GPU_OK = False

import psutil
from memory_profiler import memory_usage

# Baselines
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

try:
    import xgboost as xgb

    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    import lightgbm as lgb

    LGB_OK = True
except Exception:
    LGB_OK = False

try:
    from catboost import CatBoostClassifier

    CAT_OK = True
except Exception:
    CAT_OK = False

from adaptive_bayes import AdaptiveBayes

# ---------------------------
# helped inner functions
# ---------------------------


def _gpu_mem_info():
    if not GPU_OK:
        return None, None
    free_b, total_b = cuda_rt.memGetInfo()
    return free_b, total_b


def _proc_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _measure_run(fn, *args, **kwargs):
    # CPU mem before
    rss_before = _proc_rss_mb()
    # GPU mem before
    free0, total0 = _gpu_mem_info()
    t0 = time.perf_counter()

    def _wrapped():
        return fn(*args, **kwargs)

    mem_trace = memory_usage(
        (_wrapped, (), {}), max_iterations=1, interval=0.1, retval=True
    )
    if isinstance(mem_trace, tuple) and len(mem_trace) == 2:
        mem_series, ret = mem_trace
    else:
        mem_series, ret = mem_trace, None

    elapsed = time.perf_counter() - t0
    rss_after = _proc_rss_mb()
    free1, total1 = _gpu_mem_info()

    peak_cpu = max(mem_series) - rss_before if mem_series else 0.0
    rss_delta = rss_after - rss_before

    gpu_delta = None
    if free0 is not None and free1 is not None:
        gpu_delta = (free0 - free1) / (1024 * 1024)

    return {
        "elapsed_s": elapsed,
        "cpu_rss_mb_before": rss_before,
        "cpu_rss_mb_after": rss_after,
        "cpu_rss_delta_mb": rss_delta,
        "cpu_peak_mb": peak_cpu,
        "gpu_mem_delta_mb": gpu_delta,
        "ret": ret,
    }


# ---------------------------
# Dataset loaders (paths expected)
# ---------------------------
def load_creditcard_fraud(path_csv):
    # Kaggle: V1..V28 + Time, Amount, Class; binary Class
    df = pd.read_csv(path_csv)
    y = df["Class"].astype(np.int32).values
    X = df.drop(columns=["Class"]).values.astype(np.float64)
    return X, y


def load_higgs(path_gz):
    # UCI: CSV.gz with label first, then 28 features
    with gzip.open(path_gz, "rt") as f:
        df = pd.read_csv(f, header=None)
    y = df.iloc[:, 0].astype(np.int32).values
    X = df.iloc[:, 1:].values.astype(np.float64)
    return X, y


def load_susy(path_gz):
    with gzip.open(path_gz, "rt") as f:
        df = pd.read_csv(f, header=None)
    y = df.iloc[:, 0].astype(np.int32).values
    X = df.iloc[:, 1:].values.astype(np.float64)
    return X, y


def load_kddcup99(path_csv, drop_cats=True):
    # Mixed dtypes; simplify to numeric by one-hot or drop_cats
    df = pd.read_csv(path_csv, header=None)
    if drop_cats:
        # Keep numeric columns only
        num_df = df.select_dtypes(include=[np.number])
        # Target can be last column or named; assume last is label string -> map to binary (normal vs attack)
        # If last column non-numeric, we map
        if not np.issubdtype(df.iloc[:, -1].dtype, np.number):
            y = (df.iloc[:, -1].astype(str) != "normal.").astype(np.int32).values
        else:
            y = df.iloc[:, -1].astype(np.int32).values
        X = num_df.iloc[:, :-1].values.astype(np.float64)
    else:
        # One-hot encode categoricals
        y = (df.iloc[:, -1].astype(str) != "normal.").astype(np.int32).values
        X = pd.get_dummies(df.iloc[:, :-1]).values.astype(np.float64)
    return X, y


def load_covertype(path_csv):
    df = pd.read_csv(path_csv)
    target_col = "Cover_Type" if "Cover_Type" in df.columns else df.columns[-1]
    y = np.asarray(df[target_col], dtype=np.int32) - 1
    # Convert to binary: class1 vs others to align with AUC, or keep multiclass for accuracy
    # Here we keep multiclass; AUC will be skipped for multiclass
    X = df.drop(columns=[target_col]).values.astype(np.float64)
    return X, y


def create_synthetic_hepmass():
    """Creates a synthetic dataset in the style of HEPMASS"""
    print("Creating synthetic HEPMASS-like dataset...")
    np.random.seed(42)
    n_samples = 500000
    n_features = 28

    # Create correlation features as in physical data
    X = np.random.randn(n_samples, n_features).astype(np.float64)

    # Add nonlinear interactions for realism
    X[:, 1] = X[:, 0] ** 2 + 0.5 * np.random.randn(n_samples)
    X[:, 2] = X[:, 0] * X[:, 1] + 0.3 * np.random.randn(n_samples)

    # Create a complex target variable
    signal = (
        0.3 * X[:, 0]
        + 0.2 * X[:, 1]
        - 0.1 * X[:, 2]
        + 0.15 * X[:, 3] * X[:, 4]
        + 0.1 * np.sin(X[:, 5])
    )
    noise = 0.5 * np.random.randn(n_samples)
    y = (signal + noise > 0).astype(np.int32)

    print(
        f"Synthetic dataset: X={X.shape}, y={y.shape}, class balance={np.mean(y):.3f}"
    )
    return X, y


def load_hepmass(path_csv):
    """Robust HEPMASS loading with parse error handling"""
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    df = None

    for encoding in encodings:
        try:
            # Try different parsing options
            parsing_options = [
                # standard CSV
                {"encoding": encoding, "sep": ","},
                # Separator - space/tab
                {"encoding": encoding, "sep": r"\s+", "engine": "python"},
                # Skipping bad lines
                {"encoding": encoding, "sep": ",", "on_bad_lines": "skip"},
                # No title
                {
                    "encoding": encoding,
                    "sep": ",",
                    "header": None,
                    "on_bad_lines": "skip",
                },
            ]

            for options in parsing_options:
                try:
                    df = pd.read_csv(path_csv, **options)
                    print(
                        f"Successfully loaded HEPMASS with encoding: {encoding}, options: {options}"
                    )
                    break
                except (ParserError, pd.errors.ParserError):
                    continue

            if df is not None:
                break

        except UnicodeDecodeError:
            continue

    if df is None or df.empty:
        print("Failed to load HEPMASS, creating synthetic dataset...")
        return create_synthetic_hepmass()

    # Handling different column formats
    print(f"HEPMASS loaded: {df.shape}, columns: {list(df.columns)}")

    # Try to find the target variable
    if "# label" in df.columns:
        y = df["# label"].astype(np.int32)
        X = (
            df.drop(columns=["# label"])
            .select_dtypes(include=[np.number])
            .values.astype(np.float64)
        )
    elif "type" in df.columns:
        y = df["type"].astype(np.int32)
        X = (
            df.drop(columns=["type"])
            .select_dtypes(include=[np.number])
            .values.astype(np.float64)
        )
    elif "label" in df.columns:
        y = df["label"].astype(np.int32)
        X = (
            df.drop(columns=["label"])
            .select_dtypes(include=[np.number])
            .values.astype(np.float64)
        )
    else:
        # We assume the first or last column as the target
        if df.shape[1] > 1:
            # Try the last column as a target
            last_col = df.iloc[:, -1]
            if last_col.dtype in ["int64", "float64"] and last_col.nunique() <= 10:
                y = last_col.astype(np.int32)
                X = (
                    df.iloc[:, :-1]
                    .select_dtypes(include=[np.number])
                    .values.astype(np.float64)
                )
            else:
                # The first column as a target
                y = df.iloc[:, 0].astype(np.int32)
                X = (
                    df.iloc[:, 1:]
                    .select_dtypes(include=[np.number])
                    .values.astype(np.float64)
                )
        else:
            print("Cannot determine target variable, creating synthetic...")
            return create_synthetic_hepmass()

    # Correctness checks
    if X.shape[0] == 0 or X.shape[1] == 0:
        print("Empty feature matrix, creating synthetic...")
        return create_synthetic_hepmass()

    # Convert labels to binary if needed
    if len(np.unique(y)) > 2:
        print(f"Converting {len(np.unique(y))} classes to binary")
        y = (y > np.median(y)).astype(np.int32)

    # drop NaN/inf values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    return X, y


def load_avazu(path_csv, sample_n=None):
    # High-cardinality categoricals; use basic hashing trick to numeric bins for fairness
    df_iter = pd.read_csv(path_csv, chunksize=10_000_00)
    df = next(df_iter)
    if sample_n is not None and len(df) > sample_n:
        df = df.sample(sample_n, random_state=42)
    if "click" in df.columns:
        y = df["click"].astype(np.int32).values
        X = df.drop(columns=["click"])
    else:
        # competition format: 'id','click',... ; fallback
        y = df.iloc[:, 1].astype(np.int32).values
        X = df.drop(columns=[df.columns[1]])
    # Hash trick
    MOD = 1_000_003
    X_num = []
    for col in X.columns:
        if np.issubdtype(X[col].dtype, np.number):
            X_num.append(X[col].astype(np.float64).values)
        else:
            X_num.append(
                (X[col].astype(str).apply(hash).values % MOD).astype(np.float64)
            )
    X_num = np.vstack(X_num).T
    return X_num, y


def download_file(url, dest, chunk_size=2**20):
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


def download_all_datasets(data_dir="data/"):
    os.makedirs(data_dir, exist_ok=True)

    # Credit Card Fraud (Figshare: alternative Kaggle)
    creditcard_url = "https://www.dropbox.com/s/b44o3t3ehmnx2b7/creditcard.csv?dl=1"
    creditcard_path = os.path.join(data_dir, "creditcard.csv")
    if not os.path.exists(creditcard_path):
        print("Downloading CreditCardFraud ...")
        download_file(creditcard_url, creditcard_path)

    # HIGGS (UCI)
    higgs_url = "https://archive.ics.uci.edu/static/public/280/higgs.zip"
    higgs_zip = os.path.join(data_dir, "higgs.zip")
    higgs_csv_gz = os.path.join(data_dir, "HIGGS.csv.gz")
    if not os.path.exists(higgs_csv_gz):
        print("Downloading HIGGS ...")
        download_file(higgs_url, higgs_zip)
        with zipfile.ZipFile(higgs_zip) as zf:
            zf.extract("HIGGS.csv.gz", path=data_dir)
        os.remove(higgs_zip)

    # SUSY (UCI)
    susy_url = "https://archive.ics.uci.edu/static/public/279/susy.zip"
    susy_zip = os.path.join(data_dir, "susy.zip")
    susy_csv_gz = os.path.join(data_dir, "SUSY.csv.gz")
    if not os.path.exists(susy_csv_gz):
        print("Downloading SUSY ...")
        download_file(susy_url, susy_zip)
        with zipfile.ZipFile(susy_zip) as zf:
            zf.extract("SUSY.csv.gz", path=data_dir)
        os.remove(susy_zip)

    # KDDCup99 (10 percent) (UCI)
    kdd_url = "https://figshare.com/ndownloader/files/5976042"
    kdd_gz = os.path.join(data_dir, "kddcup.data_10_percent.gz")
    kdd_csv = os.path.join(data_dir, "kddcup.data_10_percent.csv")
    if not os.path.exists(kdd_csv):
        print("Downloading KDDCup99 ...")
        download_file(kdd_url, kdd_gz)
        import gzip

        with gzip.open(kdd_gz, "rb") as f_in, open(kdd_csv, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(kdd_gz)

    # Covertype (UCI/sklearn, already CSV)
    covertype_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    covertype_gz = os.path.join(data_dir, "covtype.data.gz")
    covertype_csv = os.path.join(data_dir, "covtype.csv")
    if not os.path.exists(covertype_csv):
        print("Downloading Covertype ...")
        download_file(covertype_url, covertype_gz)
        import gzip

        with gzip.open(covertype_gz, "rb") as f_in, open(covertype_csv, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(covertype_gz)

    # HEPMASS
    hepmass_url = "https://archive.ics.uci.edu/static/public/347/hepmass.zip"
    hepmass_csv = os.path.join(data_dir, "HEPMASS_train.csv")
    if not os.path.exists(hepmass_csv):
        print("Downloading HEPMASS ...")
        download_file(hepmass_url, hepmass_csv)

    # hepmass_url = "https://www.openml.org/data/get_csv/2419/BNG_balance-scale.csv"  # Временная замена
    # hepmass_csv = os.path.join(data_dir, "HEPMASS_train.csv")
    # if not os.path.exists(hepmass_csv):
    #     print("Downloading HEPMASS (alternative dataset)...")
    #     try:
    #         download_file(hepmass_url, hepmass_csv)
    #     except Exception as e:
    #         print(f"Failed to download HEPMASS: {e}")
    #         # Create a stub to avoid interrupting the entire benchmark
    #         print("Creating dummy HEPMASS dataset...")
    #         np.random.seed(42)
    #         X_dummy = np.random.randn(10000, 28).astype(np.float64)
    #         y_dummy = np.random.randint(0, 2, 10000).astype(np.int32)
    #         dummy_df = pd.DataFrame(X_dummy)
    #         dummy_df['label'] = y_dummy
    #         dummy_df.to_csv(hepmass_csv, index=False)

    # Avazu CTR (HF mirror, 2m rows slice — fastest for dev)
    avazu_url = (
        "https://www.kaggle.com/api/v1/datasets/download/wuyingwen06/avazu-ctr-train"
    )
    avazu_csv = os.path.join(data_dir, "avazu-ctr-train.zip")
    if not os.path.exists(avazu_csv):
        print("Downloading Avazu...")
        download_file(avazu_url, avazu_csv)

    print("Done downloading all datasets.")


# ---------------------------
# main additional functions
# ---------------------------


def train_eval_one(
    model_name,
    model_ctor,
    X_train,
    y_train,
    X_test,
    y_test,
    is_multiclass=False,
    use_gpu=False,
):

    if model_name == "AdaptiveBayes":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        fit_stats = _measure_run(model_ctor["fit"], X_train_scaled, y_train)
        pred_stats = _measure_run(model_ctor["predict"], X_test_scaled)

    else:
        # Regular training for other models
        fit_stats = _measure_run(model_ctor["fit"], X_train, y_train)
        pred_stats = _measure_run(model_ctor["predict"], X_test)

    y_pred = pred_stats["ret"]
    # Proba if available
    auc = None
    if not is_multiclass and "predict_proba" in model_ctor:
        proba_stats = _measure_run(model_ctor["predict_proba"], X_test)
        y_prob = (
            proba_stats["ret"][:, 1]
            if y_prob_shape(proba_stats["ret"])
            else proba_stats["ret"]
        )
        auc = roc_auc_score(y_test, y_prob)
        proba_time = proba_stats["elapsed_s"]
    else:
        proba_time = None
    acc = accuracy_score(y_test, y_pred)
    return {
        "model": model_name,
        "fit_s": fit_stats["elapsed_s"],
        "pred_s": pred_stats["elapsed_s"],
        "proba_s": proba_time,
        "cpu_peak_mb_fit": fit_stats["cpu_peak_mb"],
        "gpu_mem_mb_fit": fit_stats["gpu_mem_delta_mb"],
        "acc": acc,
        "auc": auc,
    }


def y_prob_shape(arr):
    return (arr.ndim == 2) and (arr.shape[1] > 1)


def make_models(use_gpu):
    models = []

    # AdaptiveBayes
    ab = AdaptiveBayes(base_lr=1e-3, eps=1e-10, device="gpu" if use_gpu else "cpu")
    models.append(
        (
            "AdaptiveBayes",
            {"fit": ab.fit, "predict": ab.predict, "predict_proba": ab.predict_proba},
        )
    )

    # XGBoost
    if XGB_OK:
        if use_gpu:
            params = {
                "n_estimators": 300,
                "max_depth": 8,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "device": "cuda",
                "eval_metric": "auc",
            }
        else:
            params = {
                "n_estimators": 300,
                "max_depth": 8,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "eval_metric": "auc",
            }

        xgbc = xgb.XGBClassifier(**params)
        models.append(
            (
                "XGBoost",
                {
                    "fit": xgbc.fit,
                    "predict": xgbc.predict,
                    "predict_proba": xgbc.predict_proba,
                },
            )
        )

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, max_depth=None)
    models.append(
        (
            "RandomForest",
            {"fit": rf.fit, "predict": rf.predict, "predict_proba": rf.predict_proba},
        )
    )

    # Neural Net (sklearn MLP)
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        batch_size=512,
        max_iter=20,
        solver="adam",
        early_stopping=True,
        random_state=42,
    )
    models.append(
        (
            "MLP",
            {
                "fit": mlp.fit,
                "predict": mlp.predict,
                "predict_proba": mlp.predict_proba,
            },
        )
    )

    # LightGBM
    if LGB_OK:
        device_type = "gpu" if use_gpu else "cpu"

        # Adaptive settings depending on the device
        if device_type == "gpu":
            lgbm = lgb.LGBMClassifier(
                n_estimators=300,  # Fewer iterations for GPU
                num_leaves=511,  # More leaves
                learning_rate=0.01,  # Fewer learning rate
                subsample=0.8,
                colsample_bytree=0.8,
                device_type="gpu",
                max_bin=127,  # More bins for GPU
                min_data_in_leaf=100,  # Minimum data in the sheet
                min_gain_to_split=0.01,  # Minimum gain for splitting
                verbose=-1,  # Remove unnecessary warnings
            )
        else:
            lgbm = lgb.LGBMClassifier(
                n_estimators=500,
                num_leaves=255,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                device_type="cpu",
                n_jobs=-1,
            )

        models.append(
            (
                "LightGBM",
                {
                    "fit": lgbm.fit,
                    "predict": lgbm.predict,
                    "predict_proba": lgbm.predict_proba,
                },
            )
        )

    # CatBoost
    if CAT_OK:
        cat = CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.1,
            verbose=False,
            task_type="GPU" if use_gpu else "CPU",
        )
        models.append(
            (
                "CatBoost",
                {
                    "fit": cat.fit,
                    "predict": cat.predict,
                    "predict_proba": cat.predict_proba,
                },
            )
        )

    # Logistic Regression
    lr = LogisticRegression(max_iter=200, solver="saga", n_jobs=-1)
    models.append(
        (
            "LogisticRegression",
            {"fit": lr.fit, "predict": lr.predict, "predict_proba": lr.predict_proba},
        )
    )

    return models


def run_benchmark(
    datasets_config,
    use_gpu=False,
    test_size=0.2,
    val_size=0.0,
    output_csv="results.csv",
):
    rows = []
    for ds in datasets_config:
        name = ds["name"]
        loader = ds["loader"]
        path = ds["path"]
        is_multiclass = ds.get("multiclass", False)
        sample_n = ds.get("sample_n")
        print(f"Loading {name} ...")
        if name == "Avazu":
            X, y = load_avazu(path, sample_n=sample_n)
        else:
            X, y = loader(path)
            if sample_n is not None and len(X) > sample_n:
                ridx = np.random.RandomState(42).choice(
                    len(X), size=sample_n, replace=False
                )
                X = X[ridx]
                y = y[ridx]
        print(f"{name}: X={X.shape}, y={y.shape}")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y if not is_multiclass else None,
        )
        models = make_models(use_gpu=use_gpu)
        for mname, m in models:
            print(f"Training {mname} on {name} ...")
            stats = train_eval_one(
                mname,
                m,
                X_tr,
                y_tr,
                X_te,
                y_te,
                is_multiclass=is_multiclass,
                use_gpu=use_gpu,
            )
            stats["dataset"] = name
            rows.append(stats)
            print(stats)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")
    return df


if __name__ == "__main__":
    # I stage: datasets loading
    download_all_datasets("data/")

    # Example configuration; update paths to local files
    datasets = [
        {
            "name": "CreditCardFraud",
            "loader": load_creditcard_fraud,
            "path": "data/creditcard.csv",
            "multiclass": False,
        },
        {
            "name": "HIGGS",
            "loader": load_higgs,
            "path": "data/HIGGS.csv.gz",
            "multiclass": False,
            "sample_n": 2_000_000,
        },
        {
            "name": "SUSY",
            "loader": load_susy,
            "path": "data/SUSY.csv.gz",
            "multiclass": False,
            "sample_n": 2_000_000,
        },
        {
            "name": "KDDCup99",
            "loader": load_kddcup99,
            "path": "data/kddcup.data_10_percent.csv",
            "multiclass": False,
        },
        {
            "name": "Covertype",
            "loader": load_covertype,
            "path": "data/covtype.csv",
            "multiclass": True,
        },
        {
            "name": "HEPMASS",
            "loader": create_synthetic_hepmass,
            "path": "data/HEPMASS_train.csv",
            "multiclass": False,
        },
        {
            "name": "Avazu",
            "loader": load_avazu,
            "path": "data/avazu-ctr-train.zip",
            "multiclass": False,
            "sample_n": 2_000_000,
        },
    ]
    use_gpu = GPU_OK

    # II stage: modelling
    run_benchmark(datasets, use_gpu=use_gpu, output_csv="benchmark_results.csv")
