# tests/test_plot_visualization.py
"""
/*
 * Copyright (c) [2026] [Sergii Kavun]
 *
 * This software is dual-licensed:
 * - PolyForm Noncommercial 1.0.0 (default)
 * - Commercial license available
 *
 * See LICENSE for details
 */

# Test Suite Specification

**Total tests: 50** — distributed by class, one per method

| Class | Tests | What is tested |
|-------|-------|----------------|
| `TestPlotKS` | 7 | return types, [0,1] range, high KS for good data, low KS for noise, Pandas Series, figsize |
| `TestPlotShap` | 6 | `ImportError` without shap (via patch), `ValueError` for unknown plot_type, correct shap function call for each plot_type, return value |
| `TestPlotQQ` | 7 | dict keys, r_squared ∈ [0,1], normal data → high R², non‑normal → lower, Pandas Series, different `dist`, figsize |
| `TestPlotCumulativeExplainedVariance` | 7 | return type, n_comp ≥ 1, monotonicity, sum ≤ 1, threshold respected, `max_components` limits, DataFrame |
| `TestPlotGiniEntropy` | 2 | runs without errors, figsize |
| `TestPlotBiasVariance` | 4 | return within `param_range`, matches argmin, lists, `param_name + figsize` |
| `TestPlotRocCurve` | 5 | single/multi‑model, optional names, figsize, single‑element list |
| `TestPlotPrecisionRecall` | 5 | single/multi‑model, optional names, figsize, class imbalance |
| `TestPlotElbowCurve` | 8 | return type/length, k in range, inertia decreasing, tuple format, reasonable k for 3‑cluster data, DataFrame, figsize |

## Key Techniques

- `matplotlib.use("Agg")` — headless, works in CI
- `unittest.mock.patch` — mocks presence/absence of `shap` for `plot_shap`
- `scope="module"` — for heavy fixtures (matrices, K‑Means)

## Test Data Design

| Fixture | Purpose |
|---------|---------|
| `binary_clf_data` | good signal for classification |
| `random_clf_data` | pure noise |
| `clustered_matrix` | 3 clear clusters |
"""

from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def binary_clf_data():
    """Well-separated binary classification data."""
    np.random.seed(42)
    n = 400
    y_true = np.random.randint(0, 2, size=n)
    base_probs = np.where(y_true == 1, 0.75, 0.25)
    noise = np.random.normal(0, 0.15, size=n)
    y_proba = np.clip(base_probs + noise, 0.01, 0.99)
    return y_true, y_proba


@pytest.fixture(scope="module")
def random_clf_data():
    """Random (no-signal) binary classification data."""
    np.random.seed(0)
    n = 300
    y_true = np.random.randint(0, 2, size=n)
    y_proba = np.random.uniform(0, 1, size=n)
    return y_true, y_proba


@pytest.fixture(scope="module")
def feature_matrix():
    """Small feature matrix for PCA / K-Means tests."""
    np.random.seed(7)
    return np.random.randn(200, 10)


@pytest.fixture(scope="module")
def clustered_matrix():
    """Feature matrix with 3 clear clusters for elbow test."""
    np.random.seed(99)
    c1 = np.random.randn(60, 2) + np.array([0, 0])
    c2 = np.random.randn(60, 2) + np.array([6, 6])
    c3 = np.random.randn(60, 2) + np.array([12, 0])
    return np.vstack([c1, c2, c3])


# ---------------------------------------------------------------------------
# plot_ks
# ---------------------------------------------------------------------------


class TestPlotKS:
    def test_returns_tuple_of_floats(self, tools, binary_clf_data):
        y_true, y_proba = binary_clf_data
        result = tools.plot_ks(y_true, y_proba)
        assert isinstance(result, tuple) and len(result) == 2
        ks_stat, ks_threshold = result
        assert isinstance(ks_stat, float)
        assert isinstance(ks_threshold, float)

    def test_ks_stat_in_unit_interval(self, tools, binary_clf_data):
        y_true, y_proba = binary_clf_data
        ks_stat, _ = tools.plot_ks(y_true, y_proba)
        assert 0.0 <= ks_stat <= 1.0

    def test_ks_threshold_in_unit_interval(self, tools, binary_clf_data):
        y_true, y_proba = binary_clf_data
        _, ks_threshold = tools.plot_ks(y_true, y_proba)
        assert 0.0 <= ks_threshold <= 1.0

    def test_well_separated_data_high_ks(self, tools, binary_clf_data):
        """A model with signal should yield KS > 0.3."""
        y_true, y_proba = binary_clf_data
        ks_stat, _ = tools.plot_ks(y_true, y_proba)
        assert ks_stat > 0.3

    def test_random_data_low_ks(self, tools, random_clf_data):
        """A random model should yield KS close to 0."""
        y_true, y_proba = random_clf_data
        ks_stat, _ = tools.plot_ks(y_true, y_proba)
        assert ks_stat < 0.3

    def test_accepts_pandas_series(self, tools, binary_clf_data):
        y_true, y_proba = binary_clf_data
        ks_stat, _ = tools.plot_ks(pd.Series(y_true), pd.Series(y_proba))
        assert 0.0 <= ks_stat <= 1.0

    def test_custom_figsize(self, tools, binary_clf_data):
        y_true, y_proba = binary_clf_data
        tools.plot_ks(y_true, y_proba, figsize=(8, 4))


# ---------------------------------------------------------------------------
# plot_shap
# ---------------------------------------------------------------------------


class TestPlotShap:
    def test_raises_import_error_when_shap_missing(self, tools):
        """Should raise ImportError with a helpful message if shap is absent."""
        with patch.dict("sys.modules", {"shap": None}):
            with pytest.raises(ImportError, match="shap"):
                tools.plot_shap(MagicMock(), np.random.randn(10, 3))

    def test_raises_value_error_for_unknown_plot_type(self, tools):
        """Invalid plot_type should raise ValueError."""
        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_shap.Explainer.return_value = mock_explainer
        mock_explainer.return_value = MagicMock()

        with patch.dict("sys.modules", {"shap": mock_shap}):
            with pytest.raises(ValueError, match="Unknown plot_type"):
                tools.plot_shap(
                    MagicMock(), np.random.randn(10, 3), plot_type="invalid"
                )

    def test_summary_plot_calls_shap_summary_plot(self, tools):
        """plot_type='summary' should delegate to shap.summary_plot."""
        mock_shap = MagicMock()
        X = np.random.randn(20, 4)
        model = MagicMock()

        with patch.dict("sys.modules", {"shap": mock_shap}):
            tools.plot_shap(model, X, plot_type="summary")

        mock_shap.summary_plot.assert_called_once()

    def test_bar_plot_calls_shap_bar(self, tools):
        mock_shap = MagicMock()
        X = np.random.randn(20, 4)

        with patch.dict("sys.modules", {"shap": mock_shap}):
            tools.plot_shap(MagicMock(), X, plot_type="bar")

        mock_shap.plots.bar.assert_called_once()

    def test_beeswarm_plot_calls_shap_beeswarm(self, tools):
        mock_shap = MagicMock()
        X = np.random.randn(20, 4)

        with patch.dict("sys.modules", {"shap": mock_shap}):
            tools.plot_shap(MagicMock(), X, plot_type="beeswarm")

        mock_shap.plots.beeswarm.assert_called_once()

    def test_returns_shap_values(self, tools):
        mock_shap = MagicMock()
        fake_shap_values = MagicMock()
        mock_shap.Explainer.return_value.return_value = fake_shap_values
        X = np.random.randn(10, 3)

        with patch.dict("sys.modules", {"shap": mock_shap}):
            result = tools.plot_shap(MagicMock(), X)

        assert result is fake_shap_values


# ---------------------------------------------------------------------------
# plot_qq
# ---------------------------------------------------------------------------


class TestPlotQQ:
    def test_returns_dict_with_correct_keys(self, tools):
        data = np.random.normal(0, 1, 200)
        result = tools.plot_qq(data)
        assert isinstance(result, dict)
        assert {"slope", "intercept", "r_squared"} == set(result.keys())

    def test_r_squared_in_unit_interval(self, tools):
        data = np.random.normal(0, 1, 300)
        result = tools.plot_qq(data)
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_normal_data_high_r_squared(self, tools):
        """Normally distributed data vs norm distribution → high R²."""
        np.random.seed(1)
        data = np.random.normal(5, 2, 500)
        result = tools.plot_qq(data, dist="norm")
        assert result["r_squared"] > 0.98

    def test_non_normal_data_lower_r_squared(self, tools):
        """Exponential data vs norm distribution → lower R² than normal data."""
        np.random.seed(2)
        normal_data = np.random.normal(0, 1, 500)
        exp_data = np.random.exponential(scale=1, size=500)
        r2_normal = tools.plot_qq(normal_data, dist="norm")["r_squared"]
        r2_exp = tools.plot_qq(exp_data, dist="norm")["r_squared"]
        assert r2_normal > r2_exp

    def test_accepts_pandas_series(self, tools):
        data = pd.Series(np.random.normal(0, 1, 100))
        result = tools.plot_qq(data)
        assert "r_squared" in result

    def test_custom_distribution(self, tools):
        """Should accept any scipy.stats distribution name."""
        data = np.random.exponential(scale=1, size=200)
        result = tools.plot_qq(data, dist="expon")
        assert "r_squared" in result

    def test_custom_figsize(self, tools):
        data = np.random.normal(0, 1, 100)
        tools.plot_qq(data, figsize=(6, 6))


# ---------------------------------------------------------------------------
# plot_cumulative_explained_variance
# ---------------------------------------------------------------------------


class TestPlotCumulativeExplainedVariance:
    def test_returns_tuple(self, tools, feature_matrix):
        result = tools.plot_cumulative_explained_variance(feature_matrix)
        assert isinstance(result, tuple) and len(result) == 2

    def test_n_components_is_positive_integer(self, tools, feature_matrix):
        n_comp, _ = tools.plot_cumulative_explained_variance(feature_matrix)
        assert isinstance(n_comp, int) and n_comp >= 1

    def test_cumulative_variance_is_monotone(self, tools, feature_matrix):
        _, cum_var = tools.plot_cumulative_explained_variance(feature_matrix)
        assert np.all(
            np.diff(cum_var) >= -1e-10
        ), "Cumulative variance must be non-decreasing"

    def test_cumulative_variance_bounded(self, tools, feature_matrix):
        _, cum_var = tools.plot_cumulative_explained_variance(feature_matrix)
        assert cum_var[-1] <= 1.0 + 1e-9

    def test_threshold_respected(self, tools, feature_matrix):
        """n_components should be the minimum to reach the threshold."""
        threshold = 0.80
        n_comp, cum_var = tools.plot_cumulative_explained_variance(
            feature_matrix, threshold=threshold
        )
        assert cum_var[n_comp - 1] >= threshold

    def test_max_components_limits_output(self, tools, feature_matrix):
        max_comp = 5
        _, cum_var = tools.plot_cumulative_explained_variance(
            feature_matrix, max_components=max_comp
        )
        assert len(cum_var) == max_comp

    def test_accepts_dataframe(self, tools, feature_matrix):
        df = pd.DataFrame(feature_matrix)
        n_comp, _ = tools.plot_cumulative_explained_variance(df)
        assert n_comp >= 1


# ---------------------------------------------------------------------------
# plot_gini_entropy
# ---------------------------------------------------------------------------


class TestPlotGiniEntropy:
    def test_runs_without_error(self, tools):
        tools.plot_gini_entropy()

    def test_custom_figsize(self, tools):
        tools.plot_gini_entropy(figsize=(8, 5))


# ---------------------------------------------------------------------------
# plot_bias_variance
# ---------------------------------------------------------------------------


class TestPlotBiasVariance:
    @pytest.fixture(scope="class")
    def bv_data(self):
        """Synthetic bias-variance data with clear minimum at depth=5."""
        depths = np.arange(1, 11)
        train_errors = 1.0 / depths
        val_errors = 1.0 / depths + 0.05 * (depths - 5) ** 2
        return train_errors, val_errors, depths

    def test_returns_optimal_param(self, tools, bv_data):
        train_errs, val_errs, depths = bv_data
        result = tools.plot_bias_variance(train_errs, val_errs, depths)
        assert result in depths

    def test_optimal_is_argmin_val_error(self, tools, bv_data):
        train_errs, val_errs, depths = bv_data
        result = tools.plot_bias_variance(train_errs, val_errs, depths)
        expected = depths[np.argmin(val_errs)]
        assert result == expected

    def test_accepts_lists(self, tools):
        train = [0.5, 0.4, 0.35, 0.3]
        val = [0.6, 0.5, 0.55, 0.7]
        params = [1, 2, 3, 4]
        result = tools.plot_bias_variance(train, val, params)
        assert result in params

    def test_custom_param_name_and_figsize(self, tools, bv_data):
        train_errs, val_errs, depths = bv_data
        tools.plot_bias_variance(
            train_errs, val_errs, depths, param_name="Tree Depth", figsize=(12, 5)
        )


# ---------------------------------------------------------------------------
# plot_roc_curve
# ---------------------------------------------------------------------------


class TestPlotRocCurve:
    def test_single_model_runs(self, tools, binary_clf_data):
        y_true, y_proba = binary_clf_data
        tools.plot_roc_curve(y_true, y_proba)

    def test_multi_model_comparison(self, tools, binary_clf_data):
        y_true, y_proba = binary_clf_data
        tools.plot_roc_curve(
            [y_true, y_true],
            [y_proba, y_proba],
            model_names=["Model A", "Model B"],
        )

    def test_model_names_optional(self, tools, binary_clf_data):
        y_true, y_proba = binary_clf_data
        tools.plot_roc_curve([y_true, y_true], [y_proba, y_proba])

    def test_custom_figsize(self, tools, binary_clf_data):
        y_true, y_proba = binary_clf_data
        tools.plot_roc_curve(y_true, y_proba, figsize=(6, 6))

    def test_single_list_input(self, tools, binary_clf_data):
        """List with one element should work like a single-model call."""
        y_true, y_proba = binary_clf_data
        tools.plot_roc_curve([y_true], [y_proba], model_names=["Solo"])


# ---------------------------------------------------------------------------
# plot_precision_recall
# ---------------------------------------------------------------------------


class TestPlotPrecisionRecall:
    def test_single_model_runs(self, tools, binary_clf_data):
        y_true, y_proba = binary_clf_data
        tools.plot_precision_recall(y_true, y_proba)

    def test_multi_model_comparison(self, tools, binary_clf_data):
        y_true, y_proba = binary_clf_data
        tools.plot_precision_recall(
            [y_true, y_true],
            [y_proba, y_proba],
            model_names=["LR", "XGB"],
        )

    def test_model_names_optional(self, tools, binary_clf_data):
        y_true, y_proba = binary_clf_data
        tools.plot_precision_recall([y_true, y_true], [y_proba, y_proba])

    def test_custom_figsize(self, tools, binary_clf_data):
        y_true, y_proba = binary_clf_data
        tools.plot_precision_recall(y_true, y_proba, figsize=(7, 7))

    def test_imbalanced_data(self, tools):
        """Should handle imbalanced datasets (very few positives)."""
        np.random.seed(5)
        n = 300
        y_true = np.zeros(n, dtype=int)
        y_true[:15] = 1  # 5% positive
        y_proba = np.random.uniform(0, 1, n)
        y_proba[:15] += 0.4
        y_proba = np.clip(y_proba, 0, 1)
        tools.plot_precision_recall(y_true, y_proba)


# ---------------------------------------------------------------------------
# plot_elbow_curve
# ---------------------------------------------------------------------------


class TestPlotElbowCurve:
    def test_returns_tuple(self, tools, clustered_matrix):
        result = tools.plot_elbow_curve(clustered_matrix, max_clusters=6)
        assert isinstance(result, tuple) and len(result) == 2

    def test_suggested_k_in_valid_range(self, tools, clustered_matrix):
        max_k = 8
        suggested_k, _ = tools.plot_elbow_curve(clustered_matrix, max_clusters=max_k)
        assert 1 <= suggested_k <= max_k

    def test_inertia_list_length(self, tools, clustered_matrix):
        max_k = 6
        _, inertias = tools.plot_elbow_curve(clustered_matrix, max_clusters=max_k)
        assert len(inertias) == max_k

    def test_inertia_is_decreasing(self, tools, clustered_matrix):
        _, inertias = tools.plot_elbow_curve(clustered_matrix, max_clusters=8)
        values = [v for _, v in inertias]
        assert all(
            values[i] >= values[i + 1] for i in range(len(values) - 1)
        ), "Inertia should be non-increasing as k grows"

    def test_inertia_tuple_format(self, tools, clustered_matrix):
        _, inertias = tools.plot_elbow_curve(clustered_matrix, max_clusters=5)
        for k, inertia in inertias:
            assert isinstance(k, int)
            assert isinstance(inertia, float)

    def test_well_separated_clusters_suggest_correct_k(self, tools, clustered_matrix):
        """For 3-cluster data the elbow should suggest k near 3."""
        suggested_k, _ = tools.plot_elbow_curve(clustered_matrix, max_clusters=8)
        assert 2 <= suggested_k <= 5, f"Expected elbow near k=3, got {suggested_k}"

    def test_accepts_dataframe(self, tools, clustered_matrix):
        df = pd.DataFrame(clustered_matrix)
        suggested_k, _ = tools.plot_elbow_curve(df, max_clusters=5)
        assert suggested_k >= 1

    def test_custom_figsize(self, tools, clustered_matrix):
        tools.plot_elbow_curve(clustered_matrix, max_clusters=4, figsize=(8, 4))
