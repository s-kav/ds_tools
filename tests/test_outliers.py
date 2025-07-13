import pytest
import pandas as pd
import numpy as np
from src.ds_tool import DSTools, OutlierConfig

@pytest.fixture(scope="module")
def df_with_outliers():
    np.random.seed(42)
    normal_data = np.random.normal(loc=100, scale=20, size=90)
    outliers = np.array([-50, -40, 250, 300, 310])
    full_data = np.concatenate([normal_data, outliers])
    np.random.shuffle(full_data)
    df = pd.DataFrame({'value': full_data})
    df['category'] = np.random.choice(['A', 'B'], size=len(df))
    return df

@pytest.fixture(scope="module")
def tools():
    return DSTools()

def test_outlier_bounds_iqr_manual(df_with_outliers):
    # Calculate bounds manually for sigma=1.5
    q1 = df_with_outliers['value'].quantile(0.25)
    q3 = df_with_outliers['value'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Check expected approximate bounds (sanity check)
    assert lower_bound < q1 < q3 < upper_bound

def test_replace_outliers_default_mode(df_with_outliers, tools):
    # Default: replace outliers with boundary values, sigma=1.5
    df_copy = df_with_outliers.copy()
    df_replaced, p_upper, p_lower = tools.remove_outliers_iqr(df_copy, 'value')
    
    # Check that percentages are floats between 0 and 100
    assert isinstance(p_upper, (int, float))
    assert 0 <= p_upper <= 100
    assert isinstance(p_lower, (int, float))
    assert 0 <= p_lower <= 100
    
    # Check that replaced values are within bounds
    q1 = df_with_outliers['value'].quantile(0.25)
    q3 = df_with_outliers['value'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    assert df_replaced['value'].min() >= lower_bound
    assert df_replaced['value'].max() <= upper_bound

def test_remove_rows_outliers_mode(df_with_outliers, tools):
    # Remove rows containing outliers
    config_remove = OutlierConfig(change_remove=False)
    df_copy = df_with_outliers.copy()
    df_removed, _, _ = tools.remove_outliers_iqr(df_copy, 'value', config=config_remove)
    
    # Check DataFrame size reduced
    assert df_removed.shape[0] < df_with_outliers.shape[0]
    
    # Check all values within bounds after removal
    q1 = df_with_outliers['value'].quantile(0.25)
    q3 = df_with_outliers['value'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    assert df_removed['value'].min() >= lower_bound
    assert df_removed['value'].max() <= upper_bound

def test_replace_outliers_stricter_sigma_no_percentage(df_with_outliers, tools):
    # Use sigma=1.0 and percentage=False (return only DataFrame)
    config_custom = OutlierConfig(sigma=1.0, percentage=False)
    df_copy = df_with_outliers.copy()
    df_strict = tools.remove_outliers_iqr(df_copy, 'value', config=config_custom)
    
    # Return type should be DataFrame only
    assert isinstance(df_strict, pd.DataFrame)
    
    # Bounds stricter than sigma=1.5, so min/max should be within narrower range
    q1 = df_with_outliers['value'].quantile(0.25)
    q3 = df_with_outliers['value'].quantile(0.75)
    iqr = q3 - q1
    lower_bound_1_5 = q1 - 1.5 * iqr
    upper_bound_1_5 = q3 + 1.5 * iqr
    lower_bound_1_0 = q1 - 1.0 * iqr
    upper_bound_1_0 = q3 + 1.0 * iqr
    
    assert df_strict['value'].min() >= lower_bound_1_0
    assert df_strict['value'].max() <= upper_bound_1_0
    
    # Also stricter bounds are inside the looser bounds
    assert lower_bound_1_0 > lower_bound_1_5
    assert upper_bound_1_0 < upper_bound_1_5

def test_remove_outliers_iqr_raises_value_error_for_missing_column(df_with_outliers, tools):
    with pytest.raises(ValueError):
        tools.remove_outliers_iqr(df_with_outliers, 'non_existent_column')
