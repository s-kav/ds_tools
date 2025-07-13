import pytest
import pandas as pd
import numpy as np
from src.ds_tool import DSTools

@pytest.fixture
def sample_df():
    categories_list = ['C'] * 50 + ['B'] * 30 + ['A'] * 15 + ['D'] * 5
    np.random.seed(42)
    np.random.shuffle(categories_list)
    return pd.DataFrame({'product_category': categories_list})

@pytest.fixture
def tools():
    return DSTools()

def test_labeling_order_flag_true(sample_df, tools):
    df = sample_df
    target_column = 'product_category'
    df_ordered = tools.labeling(df, target_column, order_flag=True)
    
    mapping = df_ordered[[target_column]].drop_duplicates().sort_values(by=target_column)
    
    mapping = df_ordered[[target_column]].drop_duplicates()
    codes = df_ordered[[target_column]].drop_duplicates().copy()
    codes['code'] = df_ordered[target_column].drop_duplicates().values
    
    dict_cat_code = dict(zip(df_ordered['product_category'], df_ordered[target_column]))
   
    freq = df['product_category'].value_counts()
    unique_pairs = df_ordered[[target_column]].drop_duplicates()
    
    cat_code_pairs = pd.DataFrame({
        'category': df['product_category'],
        'code': df_ordered[target_column]
    }).drop_duplicates().sort_values('code')
    
    code_d = cat_code_pairs.loc[cat_code_pairs['category'] == 'D', 'code'].values[0]
    code_a = cat_code_pairs.loc[cat_code_pairs['category'] == 'A', 'code'].values[0]
    code_b = cat_code_pairs.loc[cat_code_pairs['category'] == 'B', 'code'].values[0]
    code_c = cat_code_pairs.loc[cat_code_pairs['category'] == 'C', 'code'].values[0]

    assert code_d < code_a < code_b < code_c, "Codes do not correspond to frequencies (rare ones should have a lower code)"

def test_labeling_order_flag_false(sample_df, tools):
    df = sample_df
    target_column = 'product_category'
    df_simple = tools.labeling(df, target_column, order_flag=False)

    unique_codes = df_simple[target_column].unique()
    unique_categories = df[target_column].unique()

    assert len(unique_codes) == len(unique_categories), "The number of unique codes does not match the number of categories"

    assert np.issubdtype(df_simple[target_column].dtype, np.integer), "The data type of codes must be integer."

def test_labeling_raises_error_for_nonexistent_column(sample_df, tools):
    with pytest.raises(ValueError):
        tools.labeling(sample_df, 'non_existent_column')
