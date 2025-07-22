import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    categories_list = ["C"] * 50 + ["B"] * 30 + ["A"] * 15 + ["D"] * 5
    np.random.seed(42)
    np.random.shuffle(categories_list)
    return pd.DataFrame({"product_category": categories_list})


def test_labeling_order_flag_true(sample_df, tools):
    df = sample_df
    target_column = "product_category"
    df_ordered = tools.labeling(df, target_column, order_flag=True)

    codes = df_ordered[[target_column]].drop_duplicates().copy()
    codes["code"] = df_ordered[target_column].drop_duplicates().values

    cat_code_pairs = (
        pd.DataFrame(
            {"category": df["product_category"], "code": df_ordered[target_column]}
        )
        .drop_duplicates()
        .sort_values("code")
    )

    code_d = cat_code_pairs.loc[cat_code_pairs["category"] == "D", "code"].values[0]
    code_a = cat_code_pairs.loc[cat_code_pairs["category"] == "A", "code"].values[0]
    code_b = cat_code_pairs.loc[cat_code_pairs["category"] == "B", "code"].values[0]
    code_c = cat_code_pairs.loc[cat_code_pairs["category"] == "C", "code"].values[0]

    assert (
        code_d < code_a < code_b < code_c
    ), "Codes do not correspond to frequencies (rare ones should have a lower code)"


def test_labeling_order_flag_false(sample_df, tools):
    df = sample_df
    target_column = "product_category"
    df_simple = tools.labeling(df, target_column, order_flag=False)

    unique_codes = df_simple[target_column].unique()
    unique_categories = df[target_column].unique()

    assert len(unique_codes) == len(
        unique_categories
    ), "The number of unique codes does not match the number of categories"

    assert np.issubdtype(
        df_simple[target_column].dtype, np.integer
    ), "The data type of codes must be integer."


def test_labeling_raises_error_for_nonexistent_column(sample_df, tools):
    with pytest.raises(ValueError):
        tools.labeling(sample_df, "non_existent_column")
