import numpy as np
import pandas as pd
import pytest

from src.ds_tool import DSTools

tools = DSTools()

@pytest.fixture
def sample_df():
    np.random.seed(42)
    cities = ['Moscow', 'Saint Petersburg', 'Novosibirsk', 'Ekaterinburg', 'Kazan']
    status = ['Active', 'Inactive', 'Pending', 'Archive']
    product_type = ['Electronics', 'Clothing', 'Books']

    data = {
        'City': np.random.choice(cities, size=100, p=[0.4, 0.3, 0.1, 0.1, 0.1]),
        'Customer_status': np.random.choice(status + [np.nan], size=100, p=[0.5, 0.2, 0.1, 0.1, 0.1]),
        'Product_type': np.random.choice(product_type, size=100)
    }

    return pd.DataFrame(data)

def test_category_stats_city_column(capsys, sample_df):
    tools.category_stats(sample_df, 'City')
    captured = capsys.readouterr()
    assert "City" in captured.out
    assert "percentage" in captured.out

def test_category_stats_customer_status_column(capsys, sample_df):
    tools.category_stats(sample_df, 'Customer_status')
    captured = capsys.readouterr()
    assert "Customer_status" in captured.out
    assert "Active" in captured.out or "Inactive" in captured.out

def test_category_stats_invalid_column(sample_df):
    with pytest.raises(ValueError):
        tools.category_stats(sample_df, 'Non-existent_column')
