import numpy as np
import pandas as pd
import pytest

from src.ds_tool import DSTools

tools = DSTools()

@pytest.mark.parametrize("input_data, expected_msg", [
    (pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}), "no NaN or infinite"),
    (pd.DataFrame({'col1': [1, np.nan, 3], 'col2': [4, 5, 6]}), "NaN values"),
    (pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, np.inf, 6]}), "infinite values"),
    (pd.DataFrame({'col1': [1, np.nan, 3], 'col2': [-np.inf, 5, 6]}), "both NaN and infinite"),
])
def test_check_NINF_pandas(input_data, expected_msg, capsys):
    tools.check_NINF(input_data)
    captured = capsys.readouterr()
    assert expected_msg in captured.out

@pytest.mark.parametrize("input_array, expected_msg", [
    (np.array([[1, 2, 3], [4, 5, 6]]), "no NaN or infinite"),
    (np.array([[1, np.nan, 3], [4, 5, 6]]), "NaN values"),
    (np.array([[1, 2, 3], [4, np.inf, 6]]), "infinite values"),
    (np.array([[1, np.nan, 3], [-np.inf, 5, 6]]), "both NaN and infinite"),
])
def test_check_NINF_numpy(input_array, expected_msg, capsys):
    tools.check_NINF(input_array)
    captured = capsys.readouterr()
    assert expected_msg in captured.out
