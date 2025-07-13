import os
import pytest
import pandas as pd
import polars as pl
from src.ds_tool import DSTools

@pytest.fixture(scope="module")
def dfs_to_save():
    pd_index = pd.Index(['id_1', 'id_2', 'id_3', 'id_4'], name='custom_index')
    pd_df = pd.DataFrame(
        {'A': [1, 2, 3, 4], 'B': ['x', 'y', 'z', 'w']},
        index=pd_index
    )
    pl_df = pl.DataFrame(
        {'C': [10.5, 20.5, 30.5], 'D': [True, False, True]}
    )
    return {'pandas_data': pd_df, 'polars_data': pl_df}

@pytest.fixture(scope="module")
def tools():
    return DSTools()

@pytest.mark.parametrize("format_,zip_filename", [
    ('parquet', 'test_archive.parquet.zip'),
    ('csv', 'test_archive.csv.zip'),
])
def test_save_and_load_dataframes(dfs_to_save, tools, format_, zip_filename):
    # Save dataframes to zip archive
    tools.save_dataframes_to_zip(
        dataframes=dfs_to_save,
        zip_filename=zip_filename,
        format=format_,
        save_index=True
    )
    assert os.path.exists(zip_filename), f"{zip_filename} should be created."

    # Read back with polars (only for parquet, csv with polars might not be supported or differ)
    if format_ == 'parquet':
        loaded_polars = tools.read_dataframes_from_zip(
            zip_filename=zip_filename,
            backend='polars'
        )
        # Check polars DataFrame equality
        assert loaded_polars['polars_data'].equals(dfs_to_save['polars_data'], 
                                                        null_equal=True), "Polars data must match after reload"

    # Read back with pandas
    loaded_pandas = tools.read_dataframes_from_zip(
        zip_filename=zip_filename,
        format=format_,
        backend='pandas'
    )
    # For pandas data, check equality carefully
    if format_ == 'parquet':
        pd.testing.assert_frame_equal(loaded_pandas['pandas_data'], dfs_to_save['pandas_data'])
    else:
        # CSV reading typically resets index, so compare after resetting index
        pd.testing.assert_frame_equal(
            loaded_pandas['pandas_data'].reset_index(drop=True),
            dfs_to_save['pandas_data'].reset_index(drop=True)
        )

    # Clean up
    os.remove(zip_filename)
    assert not os.path.exists(zip_filename), f"{zip_filename} should be deleted after test."
