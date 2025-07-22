import os
import tempfile
import zipfile

import pandas as pd
import polars as pl
import pytest


@pytest.fixture(scope="module")
def dfs_to_save():
    pd_index = pd.Index(["id_1", "id_2", "id_3", "id_4"], name="custom_index")
    pd_df = pd.DataFrame({"A": [1, 2, 3, 4], "B": ["x", "y", "z", "w"]}, index=pd_index)
    pl_df = pl.DataFrame({"C": [10.5, 20.5, 30.5], "D": [True, False, True]})
    return {"pandas_data": pd_df, "polars_data": pl_df}


@pytest.mark.parametrize(
    "format_, zip_filename",
    [
        ("parquet", "test_archive.parquet.zip"),
        ("csv", "test_archive.csv.zip"),
    ],
)
def test_save_and_load_dataframes(dfs_to_save, tools, format_, zip_filename):
    # Save dataframes to zip archive
    tools.save_dataframes_to_zip(
        dataframes=dfs_to_save,
        zip_filename=zip_filename,
        format=format_,
        save_index=True,
    )
    assert os.path.exists(zip_filename), f"{zip_filename} should be created."

    # Read back with polars (only for parquet, csv with polars might not be supported or differ)
    if format_ == "parquet":
        loaded_polars = tools.read_dataframes_from_zip(
            zip_filename=zip_filename, backend="polars"
        )
        # Check polars DataFrame equality
        assert loaded_polars["polars_data"].equals(
            dfs_to_save["polars_data"], null_equal=True
        ), "Polars data must match after reload"

    # Read back with pandas
    loaded_pandas = tools.read_dataframes_from_zip(
        zip_filename=zip_filename, format=format_, backend="pandas"
    )
    # For pandas data, check equality carefully
    if format_ == "parquet":
        pd.testing.assert_frame_equal(
            loaded_pandas["pandas_data"], dfs_to_save["pandas_data"]
        )
    else:
        # CSV reading typically resets index, so compare after resetting index
        pd.testing.assert_frame_equal(
            loaded_pandas["pandas_data"].reset_index(drop=True),
            dfs_to_save["pandas_data"].reset_index(drop=True),
        )

    # Clean up
    os.remove(zip_filename)
    assert not os.path.exists(
        zip_filename
    ), f"{zip_filename} should be deleted after test."


# --- tests for save_dataframes_to_zip ---
def test_save_dataframes_to_zip_invalid_input_type_raises_error(tools):
    """
    Tests that save_dataframes_to_zip raises a TypeError if input is not a dict.
    """
    with pytest.raises(TypeError, match="`dataframes` must be a dictionary"):
        # get a list instead dictionary
        tools.save_dataframes_to_zip([pd.DataFrame()], "test.zip")


def test_save_dataframes_to_zip_unsupported_format_raises_error(tools):
    """
    Tests that save_dataframes_to_zip raises a ValueError for an unsupported format.
    """
    df = pd.DataFrame({"a": [1]})
    dataframes = {"my_df": df}

    with pytest.raises(ValueError, match="Unsupported format: 'txt'"):
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "test.zip")
            tools.save_dataframes_to_zip(dataframes, zip_path, format="txt")


def test_save_dataframes_to_zip_unsupported_dataframe_type_raises_error(tools):
    """

    Tests that save_dataframes_to_zip raises a TypeError for an unsupported object type.
    """
    # Create a dummy object that is neither a Pandas nor a Polars DataFrame
    unsupported_df = {"not": "a dataframe"}
    dataframes = {"my_unsupported_df": unsupported_df}

    with pytest.raises(TypeError, match="Unsupported DataFrame type"):
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "test.zip")
            tools.save_dataframes_to_zip(dataframes, zip_path)


# --- tests for read_dataframes_from_zip ---
def test_read_dataframes_from_zip_invalid_backend_raises_error(tools):
    """
    Tests that read_dataframes_from_zip raises ValueError for an invalid backend.
    """
    # It doesn't matter if the file exists, because the error should occur first.
    with pytest.raises(ValueError, match="`backend` must be 'polars' or 'pandas'"):
        tools.read_dataframes_from_zip("any.zip", backend="invalid_backend")


def test_read_dataframes_from_zip_pandas_fallback_to_csv(
    tools, mocker, sample_pandas_df
):
    """
    Tests the fallback logic in read_dataframes_from_zip for pandas.
    This simulates a case where read_parquet fails and read_csv succeeds.
    """
    # mock pd.read_parquet so it throws an error,
    # and check that after this pd.read_csv will be called.
    mocker.patch(
        "ds_tool.pd.read_parquet", side_effect=ValueError("Simulated read error")
    )
    mock_read_csv = mocker.patch("ds_tool.pd.read_csv", return_value=sample_pandas_df)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a dummy zip archive with a "non-parquet" file, but with the .parquet extension
        zip_path = os.path.join(temp_dir, "test.zip")
        csv_path = os.path.join(temp_dir, "file1.csv")
        sample_pandas_df.to_csv(csv_path, index=False)

        with zipfile.ZipFile(zip_path, "w") as zf:
            # Important: save as file1.parquet so that read_parquet is called first
            zf.write(csv_path, "file1.parquet")

        # function call with backend='pandas'
        loaded_dfs = tools.read_dataframes_from_zip(
            zip_path, format="parquet", backend="pandas"
        )

        # check that fallback is done
        mock_read_csv.assert_called_once()
        assert "file1" in loaded_dfs
