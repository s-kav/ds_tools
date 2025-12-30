"""
/*
 * Copyright (c) [2025] [Sergii Kavun]
 *
 * This software is dual-licensed:
 * - PolyForm Noncommercial 1.0.0 (default)
 * - Commercial license available
 *
 * See LICENSE for details
 */
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from ds_tools import DSTools


@pytest.fixture(autouse=True)
def close_plots_after_test():
    """A fixture to automatically close all plots after each test runs."""
    yield
    plt.close("all")


# Add the project root directory to the Python search path
# This will allow imports like from src.module import ... to work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(autouse=True, scope="session")
def setup_sys_path():
    """Add the project root to the sys.path for imports."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


@pytest.fixture(scope="session")
def tools():
    """
    Provides a single, session-wide instance of the DSTools class.
    This is efficient as the class is initialized only once for all tests.
    """
    return DSTools()


@pytest.fixture(scope="session")
def sample_sparse_df():
    """
    Provides a sample DataFrame that is mostly sparse (filled with zeros/NaNs)
    but has a few non-zero values.
    """
    # create zero-matrix 10x10
    data = np.zeros((10, 10))

    # Inserting multiple non-zero values into random locations
    # density = 5 / (10*10) = 5%
    data[2, 3] = 5
    data[5, 8] = -2
    data[7, 1] = 10
    data[0, 9] = 1
    data[9, 0] = 3

    # can also add NaNs, as they are also considered "missing"
    # in the context of sparsity for some data types.
    data[1, 1] = np.nan

    return pd.DataFrame(data, columns=[f"col_{i}" for i in range(10)])


@pytest.fixture(scope="session")
def sample_pandas_df():
    """
    Provides a simple, sample pandas DataFrame for various I/O and manipulation tests.
    """
    data = {
        "id": [101, 102, 103, 104],
        "category": ["A", "B", "A", "C"],
        "value": [15.5, 20.0, 12.3, 8.0],
        "is_active": [True, False, True, True],
    }
    return pd.DataFrame(data)
