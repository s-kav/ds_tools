import sys
import os
import pytest
import matplotlib.pyplot as plt

@pytest.fixture(autouse=True)
def close_plots_after_test():
    """A fixture to automatically close all plots after each test runs."""
    yield
    plt.close('all')
# Add the project root directory to the Python search path
# This will allow imports like from src.module import ... to work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
