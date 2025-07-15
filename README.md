# DSTools: Data Science Research Toolkit

[![Tests](https://github.com/s-kav/ds_tools/actions/workflows/python-publish.yml/badge.svg)](https://github.com/s-kav/ds_tools/actions)
[![PyPI version](https://badge.fury.io/py/dscience-tools.svg)](https://badge.fury.io/py/dscience-tools)
[![codecov](https://codecov.io/gh/s-kav/ds_tools/branch/main/graph/badge.svg)](https://codecov.io/gh/s-kav/ds_tools)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Authors

- [@sergiikavun](https://www.linkedin.com/in/sergii-kavun/)

**DSTools** is a Python library designed to assist data scientists and researchers by providing a collection of helpful functions for various stages of a data science project, from data exploration and preprocessing to model evaluation and synthetic data generation.

This comprehensive library with helper functions to accelerate and simplify various stages of the data science research cycle.

This toolkit is built on top of popular libraries like Pandas, Polars, Scikit-learn, and Matplotlib, providing a higher-level API for common tasks in Exploratory Data Analysis (EDA), feature preprocessing, model evaluation, and synthetic data generation. It is designed for data scientists, analysts, and researchers who want to write cleaner, more efficient, and more reproducible code.

# Table of Contents

*   [Key Features](#key-features)
*   [Installation](#installation)
*   [Usage](#usage)
*   [Function Overview](#function-overview)
*   [Example](#example)
*   [Available Tools](#available-tools)
*   [Contributing](#contributing)
*   [References](#references)
*   [License](#license)


# Key Features


-   **Advanced Data Analysis:** Get quick and detailed statistics for numerical and categorical columns.
-   **Powerful Visualizations:** Generate insightful correlation matrices and confusion matrices with a single function call.
-   **Comprehensive Model Evaluation:** Calculate a wide range of classification metrics and visualize performance curves effortlessly.
-   **Synthetic Data Generation:** Create datasets with specific statistical properties (mean, median, std, skew, kurtosis) for robust testing and simulation. Create complex numerical distributions matching specific statistical moments (`generate_distribution`, `generate_distribution_from_metrics`).
-   **Efficient Preprocessing:** Encode categorical variables, handle outliers, and create features from missing values.
-   **Utility Functions:** A collection of helpers for stationarity testing, data validation, and file I/O operations.
-   **Data Exploration:** Quickly get statistics for numerical and categorical features (`describe_numeric`, `describe_categorical`), check for missing values (`check_NINF`), and visualize correlations (`corr_matrix`).
-   **Model Evaluation:** Comprehensive classification model evaluation (`evaluate_classification`, `compute_metrics`) with clear visualizations (`plot_confusion_matrix`).
-   **Data Preprocessing:** Encode categorical variables (`labeling`), handle outliers (`remove_outliers_iqr`), and scale features (`min_max_scale`).
-   **Time Series Analysis:** Test for stationarity using the Dickey-Fuller test (`test_stationarity
-   **Advanced Statistics:** Calculate non-parametric correlation (`chatterjee_correlation`), entropy, and KL-divergence.
-   **Utilities:** Save/load DataFrames to/from ZIP archives, generate random alphanumeric codes, and more.


# Installation

## Clone the Repository

```bash
git clone https://github.com/s-kav/ds_tools.git

```
or

Install `dscience-tools` directly from PyPI:

```bash
pip install dscience-tools
```

## Navigate to the Project Directory

```bash
cd ds_tools

```

## Install Dependencies

Ensure you have Python version 3.8 or higher and install the required packages:

```bash
pip install -r requirements.txt

```

# Usage

Here's a simple example of how to use the library to evaluate a classification model.


```python

import numpy as np
from ds_tools import DSTools

# 1. Initialize the toolkit
tools = DSTools()

# 2. Generate some dummy data
y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
y_probs = np.array([0.1, 0.8, 0.6, 0.3, 0.9, 0.2, 0.4, 0.7])

# 3. Get a comprehensive evaluation report
# This will print metrics and show plots for ROC and Precision-Recall curves.
results = tools.evaluate_classification(true_labels=y_true, pred_probs=y_probs)

# The results are also returned as a dictionary
print(f"\nROC AUC Score: {results['roc_auc']:.4f}")

```


Full code base for other function testing you can find [here](https://github.com/s-kav/ds_tools/blob/main/tests/).


# Function Overview

The library provides a wide range of functions. To see a full, formatted list of available tools, you can use the function_list method:

```python

from ds_tools import DSTools

tools = DSTools()
tools.function_list()

```

# Example

Generating a Synthetic Distribution: need to create a dataset with specific statistical properties?
generate_distribution_from_metrics can do that.


```python

from ds_tools import DSTools, DistributionConfig

tools = DSTools()

# Define the desired metrics
metrics_config = DistributionConfig(
    mean=1042,
    median=330,
    std=1500,
    min_val=1,
    max_val=120000,
    skewness=13.2,
    kurtosis=245, # Excess kurtosis
    n=10000
)

# Generate the data
generated_data = tools.generate_distribution_from_metrics(n=10000, metrics=metrics_config)

print(f"Generated Mean: {np.mean(generated_data):.2f}")
print(f"Generated Std: {np.std(generated_data):.2f}")

```


Full code base for other function testing you can find [here](https://github.com/s-kav/ds_tools/blob/main/tests/).


# Available Tools

The library includes a wide range of functions. Here is a complete list:

1. compute_metrics: Calculate main pre-selected classification metrics.
2. corr_matrix: Calculate and visualize a correlation matrix.
3. category_stats: Calculate and print categorical statistics.
4. sparse_calc: Calculate sparsity level as a coefficient.
5. trials_res_df: Aggregate Optuna optimization trials into a DataFrame.
6. labeling: Encode categorical variables with optional ordering.
7. remove_outliers_iqr: Remove or cap outliers using the IQR method.
8. stat_normal_testing: Perform D'Agostino's K² test for normality.
9. test_stationarity: Perform the Dickey-Fuller test for time-series stationarity.
10. check_NINF: Check for NaN and infinite values in a DataFrame.
11. df_stats: Get a quick, comprehensive overview of a DataFrame's structure.
12. describe_categorical: Generate a detailed description of categorical columns.
13. describe_numeric: Generate a detailed description of numerical columns.
14. generate_distribution: Generate a synthetic numerical distribution with specific statistical properties.
15. evaluate_classification: A master function to calculate, print, and visualize metrics for a binary classification model.
16. grubbs_test: Perform Grubbs' test to identify a single outlier.
17. plot_confusion_matrix: Plot a clear and readable confusion matrix.
18. add_missing_value_features: Add features based on the count of missing values per row.
19. chatterjee_correlation: Calculate Chatterjee's rank correlation coefficient (Xi).
20. calculate_entropy & calculate_kl_divergence: Compute information theory metrics.
21. min_max_scale: Scale DataFrame columns to the range [0, 1].
22. save_dataframes_to_zip & read_dataframes_from_zip: Save and load multiple dataframes from a single ZIP archive.
23. generate_alphanum_codes: Generate an array of random alphanumeric codes.
24. generate_distribution_from_metrics: A powerful function to generate a synthetic distribution matching a full suite of statistical metrics.


# Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue on the GitHub repository. If you have ideas for new features or improvements, please open an issue first to discuss what you would like to change.

To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix (git checkout -b feature/AmazingFeature).
3. Make your changes.
4. Add tests for your new feature. Ensure all tests pass (pytest).
5. Format your code (black . and ruff --fix .).
6. Commit your changes with clear messages (git commit -m 'Add some AmazingFeature').
7. Push to the branch/fork (git push origin feature/AmazingFeature).
8. Open a Pull Request.

Please ensure your code adheres to PEP8 standards and includes appropriate docstrings and comments.


# References

For citing you should use:

Sergii Kavun. (2025). s-kav/ds_tools: Version 0.9.1 (v0.9.1). Zenodo. https://doi.org/10.5281/zenodo.15864146

[![DOI](https://zenodo.org/badge/1001952407.svg)](https://doi.org/10.5281/zenodo.15864146)


# License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/s-kav/ds_tools/blob/main/LICENSE) file for details.
