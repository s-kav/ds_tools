# DSTools: Data Science Tools Library

[![PyPI version](https://badge.fury.io/py/ds-tools-your-username.svg)](https://badge.fury.io/py/ds-tools-Sergii-Kavun)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DSTools** is a Python library designed to assist data scientists and researchers by providing a collection of helpful functions for various stages of a data science project, from data exploration and preprocessing to model evaluation and synthetic data generation.

## Features

- **Data Exploration:** Quickly get statistics for numerical and categorical features (`describe_numeric`, `describe_categorical`), check for missing values (`check_NINF`), and visualize correlations (`corr_matrix`).
- **Model Evaluation:** Comprehensive classification model evaluation (`evaluate_classification`, `compute_metrics`) with clear visualizations (`plot_confusion_matrix`).
- **Data Preprocessing:** Encode categorical variables (`labeling`), handle outliers (`remove_outliers_iqr`), and scale features (`min_max_scale`).
- **Time Series Analysis:** Test for stationarity using the Dickey-Fuller test (`test_stationarity`).
- **Synthetic Data Generation:** Create complex numerical distributions matching specific statistical moments (`generate_distribution`, `generate_distribution_from_metrics`).
- **Advanced Statistics:** Calculate non-parametric correlation (`chatterjee_correlation`), entropy, and KL-divergence.
- **Utilities:** Save/load DataFrames to/from ZIP archives, generate random alphanumeric codes, and more.

## Installation

You can install DSTools directly from PyPI:

```bash
pip install dscience_tools
```

## Quick Start

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

## Function Overview

The library provides a wide range of functions. To see a full, formatted list of available tools, you can use the function_list method:

```python

from ds_tools import DSTools

tools = DSTools()
tools.function_list()

```

## Example: Generating a Synthetic Distribution

Need to create a dataset with specific statistical properties? generate_distribution_from_metrics can do that.

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


Full code base for other function testing you can find [here](https://github.com/s-kav/ds_tools/blob/main/tests/code_checking_dstool.py).


## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue on the GitHub repository.


## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/s-kav/ds_tools/blob/main/LICENSE) file for details.
