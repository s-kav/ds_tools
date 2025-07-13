# src/ds_tool/ds_tools.py
# version 0.9.2 - Refactored and optimized


# Library consisting of additional & helpful functions for data science research stages.


import random
import re
import os
import zipfile
import tempfile
import fastparquet

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import polars as pl

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core.core_schema import ValidationInfo

from scipy import stats
from scipy.stats import rankdata
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    class_likelihood_ratios,
    cohen_kappa_score,
    det_curve,
    hamming_loss,
    jaccard_score,
    log_loss,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)
from sklearn.preprocessing import OrdinalEncoder
from statsmodels.tsa.stattools import adfuller


EPSILON = 1e-8  # A small constant to avoid log(0) or division by zero


class MetricsConfig(BaseModel):
    """Configuration for metrics computation."""
    
    error_vis: bool = Field(True, description = 'Flag for error visualization')
    print_values: bool = Field(False, description = 'Flag for printing metric values')


class CorrelationConfig(BaseModel):
    """Configuration for correlation matrix visualization."""
    
    font_size: int = Field(14, ge = 8, le = 20, description = 'Font size for heatmap')
    build_method: str = Field('pearson', description = 'Correlation method')
    image_size: Tuple[int, int] = Field((16, 16), description = 'Image size as tuple')
    
    @field_validator('build_method')
    @classmethod
    def validate_method(cls, v) -> str:
        valid_methods = ['pearson', 'kendall', 'spearman']
        if v not in valid_methods:
            raise ValueError(f'Method must be one of {valid_methods}')
        return v


class OutlierConfig(BaseModel):
    """Configuration for outlier removal."""
    
    sigma: float = Field(1.5, gt = 0, description = 'IQR multiplier for outlier detection')
    change_remove: bool = Field(True, description = 'True to change outliers, False to remove')
    percentage: bool = Field(True, description = 'Calculate outlier percentages')


class DistributionConfig(BaseModel):
    """Configuration for distribution generation."""
    mean: float = Field(description = 'Target mean')
    median: float = Field(description = 'Target median')
    std: float = Field(gt=0, description = 'Target standard deviation')
    min_val: float = Field(description = 'Minimum value')
    max_val: float = Field(description = 'Maximum value')
    skewness: float = Field(description = 'Target skewness')
    kurtosis: float = Field(description = 'Target kurtosis')
    n: int = Field(gt = 0, description = 'Number of data points')
    accuracy_threshold: float = Field(0.01, gt = 0, le = 0.1, description = 'Accuracy threshold')
    outlier_ratio: float = Field(0.025, ge = 0, le = 0.1, description = 'Proportion of outliers')
    
    @model_validator(mode='after')
    def validate_max_greater_than_min(cls, v, values) -> 'DistributionConfig':
        if 'min_val' in values and v <= values['min_val']:
            raise ValueError('max_val must be greater than min_val')
        return v
    
    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
        validate_assignment = True


class GrubbsTestResult(BaseModel):
    """
    Pydantic model for storing the results of Grubbs' test for outliers.
    """
    is_outlier: bool = Field(..., description = 'True if an outlier is detected, False otherwise.')
    g_calculated: float = Field(..., description = 'The calculated G-statistic for the test.')
    g_critical: float = Field(..., description = 'The critical G-value for the given alpha.')
    outlier_value: Optional[Union[int, float]] = Field(None, description = 'The value of the detected outlier.')
    outlier_index: Optional[int] = Field(None, description = 'The index of the detected outlier.')


class DSTools:
    """Data Science Tools for research and analysis.
    
    Agenda:
    -------
    function_list:
        Prints the list of available tools
    compute_metrics: 
        Calculate main pre-selected classification metrics
    corr_matrix: 
        Calculate and visualize correlation matrix
    category_stats: 
        Calculate and print categorical statistics (unique values analysis)
    sparse_calc: 
        Calculate sparsity level as coefficient
    trials_res_df: 
        Aggregate Optuna optimization trials as DataFrame
    labeling:
        Encode categorical variables with optional ordering
    remove_outliers_iqr:
        Remove outliers using IQR method
    stat_normal_testing:
        Perform D'Agostino's K² test for normality
    test_stationarity:
        Perform Dickey-Fuller test for stationarity
    check_NINF:
        Check for NaN and infinite values in DataFrame
    df_stats:
        Quick overview of DataFrame structure
    describe_categorical:
        Detailed description of categorical columns
    describe_numeric:
        Detailed description of numerical columns
    generate_distribution:
        Generate synthetic numerical distribution with specific statistical properties
    validate_moments:
        Helper method to check if the requested statistical moments are physically possible
    evaluate_classification:
        Calculates, prints, and visualizes metrics for a binary classification model
    grubbs_test:
        Performs Grubbs' test to identify a single outlier in a dataset
    plot_confusion_matrix:
        Plots a clear and readable confusion matrix using seaborn
    add_missing_value_features:
        Adds features based on the count of missing values per row
    chatterjee_correlation:
        Calculates Chatterjee's rank correlation coefficient (Xi) between two variables.
    calculate_entropy:
        Calculates the Shannon entropy of a probability distribution.
    calculate_kl_divergence:
        Calculates the Kullback-Leibler (KL) divergence between two
        probability distributions.
    min_max_scale:
        Scales specified columns of a DataFrame to the range [0, 1].
    save_dataframes_to_zip:
        Saves one or more Pandas/Polars DataFrames into a single ZIP archive.
    read_dataframes_from_zip:
        Reads one or more Pandas/Polars DataFrames from a ZIP archive.
    generate_alphanum_codes:
        Generates an array of random alphanumeric codes of a specified length.
    generate_distribution_from_metrics:
        Generates a synthetic distribution of numbers matching given statistical metrics.
    """
    
    def __init__(self):
        """Initialize the DSTools class with default configurations."""
        
        plt.rcParams['figure.figsize'] = (15, 9)
        pd.options.display.float_format = '{:.2f}'.format
        np.set_printoptions(suppress = True, precision = 4)
        
        # Set random seeds for reproducibility
        random_seed = 42
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Theme configuration
        self.plotly_theme = 'plotly_dark'
    
    
    def function_list(self) -> pd.DataFrame:
        """
        Parses the list of available tools (the 'Agenda') from the class
        docstring as a formatted table (Pandas DataFrame).

        Returns:
        pd.DataFrame: A DataFrame with 'Function Name' and 'Description'
                      columns. Returns an empty DataFrame if the 'Agenda'
                      section is not found.
        
        Usage:
            pd.set_option('display.max_colwidth', 200)
            tools = DSTools()
            tools.function_list()
        """
        # 1. Get the main docstring of the class
        doc = self.__class__.__doc__
    
        if not doc:
            print("Warning: No documentation found for this class.")
            return pd.DataFrame(columns=['Function Name', 'Description'])
    
        # 2. Find the 'Agenda' section
        match = re.search(r'Agenda:\s*---+\s*(.*)', doc, re.S)
    
        if not match:
            print("Warning: No 'Agenda' section found in the class documentation.")
            return pd.DataFrame(columns=['Function Name', 'Description'])
    
        # 3. Parse the content with the robust regex method
        agenda_content = match.group(1).strip()
        lines = agenda_content.split('\n')
    
        tools_data = []
        current_entry = None
        entry_pattern = re.compile(r'^\s*([a-zA-Z0-9_]+):\s*(.*)')
    
        for line in lines:
            m = entry_pattern.match(line)
            if m:
                if current_entry:
                    current_entry['Description'] = ' '.join(current_entry['Description']).strip()
                    tools_data.append(current_entry)
                
                func_name = m.group(1)
                desc_part = m.group(2).strip()
                current_entry = {
                    'Function Name': func_name,
                    'Description': [desc_part] if desc_part else []
                }
            elif current_entry and line.strip():
                current_entry['Description'].append(line.strip())
    
        if current_entry:
            current_entry['Description'] = ' '.join(current_entry['Description']).strip()
            tools_data.append(current_entry)
    
        # 4. Create and return the DataFrame
        if not tools_data:
            print("Warning: No tools found in the Agenda.")
            return pd.DataFrame(columns=['Function Name', 'Description'])
        out_df = pd.DataFrame(tools_data).iloc[1:]
        
        with pd.option_context(
            'display.max_colwidth', 200,      # Макс. ширина колонки (в символах)
            'display.width', 350,            # Общая ширина вывода
            'display.colheader_justify', 'center' # Выравнивание заголовков по левому краю
        ):
            return out_df
    
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_predict: np.ndarray,
        y_predict_proba: np.ndarray,
        config: Optional[MetricsConfig] = None
    ) -> pd.DataFrame:
        """
        Calculate main pre-selected classification metrics.
        
        Args:
            y_true: True labels
            y_predict: Predicted labels
            y_predict_proba: Predicted probabilities
            config: Configuration for metrics computation
            
        Returns:
            DataFrame with calculated metrics
            
        Usage:
            from ds_tool import DSTools, MetricsConfig
            tools = DSTools()
            metrics = tools.compute_metrics(y_test, y_pred, y_pred_proba)
        """
        if config is None:
            config = MetricsConfig()
        
        metrics_dict = {}
        
        # Average Precision Score
        aps = average_precision_score(y_true, y_predict_proba) * 100
        metrics_dict['Average_precision, %'] = round(aps, 2)
        
        if config.print_values:
            print(f'Average_precision = {aps:.3f} %')
        
        # Balanced Accuracy Score
        bas = balanced_accuracy_score(y_true, y_predict) * 100
        metrics_dict['Balanced_accuracy, %'] = round(bas, 2)
        
        if config.print_values:
            print(f'Balanced_accuracy = {bas:.3f} %')
        
        # Likelihood Ratios
        clr = class_likelihood_ratios(y_true, y_predict)
        metrics_dict['Likelihood_ratios+'] = clr[0]
        metrics_dict['Likelihood_ratios-'] = clr[1]
        
        if config.print_values:
            print(f'Likelihood_ratios+ = {clr[0]:.3f}\nLikelihood_ratios- = {clr[1]:.3f}')
        
        # Cohen's Kappa Score
        cks = cohen_kappa_score(y_true, y_predict) * 100
        metrics_dict['Kappa_score, %'] = round(cks, 2)
        
        if config.print_values:
            print(f'Kappa_score = {cks:.3f} %')
        
        # Hamming Loss
        hl = hamming_loss(y_true, y_predict) * 100
        metrics_dict['Incor_pred_labels (hamming_loss), %'] = round(hl, 2)
        
        if config.print_values:
            print(f'Incorrect_predicted_labels (hamming_loss) = {hl:.3f} %')
        
        # Jaccard Score
        hs = jaccard_score(y_true, y_predict) * 100
        metrics_dict['Jaccard_similarity, %'] = round(hs, 2)
        
        if config.print_values:
            print(f'Jaccard_similarity = {hs:.3f} %')
        
        # Log Loss
        ls = log_loss(y_true, y_predict_proba)
        metrics_dict['Cross_entropy_loss'] = ls
        
        if config.print_values:
            print(f'Cross_entropy_loss = {ls:.3f}')
        
        # Correlation Coefficient
        cc = np.corrcoef(y_true, y_predict)[0][1] * 100
        metrics_dict['Coef_correlation, %'] = round(cc, 2)
        
        if config.print_values:
            print(f'Coef_correlation = {cc:.3f} %')
        
        # Error visualization
        if config.error_vis:
            fpr, fnr, thresholds = det_curve(y_true, y_predict_proba)
            plt.plot(thresholds, fpr, label = 'False Positive Rate (FPR)')
            plt.plot(thresholds, fnr, label = 'False Negative Rate (FNR)')
            plt.title('Error Rates vs Threshold Levels')
            plt.xlabel('Threshold Level')
            plt.ylabel('Error Rate')
            plt.legend()
            plt.grid(True)
            plt.show()
        
        return pd.DataFrame([metrics_dict])
    
    
    def corr_matrix(
        self,
        df: pd.DataFrame,
        config: Optional[CorrelationConfig] = None
    ) -> None:
        """
        Calculate and visualize correlation matrix.
        
        Args:
            df: Input DataFrame with numerical columns
            config: Configuration for correlation matrix visualization
            
        Usage:
             from ds_tool import DSTools, CorrelationConfig
             tools = DSTools()
             tools.corr_matrix(df, CorrelationConfig(font_size=12))
        """
        if config is None:
            config = CorrelationConfig()
        
        # Calculate correlation matrix
        corr = df.corr(method = config.build_method)
        mask = np.triu(np.ones_like(corr, dtype = bool))
        
        # Determine figure size based on number of columns
        n_cols = len(df.columns)
        
        if n_cols < 5:
            fig_size = (8, 8)
        elif n_cols < 9:
            fig_size = (10, 10)
        elif n_cols < 15:
            fig_size = (22, 22)
        else:
            fig_size = config.image_size
        
        fig, ax = plt.subplots(figsize = fig_size)
        
        # Create heatmap
        ax = sns.heatmap(
            corr,
            annot = True,
            annot_kws = {'size': config.font_size},
            fmt = '.3f',
            center = 0,
            linewidths = 1.0,
            linecolor = 'black',
            square = True,
            cmap = sns.diverging_palette(20, 220, n = 100),
            mask = mask
        )
        
        # Customize x-axis
        ax.tick_params(
            axis = 'x',
            which = 'major',
            direction = 'inout',
            length = 20,
            width = 4,
            color = 'm',
            pad = 10,
            labelsize = 16,
            labelcolor = 'b',
            bottom = True,
            top = True,
            labelbottom = True,
            labeltop = True,
            labelrotation = 85
        )
        
        # Customize y-axis
        ax.tick_params(
            axis = 'y',
            which = 'major',
            direction = 'inout',
            length = 20,
            width = 4,
            color = 'm',
            pad = 10,
            labelsize = 16,
            labelcolor = 'r',
            left = True,
            right = False,
            labelleft = True,
            labelright = False,
            labelrotation = 0
        )
        
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation = 0,
            fontsize = 16,
            verticalalignment = 'center'
        )
        
        plt.title(f'Correlation ({config.build_method}) matrix for selected features', fontsize = 20)
        plt.tight_layout()
        plt.show()
    
    
    def category_stats(self, df: pd.DataFrame, col_name: str) -> None:
        """
        Calculate and print categorical statistics for unique values analysis.
        
        Args:
            df: Input DataFrame
            col_name: Column name for statistics calculation
            
        Usage:
             tools = DSTools()
             tools.category_stats(df, 'category_column')
        """
        if col_name not in df.columns:
            raise ValueError(f'Column {col_name} not found in DataFrame')
        
        value_counts = df[col_name].value_counts()
        percentage = df[col_name].value_counts(normalize=True) * 100
        
        aggr_stats = pd.DataFrame({
            'uniq_names': value_counts.index.tolist(),
            'amount_values': value_counts.values.tolist(),
            'percentage': percentage.values.tolist()
        })
        
        aggr_stats.columns = pd.MultiIndex.from_product([[col_name], aggr_stats.columns])
        print(aggr_stats)
    
    
    def sparse_calc(self, df: pd.DataFrame) -> float:
        """
        Calculate sparsity level as coefficient.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Sparsity coefficient as percentage
            
        Usage:
             tools = DSTools()
             sparsity = tools.sparse_calc(df)
        """
        sparse_coef = round(df.apply(pd.arrays.SparseArray).sparse.density * 100, 2)
        print(f'Level of sparsity = {sparse_coef} %')
        
        return sparse_coef
    
    
    def trials_res_df(self, study_trials: List[Any], metric: str) -> pd.DataFrame:
        """
        Aggregate Optuna optimization trials as DataFrame.
        
        Args:
            study_trials: List of Optuna trials (study.trials)
            metric: Metric name for sorting (e.g., 'MCC', 'F1')
            
        Returns:
            DataFrame with aggregated trial results
            
        Usage:
             tools = DSTools()
             results = tools.trials_res_df(study.trials, 'MCC')
        """
        df_results = pd.DataFrame()
        
        for trial in study_trials:
            if trial.value is None:
                continue
                
            trial_data = pd.DataFrame.from_dict(trial.params, orient = 'index').T
            trial_data.insert(0, metric, trial.value)
            
            if trial.datetime_complete and trial.datetime_start:
                duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
                trial_data['Duration'] = duration
            
            df_results = pd.concat([df_results, trial_data], ignore_index = True)
        
        return df_results.sort_values(metric, ascending = False)
    
    
    def labeling(
        self,
        df: pd.DataFrame,
        col_name: str,
        order_flag: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical variables with optional ordering.
        
        Args:
            df: Input DataFrame
            col_name: Column name for transformation
            order_flag: Whether to apply ordering based on frequency
            
        Returns:
            DataFrame with encoded column
            
        Usage:
             tools = DSTools()
             df = tools.labeling(df, 'category_column', True)
        """
        if col_name not in df.columns:
            raise ValueError(f'Column {col_name} not found in DataFrame')
        
        df_copy = df.copy()
        unique_values = df_copy[col_name].unique()
        value_index = dict(zip(unique_values, range(len(unique_values))))
        print(f'Set of unique indexes for <{col_name}>:\n{value_index}')
        
        if order_flag:
            counts = df_copy[col_name].value_counts(normalize = True).sort_values().index.tolist()
            counts_dict = {val: i for i, val in enumerate(counts)}
            encoder = OrdinalEncoder(categories = [list(counts_dict.keys())], dtype = int)
        else:
            encoder = OrdinalEncoder(dtype = int)
        
        df_copy[col_name] = encoder.fit_transform(df_copy[[col_name]])
        return df_copy
    
    
    def remove_outliers_iqr(
        self,
        df: pd.DataFrame,
        column_name: str,
        config: Optional[OutlierConfig] = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, float, float]]:
        """
        Remove outliers using IQR (Inter Quartile Range) method.
        
        Args:
            df: Input DataFrame
            column_name: Target column name
            config: Configuration for outlier removal
            
        Returns:
            Modified DataFrame, optionally with outlier percentages
            
        Usage:
             from ds_tool import DSTools, OutlierConfig
             tools = DSTools()
             config_custom = OutlierConfig(sigma=1.0, percentage=False) 
             df_clean = tools.remove_outliers_iqr(df, 'target_column', config=config_custom)
             df_replaced, p_upper, p_lower = tools.remove_outliers_iqr(df, 'target_column')
        """
        if config is None:
            config = OutlierConfig()
        
        if column_name not in df.columns:
            raise ValueError(f'Column {column_name} not found in DataFrame')
        
        df_copy = df.copy()
        target = df_copy[column_name]
        
        q1 = target.quantile(0.25)
        q3 = target.quantile(0.75)
        iqr = q3 - q1
        iqr_lower = q1 - config.sigma * iqr
        iqr_upper = q3 + config.sigma * iqr
        
        outliers_upper = target > iqr_upper
        outliers_lower = target < iqr_lower
        
        if config.change_remove:
            df_copy.loc[outliers_upper, column_name] = iqr_upper
            df_copy.loc[outliers_lower, column_name] = iqr_lower
        else:
            df_copy = df_copy[~(outliers_upper | outliers_lower)]
        
        if config.percentage:
            percent_upper = round(outliers_upper.sum() / len(df) * 100, 2)
            percent_lower = round(outliers_lower.sum() / len(df) * 100, 2)
            return df_copy, percent_upper, percent_lower
        
        return df_copy
    
    
    def stat_normal_testing(
        self,
        check_object: Union[pd.DataFrame, pd.Series],
        describe_flag: bool = False
    ) -> None:
        """
        Perform D'Agostino's K² test for normality testing.
        
        Args:
            check_object: Input data (DataFrame or Series)
            describe_flag: Whether to show descriptive statistics
            
        Usage:
             tools = DSTools()
             
             tools.stat_normal_testing(data, describe_flag=True)
        """
        if isinstance(check_object, pd.DataFrame) and len(check_object.columns) == 1:
            check_object = check_object.iloc[:, 0]
        
        # Perform normality test
        stat, p_value = stats.normaltest(check_object)
        print(f'Statistics = {stat:.3f}, p = {p_value:.3f}')
        
        alpha = 0.05
        if p_value > alpha:
            print('Data looks Gaussian (fail to reject H0). Data is normal')
        else:
            print('Data does not look Gaussian (reject H0). Data is not normal')
        
        # Calculate kurtosis and skewness
        kurtosis_val = stats.kurtosis(check_object)
        skewness_val = stats.skew(check_object)
        
        print(f'\nKurtosis: {kurtosis_val:.3f}')
        if kurtosis_val > 0:
            print('Distribution has heavier tails than normal')
        elif kurtosis_val < 0:
            print('Distribution has lighter tails than normal')
        else:
            print('Distribution has normal tail weight')
        
        print(f'\nSkewness: {skewness_val:.3f}')
        if -0.5 <= skewness_val <= 0.5:
            print('Data are fairly symmetrical')
        elif skewness_val < -1 or skewness_val > 1:
            print('Data are highly skewed')
        else:
            print('Data are moderately skewed')
        
        # Visualization
        sns.displot(check_object, bins = 30)
        plt.title('Distribution of the data')
        plt.show()
        
        if describe_flag:
            print('\nDescriptive Statistics:')
            print(check_object.describe())
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 6))
            
            ax1.hist(check_object, bins = 50, edgecolor = 'black')
            ax1.set_title('Histogram')
            ax1.set_xlabel('Values')
            ax1.set_ylabel('Frequency')
            
            stats.probplot(check_object, dist = 'norm', plot = ax2)
            ax2.set_title('Q-Q Plot')
            
            plt.tight_layout()
            plt.show()
    
    
    def test_stationarity(
        self,
        check_object: pd.Series,
        print_results_flag: bool = True,
        len_window: int = 30
    ) -> None:
        """
        Perform Dickey-Fuller test for stationarity testing.
        
        Args:
            check_object: Input time series data
            print_results_flag: Whether to print detailed results
            len_window: length of a window, default is 30
            
        Usage:
             tools = DSTools()
             tools.test_stationarity(time_series, print_results_flag=True)
        """
        # Calculate rolling statistics
        rolling_mean = check_object.rolling(window = len_window).mean()
        rolling_std = check_object.rolling(window = len_window).std()
        
        # Plot rolling statistics
        fig, ax = plt.subplots(figsize = (12, 6))
        ax.plot(check_object, color = 'blue', label = 'Original', linewidth = 2)
        ax.plot(rolling_mean, color = 'red', label = 'Rolling Mean', linewidth = 2)
        ax.plot(rolling_std, color = 'black', label = 'Rolling Std', linewidth = 2)
        
        ax.legend(loc = 'upper left')
        ax.set_title('Rolling Mean & Standard Deviation')
        ax.grid(True, alpha = 0.3)
        plt.tight_layout()
        plt.show()
        
        # Perform Dickey-Fuller test
        adf_result = adfuller(check_object, autolag = 'AIC')
        adf_output = pd.Series(
            adf_result[0:4],
            index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used']
        )
        
        for key, value in adf_result[4].items():
            adf_output[f'Critical Value ({key})'] = value
        
        if print_results_flag:
            print('Results of Dickey-Fuller Test:')
            print(adf_output)
        
        # Interpret results
        if adf_output['p-value'] <= 0.05:
            print('\nData does not have a unit root. Data is STATIONARY!')
        else:
            print('\nData has a unit root. Data is NON-STATIONARY!')
    
    
    def check_NINF(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Check DataFrame or array for NaN and infinite values.
        
        Args:
            data: Input data to check
            
        Usage:
             tools = DSTools()
             tools.check_NINF(data)
        """
        if isinstance(data, pd.DataFrame):
            has_nan = data.isnull().any().any()
            has_inf = np.isinf(data.select_dtypes(include = [np.number])).any().any()
        else:
            has_nan = np.isnan(data).any()
            has_inf = np.isinf(data).any()
        
        if not has_nan and not has_inf:
            print('Dataset has no NaN or infinite values')
        elif has_nan and not has_inf:
            print('Dataset has NaN values but no infinite values')
        elif not has_nan and has_inf:
            print('Dataset has infinite values but no NaN values')
        else:
            print('Dataset has both NaN and infinite values')
    
    
    def df_stats(self, df: pd.DataFrame) -> None:
        """
        Provide quick overview of DataFrame structure.
        
        Args:
            df: Input DataFrame
            
        Usage:
             tools = DSTools()
             tools.df_stats(df)
        """
        print(f'Columns:       \t{df.shape[1]}')
        print(f'Rows:          \t{df.shape[0]}')
        print(f'Missing (%):   \t{np.round(df.isnull().sum().sum() / df.size * 100, 1)}%')
        print(f'Memory:        \t{np.round(df.memory_usage(deep = True).sum() / 10**6, 1)} MB')
    
    
    def describe_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detailed description of categorical columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with categorical statistics
            
        Usage:
             tools = DSTools()
             cat_stats = tools.describe_categorical(df)
        """
        cols_to_process = []
        for col_name in df.columns:
            # We consider a column categorical if its type is 'object' or 'category',
            # OR if it consists entirely of blanks (to catch the edge case).
            # We ignore everything else (numbers, dates).
            is_object_or_cat = pd.api.types.is_object_dtype(df[col_name]) or \
                               pd.api.types.is_categorical_dtype(df[col_name])
            
            is_all_nan = df[col_name].isnull().all()
    
            if is_object_or_cat or is_all_nan:
                cols_to_process.append(col_name)
    
        if not cols_to_process:
            return pd.DataFrame()
    
        stats_list = []
        for col in cols_to_process:
            description = df[col].describe()
            stats = {
                'column': col,
                'missing (%)': np.round(df[col].isnull().sum() / len(df) * 100, 1),
                'unique': description.get('unique', 0),
                'top': description.get('top'), # .get() will return None if the key does not exist
                'freq': description.get('freq')
            }
            stats_list.append(stats)
        
        result_df = pd.DataFrame(stats_list)
        
        return result_df.set_index('column')
    
    
    def describe_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detailed description of numerical columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with numerical statistics
            
        Usage:
             tools = DSTools()
             num_stats = tools.describe_numeric(df)
        """
        numeric_cols = df.select_dtypes(include = [np.number]).columns
        if len(numeric_cols) == 0:
            return pd.DataFrame()
        
        numeric_df = df[numeric_cols]
        description = numeric_df.describe()
        
        result_data = {
            'sum': numeric_df.sum(),
            'missing (%)': np.round(numeric_df.isnull().sum() / len(df) * 100, 1),
            'median': numeric_df.median(),
            'skew': numeric_df.skew(),
            'kurtosis': numeric_df.kurtosis(),
        }
        
        # Add description statistics
        for stat in description.index:
            if stat != 'count':
                result_data[stat] = description.loc[stat]
        
        return pd.DataFrame(result_data, index = numeric_cols)
    
    
    @staticmethod
    def validate_moments(std: float, skewness: float, kurtosis: float) -> bool:
        """
        Validate that statistical moments are physically possible.
        A key property is that kurtosis must be greater than or equal to
        the square of skewness minus 2.
        
        Args:
            std: Standard deviation
            skewness: Skewness value
            kurtosis: Kurtosis value
            
        Returns:
            True if moments are valid, False otherwise
            
        Usage:
             tools = DSTools()
             is_valid = tools.validate_moments(1.0, 0.5, 3.0)
        """
        return std > 0 and kurtosis >= (skewness**2 - 2)
    
    
    def generate_distribution(self, config: DistributionConfig) -> np.ndarray:
        """
        Generates a distribution matching the provided statistical metrics.

        This function creates a distribution by generating a base dataset with a
        shape defined by kurtosis, adds outliers, and then iteratively scales
        and shifts the data to match the target mean and standard deviation
        within a specified accuracy threshold.

        Args:
            config: A Pydantic model instance containing all configuration parameters.

        Returns:
            A NumPy array of numerical values with the specified properties.
            
        Usage:
             tools = DSTools()
             config = DistributionConfig(
            ...     mean=100, median=95, std=15, min_val=50, max_val=200,
            ...     skewness=0.5, kurtosis=3.5, n=1000
            ... )
             data = tools.generate_distribution(config)
             print(f'Generated Mean: {np.mean(data):.2f}, Std: {np.std(data):.2f}')
        """
        if not self.validate_moments(config.std, config.skewness, config.kurtosis):
            raise ValueError(
                'Invalid statistical moments. Check that std > 0 and '
                'kurtosis >= (skewness² - 2).'
            )

        num_outliers = int(config.n * config.outlier_ratio)
        num_base = config.n - num_outliers

        # --- 1. Generate Base Distribution ---
        # Generate a base distribution with a shape influenced by kurtosis.
        # Student's t-distribution is used for heavy tails (kurtosis > 3).
        if config.kurtosis > 3.5:
            # Lower degrees of freedom lead to heavier tails
            df = max(1, int(10 / (config.kurtosis - 2.5)))
            base_data = stats.t.rvs(df = df, size = num_base)
        else:
            base_data = np.random.standard_normal(size = num_base)

        # --- 2. Add Outliers ---
        # Generate outliers to further influence the tails.
        if num_outliers > 0:
            # Outliers are generated with a larger variance to be distinct.
            outlier_scale = config.std * (1 + config.kurtosis / 3)
            outliers = np.random.normal(loc=0, scale=outlier_scale, size=num_outliers)
            data = np.concatenate([base_data, outliers])
        else:
            data = base_data
            
        np.random.shuffle(data)

        # --- 3. Iterative Scaling and Shifting ---
        # Iteratively adjust the data to match the target mean and std.
        # This is more stable than trying to adjust all moments at once.
        max_iterations = 50
        
        for _ in range(max_iterations):
            current_mean = np.mean(data)
            current_std = np.std(data, ddof=1)

            # Check for convergence
            mean_ok = abs(current_mean - config.mean) < (abs(config.mean) * config.accuracy_threshold)
            std_ok = abs(current_std - config.std) < (config.std * config.accuracy_threshold)

            if mean_ok and std_ok:
                break
            
            # Rescale and shift the data
            if current_std > EPSILON:
                data = config.mean + (data - current_mean) * (config.std / current_std)
            else:
                # Handle case where all values are the same
                data = np.full_like(data, config.mean)
        
        # --- 4. Final Adjustments ---
        # Clip data to ensure it's within the min/max bounds
        data = np.clip(data, config.min_val, config.max_val)
        
        # Ensure min and max values are present in the final distribution
        # This can slightly alter the final moments but guarantees the range.
        if data.min() > config.min_val:
            data[np.argmin(data)] = config.min_val
        if data.max() < config.max_val:
            data[np.argmax(data)] = config.max_val

        return data
    
    
    @staticmethod
    def evaluate_classification(
        true_labels: np.ndarray,
        pred_probs: np.ndarray,
        threshold: float = 0.5,
        figsize: Tuple[int, int] = (16, 7),
    ) -> Dict[str, Any]:
        """
        Calculates, prints, and visualizes metrics for a binary classification model.

        This "all-in-one" method provides a complete performance summary, including
        key scalar metrics, a classification report, a confusion matrix, and
        plots for ROC and Precision-Recall curves.

        Args:
            true_labels: Array of true binary labels (0 or 1).
            pred_probs: Array of predicted probabilities for the positive class.
            threshold: The cutoff to convert probabilities into binary predictions.
            figsize: The size of the figure for the plots.

        Returns:
            A dictionary containing the calculated metrics for programmatic use.
        """
        # --- 1. Input Validation ---
        if not isinstance(true_labels, np.ndarray) or not isinstance(pred_probs, np.ndarray):
            raise TypeError('Inputs `true_labels` and `pred_probs` must be NumPy arrays.')
        if true_labels.shape != pred_probs.shape:
            raise ValueError('Shape of true_labels and pred_probs must match.')
            
        # --- 2. Threshold-dependent Metrics ---
        pred_labels = (pred_probs >= threshold).astype(int)
        
        accuracy = accuracy_score(true_labels, pred_labels)
        report_dict = classification_report(
            true_labels, pred_labels, output_dict=True, zero_division=0
        )
        conf_matrix = confusion_matrix(true_labels, pred_labels)

        # --- 3. Threshold-independent Metrics ---
        fpr, tpr, _ = roc_curve(true_labels, pred_probs)
        ks = max(tpr - fpr) 
        roc_auc = auc(fpr, tpr)
        avg_precision = average_precision_score(true_labels, pred_probs)
        precision, recall, _ = precision_recall_curve(true_labels, pred_probs)

        # --- 4. Console Output ---
        print('*' * 60)
        print(f"{'CLASSIFICATION METRICS SUMMARY (Threshold = ' + str(threshold) + ')':^60}")
        print('*' * 60)
        print(f"  - Accuracy          : {accuracy:.4f}")
        print(f"  - ROC AUC           : {roc_auc:.4f}")
        print(f"  - Average Precision : {avg_precision:.4f}")
        print(f"  - Kolmogorov-Smirnov : {ks:.4f}")
        print('-' * 60)
        
        print(f"\n{'Classification Report':^60}\n")
        report_df = pd.DataFrame(report_dict).transpose()
        print(report_df.round(4))
        
        print(f"\n{'Confusion Matrix':^60}\n")
        print(pd.DataFrame(
            conf_matrix,
            index=['Actual 0', 'Actual 1'],
            columns=['Predicted 0', 'Predicted 1']
        ))
        print('*' * 60)

        # --- 5. Visualization ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Precision-Recall Curve
        ax1.step(recall, precision, color='b', alpha=0.8, where='post')
        ax1.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        ax1.set_xlabel('Recall', fontsize=14)
        ax1.set_ylabel('Precision', fontsize=14)
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlim([0.0, 1.0])
        ax1.set_title(f'Precision-Recall Curve\nAP = {avg_precision:.2f}', fontsize=16)
        
        # ROC Curve
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}, KS = {ks:.2f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.fill_between(fpr, tpr, fpr, 
                 where=(tpr >= fpr),
                 alpha=0.3, 
                 color='green', 
                 interpolate=True,
                 label='Above random')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate', fontsize=14)
        ax2.set_ylabel('True Positive Rate', fontsize=14)
        ax2.set_title('Receiver Operating Characteristic (ROC)', fontsize=16)
        ax2.legend(loc='lower right', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        # --- 6. Return Metrics Dictionary ---
        return {
            'threshold': threshold,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'Kolmogorov-Smirnov': ks,
            'classification_report': report_dict,
            'confusion_matrix': conf_matrix,
        }
    
    
    @staticmethod
    def grubbs_test(
        x: Union[np.ndarray, pd.Series],
        alpha: float = 0.05
    ) -> GrubbsTestResult:
        """
        Performs Grubbs' test to identify a single outlier in a dataset.

        This test assumes the data comes from a normally distributed population
        and is designed to detect one outlier at a time.

        Args:
            x: A 1D NumPy array or Pandas Series of numerical data.
            alpha: The significance level for the test (default: 0.05).

        Returns:
            A Pydantic model (GrubbsTestResult) containing the test results,
            including a boolean flag for outlier detection and the outlier's value
            and index if found.
            
        Raises:
            ValueError: If the input array has fewer than 3 elements.
        
        Usage:
            tools = DSTools()
        
            # Test 1: Data with an outlier
            print("\nTesting on data WITH an outlier:")
            result1 = tools.grubbs_test(data_with_outlier)
            print(f"  Calculated G-statistic: {result1.g_calculated:.4f}")
            print(f"  Critical G-value: {result1.g_critical:.4f}")
            if result1.is_outlier:
                print(f"Outlier detected: The value is {result1.outlier_value:.2f} at index {result1.outlier_index}.")
            else:
                print("No outlier detected.")

            # Test 2: Data without an outlier
            print("\nTesting on data WITHOUT an outlier:")
            result2 = tools.grubbs_test(data_without_outlier)
            print(f"  Calculated G-statistic: {result2.g_calculated:.4f}")
            print(f"  Critical G-value: {result2.g_critical:.4f}")
            if result2.is_outlier:
                print(f"Outlier detected, but shouldn't have been.")
            else:
                print("Correctly determined that there are no outliers.")
        """
        if not isinstance(x, (np.ndarray, pd.Series)):
            raise TypeError('Input data `x` must be a NumPy array or Pandas Series.')
            
        # Grubbs' test requires at least 3 data points
        n = len(x)
        if n < 3:
            raise ValueError('Grubbs\' test requires at least 3 data points.')

        # Convert to numpy array for calculations
        data = np.array(x)

        # 1. Calculate the G-statistic
        mean_x = np.mean(data)
        std_x = np.std(data, ddof=1) # Use sample standard deviation
        
        if np.isclose(std_x, 0):
            # If all values are the same, there are no outliers
            return GrubbsTestResult(
                is_outlier=False,
                g_calculated=0.0,
                g_critical=np.inf, # Critical value is irrelevant here
                outlier_value=None,
                outlier_index=None
            )

        max_deviation_index = np.argmax(np.abs(data - mean_x))
        max_deviation_value = data[max_deviation_index]
        
        numerator = np.abs(max_deviation_value - mean_x)
        g_calculated = numerator / std_x

        # 2. Calculate the critical G-value
        t_value = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        
        numerator_critical = (n - 1) * t_value
        denominator_critical = np.sqrt(n * (n - 2 + t_value**2))
        g_critical = numerator_critical / denominator_critical

        # 3. Compare and determine the result
        is_outlier_detected = g_calculated > g_critical

        return GrubbsTestResult(
            is_outlier=is_outlier_detected,
            g_calculated=g_calculated,
            g_critical=g_critical,
            outlier_value=max_deviation_value if is_outlier_detected else None,
            outlier_index=int(max_deviation_index) if is_outlier_detected else None
        )
    
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        class_labels: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (8, 8),
        title: str = 'Confusion Matrix',
        cmap: str = 'Blues'
    ):
        """
        Plots a clear and readable confusion matrix using seaborn.

        This method visualizes the performance of a classification model by showing
        the number of correct and incorrect predictions for each class.

        Args:
            y_true: Array-like of true labels.
            y_pred: Array-like of predicted labels.
            class_labels: Optional list of strings to use as labels for the axes.
                          If None, integer labels will be used.
            figsize: Tuple specifying the figure size.
            title: The title for the plot.
            cmap: The colormap to use for the heatmap.
        Usage:
            tools = DSTools()
            
            tools.plot_confusion_matrix(
                y_true_binary, 
                y_pred_binary, 
                class_labels=['Negative (0)', 'Positive (1)'],
                title='Binary Confusion Matrix'
            )
            tools.plot_confusion_matrix(
                y_true_multi,
                y_pred_multi,
                class_labels=['Cat', 'Dog', 'Bird'],
                title='Multi-Class Confusion Matrix'
            )
        """
        # 1. Calculate the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # 2. Determine labels for the axes
        if class_labels:
            if len(class_labels) != cm.shape[0]:
                raise ValueError(
                    f'Number of class_labels ({len(class_labels)}) does not match '
                    f'number of classes in confusion matrix ({cm.shape[0]}).'
                )
            labels = class_labels
        else:
            labels = np.arange(cm.shape[0])

        # 3. Create the plot using seaborn's heatmap for better aesthetics
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            cm,
            annot=True,       # Display the numbers in the cells
            fmt='d',          # Format numbers as integers
            cmap=cmap,        # Use the specified colormap
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,            # Draw on our created axes
            annot_kws={'size': 14} # Increase annotation font size
        )
        
        # 4. Set titles and labels for clarity
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.set_ylabel('True Label', fontsize=14)
        
        # Rotate tick labels for better readability if they are long
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.show()
    
    
    @staticmethod
    def add_missing_value_features(
        X: Union[pd.DataFrame, pl.DataFrame],
        add_std: bool = False
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Adds features based on the count of missing values per row.

        This preprocessing function calculates the number of missing values (NaN)
        for each row and adds this count as a new feature. This can significantly
        improve the performance of some machine learning models.

        Args:
            X: The input DataFrame (Pandas or Polars).
            add_std: If True, also adds the standard deviation of the nullity
                     mask as a feature (rarely used).

        Returns:
            A new DataFrame with the added feature(s). The original DataFrame
            is not modified.
            
        Usage:
            tools = DSTools()
            pd_with_features = tools.add_missing_value_features(pd_data)
            print("\nPandas DataFrame with new feature:")
            print(pd_with_features)
            
            pl_with_features = tools.add_missing_value_features(pl_data)
            print("\nPolars DataFrame with new feature:")
            print(pl_with_features)
        """
        if isinstance(X, pd.DataFrame):
            # --- Pandas Implementation ---
            # Create a copy to avoid modifying the original DataFrame
            X_new = X.copy()
            
            # Calculate the number of missing values per row
            num_missing = X.isnull().sum(axis=1)
            X_new['num_missing'] = num_missing
            
            if add_std:
                # Note: std of a boolean mask is often not very informative
                num_missing_std = X.isnull().std(axis=1)
                X_new['num_missing_std'] = num_missing_std
                
            return X_new

        elif isinstance(X, pl.DataFrame):
            # --- Polars Implementation (more efficient) ---
            # Polars expressions are highly optimized
            string_cols = [col for col, dtype in X.schema.items() if dtype in [pl.String, pl.Utf8]]
            numeric_cols = [col for col in X.columns if col not in string_cols]
            
            expressions = []            
            if string_cols:
                for col in string_cols:
                    expr = (pl.col(col).is_null()) | (pl.col(col) == "NaN")
                    expressions.append(expr)
            
            if numeric_cols:
                for col in numeric_cols:
                    expr = pl.col(col).is_null()
                    expressions.append(expr)
                    
            result_pl = X.with_columns(pl.sum_horizontal(expressions).alias('num_missing'))
            
            if add_std:
                missing_matrix = X.select(expressions)
                std_per_row = missing_matrix.select(
                    pl.concat_list(pl.all()).list.eval(
                        pl.element().cast(pl.Float64).std()
                    ).alias('num_missing_std')
                )
                result_pl = result_pl.with_columns(std_per_row)

            return result_pl
        
        else:
            raise TypeError('Input `X` must be a Pandas or Polars DataFrame.')
    
    
    @staticmethod
    def chatterjee_correlation(
        x: Union[np.ndarray, pd.Series, List[float]],
        y: Union[np.ndarray, pd.Series, List[float]],
        standard_flag: bool = True
    ) -> float:
        """
        Calculates Chatterjee's rank correlation coefficient (Xi).

        This coefficient is a non-parametric measure of dependence between two
        variables. It is asymmetric and ranges from 0 to 1, where a value
        close to 1 indicates that y is a function of x. It can capture
        non-linear relationships.

        Args:
            x: Array-like, the first variable (independent).
            y: Array-like, the second variable (dependent).
            standard_flag: bool flag which define type of calculation

        Returns:
            The Chatterjee's correlation coefficient, a float between 0 and 1.
            
        Raises:
            ValueError: If the input arrays do not have the same length.
            
        Usage:
             x = np.linspace(0, 10, 100)
             y_linear = 2 * x + 1
             y_nonlinear = np.sin(x)
             tools = DSTools()
             print(f"Linear correlation: {tools.chatterjee_correlation(x, y_linear):.4f}")
             print(f"Non-linear correlation: {tools.chatterjee_correlation(x, y_nonlinear):.4f}")
        """
        # 1. Convert inputs to NumPy arrays and validate
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        
        n = len(x_arr)
        if n != len(y_arr):
            raise ValueError('Input arrays x and y must have the same length.')
        if n < 2:
            return 0.0 # Correlation is undefined for less than 2 points

        # 2. Get the ranks of y based on the sorted order of x
        # argsort gives the indices that would sort x
        x_order_indices = np.argsort(x_arr)
        
        # Reorder y according to the sorted x
        y_ordered_by_x = y_arr[x_order_indices]
        
        # Calculate ranks of the reordered y. 'average' method handles ties.
        # This replaces the dependency on pandas.Series.rank()
        y_ranks = rankdata(y_ordered_by_x, method='average')
        
        # 3. Calculate the sum of absolute differences of consecutive ranks
        # np.diff calculates the difference between adjacent elements
        rank_diffs_sum = np.sum(np.abs(np.diff(y_ranks)))
        
        # 4. Calculate Chatterjee's Xi coefficient
        # The original formula is 1 - (3 * sum(|r_{i+1} - r_i|)) / (n^2 - 1)
        # An equivalent and more stable formula is used below.
        xi_orig = 1 - (n * rank_diffs_sum) / (2 * np.sum(np.abs(y_ranks - np.mean(y_ranks))**2))
        
        xi = 1 - (3 * rank_diffs_sum) / (n**2 - 1) if standard_flag else xi_orig

        return xi
    
    
    @staticmethod
    def calculate_entropy(
        p: Union[np.ndarray, List[float]],
        base: Optional[float] = None
    ) -> float:
        """
        Calculates the Shannon entropy of a probability distribution.

        Entropy measures the uncertainty or "surprise" inherent in a variable's
        possible outcomes.

        Args:
            p: A 1D array-like object representing a probability distribution.
               The sum of its elements should be close to 1.
            base: The logarithmic base to use for the calculation. If None (default),
                  the natural logarithm (ln) is used, and the result is in "nats".
                  Use base=2 for the result in "bits".

        Returns:
            The calculated entropy as a float.
            
        Raises:
            ValueError: If the input array contains negative values.
            
        Usage:
            tools = DSTools()
            print("\nCalculating Entropy (in nats):")
            entropy_a = tools.calculate_entropy(dist_a)
            entropy_uniform = tools.calculate_entropy(dist_uniform)
            print(f"  - Entropy of A [0.1, 0.2, 0.7]: {entropy_a:.4f}")
            print(f"  - Entropy of Uniform [0.33, 0.33, 0.33]: {entropy_uniform:.4f} (should be highest)")
            
            entropy_a_bits = tools.calculate_entropy(dist_a, base=2)
            print(f"  - Entropy of A in bits: {entropy_a_bits:.4f}")
        """
        p_arr = np.asarray(p, dtype=float)
        
        if np.any(p_arr < 0):
            raise ValueError('Probabilities cannot be negative.')
            
        # Normalize the distribution to ensure it sums to 1
        p_arr /= np.sum(p_arr)
        
        # Filter out zero probabilities to avoid issues with log(0)
        p_arr = p_arr[p_arr > 0]
        
        if base is None:
            # Use natural logarithm (nats)
            return -np.sum(p_arr * np.log(p_arr))
        else:
            # Use specified base
            return -np.sum(p_arr * np.log(p_arr) / np.log(base))


    @staticmethod
    def calculate_kl_divergence(
        p: Union[np.ndarray, List[float]],
        q: Union[np.ndarray, List[float]],
        base: Optional[float] = None
    ) -> float:
        """
        Calculates the Kullback-Leibler (KL) divergence between two distributions.

        KL divergence D_KL(P || Q) measures how one probability distribution P
        diverges from a second, expected probability distribution Q. It is
        asymmetric.

        Args:
            p: A 1D array-like object for the "true" or reference distribution (P).
            q: A 1D array-like object for the "approximating" distribution (Q).
            base: The logarithmic base to use. Defaults to natural log (nats).

        Returns:
            The KL divergence as a float.
            
        Raises:
            ValueError: If input arrays have different lengths or contain negative values.
            
        Usage:
            print("\nCalculating KL Divergence (D_KL(P || Q)):")
            # Divergence of a distribution from itself should be 0
            kl_a_a = tools.calculate_kl_divergence(dist_a, dist_a)
            print(f"  - KL(A || A): {kl_a_a:.4f} (should be 0)")
            
            # Divergence of A from B (B is a good approximation of A)
            kl_a_b = tools.calculate_kl_divergence(dist_a, dist_b)
            print(f"  - KL(A || B): {kl_a_b:.4f} (should be small)")

            # Divergence of A from C (C is a bad approximation of A)
            kl_a_c = tools.calculate_kl_divergence(dist_a, dist_c)
            print(f"  - KL(A || C): {kl_a_c:.4f} (should be large)")
            
            # Note that KL divergence is asymmetric
            kl_c_a = tools.calculate_kl_divergence(dist_c, dist_a)
            print(f"  - KL(C || A): {kl_c_a:.4f} (note: not equal to KL(A || C))")
        """
        p_arr = np.asarray(p, dtype=float)
        q_arr = np.asarray(q, dtype=float)

        if p_arr.shape != q_arr.shape:
            raise ValueError('Input distributions P and Q must have the same shape.')
        if np.any(p_arr < 0) or np.any(q_arr < 0):
            raise ValueError('Probabilities cannot be negative.')
        
        # Normalize both distributions
        p_arr /= np.sum(p_arr)
        q_arr /= np.sum(q_arr)
        
        # Add a small epsilon to avoid division by zero or log(0)
        # We only need to protect q from being zero where p is non-zero
        p_arr += EPSILON
        q_arr += EPSILON
        
        if base is None:
            return np.sum(p_arr * np.log(p_arr / q_arr))
        else:
            return np.sum(p_arr * (np.log(p_arr / q_arr) / np.log(base)))
    
    
    @staticmethod
    def min_max_scale(
        df: Union[pd.DataFrame, pl.DataFrame],
        columns: Optional[List[str]] = None,
        const_val_fill: float = 0.0
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Scales specified columns of a DataFrame to the range [0, 1].

        This method applies Min-Max scaling. If a column contains identical
        values, it will be filled with `const_val_fill`. The original
        DataFrame is not modified.

        Args:
            df: The input DataFrame (Pandas or Polars).
            columns: A list of column names to scale. If None (default),
                     all numerical columns will be scaled.
            const_val_fill: The value to use for columns where all values
                            are identical (to avoid division by zero).

        Returns:
            A new DataFrame with the specified columns scaled.
            
        Usage:
            tools = DSTools()
            pd_scaled = tools.min_max_scale(pd_data, columns=['a', 'c'])
            print("\nPandas DataFrame with scaled columns 'a' and 'c':")
            print(pd_scaled)
            
            pl_scaled = tools.min_max_scale(pl_data) # Scale all numeric columns
            print("\nPolars DataFrame with all numeric columns scaled:")
            print(pl_scaled)
            
            pl_scaled_half = tools.min_max_scale(pl_data, const_val_fill=0.5)
            print("\nPolars DataFrame with constant columns filled with 0.5:")
            print(pl_scaled_half)
        """
        if isinstance(df, pd.DataFrame):
            # --- Pandas Implementation ---
            df_new = df.copy()
            
            if columns is None:
                # Select all numeric columns if none are specified
                columns = df_new.select_dtypes(include=np.number).columns.tolist()

            for col in columns:
                if col not in df_new.columns:
                    print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
                    continue
                
                min_val = df_new[col].min()
                max_val = df_new[col].max()
                
                if min_val == max_val:
                    df_new[col] = const_val_fill
                else:
                    df_new[col] = (df_new[col] - min_val) / (max_val - min_val)
            
            return df_new

        elif isinstance(df, pl.DataFrame):
            # --- Polars Implementation ---
            
            if columns is None:
                # Select all numeric columns
                columns = [
                    col for col, dtype in df.schema.items() if dtype.is_numeric()
                ]

            expressions = []
            for col in columns:
                if col not in df.columns:
                    print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
                    continue
                    
                min_val = pl.col(col).min()
                max_val = pl.col(col).max()
                
                # Use a 'when/then/otherwise' expression for conditional logic
                expr = (
                    pl.when(min_val == max_val)
                    .then(pl.lit(const_val_fill))
                    .otherwise((pl.col(col) - min_val) / (max_val - min_val))
                    .alias(col) # Keep the original column name
                )
                expressions.append(expr)
            
            return df.with_columns(expressions)
        
        else:
            raise TypeError('Input dataframe must be a Pandas or Polars DataFrame.')
    
    
    @staticmethod
    def save_dataframes_to_zip(
        dataframes: Dict[str, Union[pd.DataFrame, pl.DataFrame]],
        zip_filename: str,
        format: str = 'parquet',
        save_index: bool = False
    ):
        """
        Saves one or more Pandas or Polars DataFrames into a single ZIP archive.

        Args:
            dataframes: A dictionary where keys are the desired filenames (without
                        extension) and values are the Pandas or Polars DataFrames.
            zip_filename: The path for the output ZIP archive.
            format: The format to save the data in ('parquet', 'csv').
            save_index: For Pandas DataFrames, whether to save the index.
                        (Ignored for Polars).
        
        Usage:
            tools = DSTools()
            dfs_to_save = {
                'pandas_data': pd_df,
                'polars_data': pl_df
            }
            zip_path = 'mixed_data_archive.zip'
            print("\n--- Saving mixed DataFrames ---")
            tools.save_dataframes_to_zip(dfs_to_save, zip_path, format='parquet', save_index=True)
            
            print("\n--- Reading back with Polars backend ---")
            loaded_with_polars = tools.read_dataframes_from_zip(zip_path, format='parquet', backend='polars')
            print("DataFrame 'pandas_data' loaded by Polars:")
            print(loaded_with_polars['pandas_data']) # The index will be lost because Polars does not have it.

            print("\n--- Reading back with Pandas backend ---")
            loaded_with_pandas = tools.read_dataframes_from_zip(zip_path, format='parquet', backend='pandas')
            print("DataFrame 'pandas_data' loaded by Pandas:")
            print(loaded_with_pandas['pandas_data']) # The index will be restored
            
            if os.path.exists(zip_path):
            os.remove(zip_path)
        """
        if not isinstance(dataframes, dict):
            raise TypeError('`dataframes` must be a dictionary of {filename: df}.')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = []
            
            for name, df in dataframes.items():
                safe_name = re.sub(r'[\\/*?:"<>|]', '_', name)
                file_path = os.path.join(temp_dir, f'{safe_name}.{format}')
                file_paths.append(file_path)

                # --- Dispatch based on DataFrame type ---
                if isinstance(df, pd.DataFrame):
                    if format == 'parquet':
                        df.to_parquet(file_path, index=save_index, engine='fastparquet')
                    elif format == 'csv':
                        df.to_csv(file_path, index=save_index)
                    else:
                        raise ValueError(f"Unsupported format: '{format}'.")
                elif isinstance(df, pl.DataFrame):
                    if format == 'parquet':
                        df.write_parquet(file_path)
                    elif format == 'csv':
                        df.write_csv(file_path)
                    else:
                        raise ValueError(f"Unsupported format: '{format}'.")
                else:
                    raise TypeError(f"Unsupported DataFrame type: {type(df)}")
            
            # Create the ZIP archive
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for path in file_paths:
                    zipf.write(path, os.path.basename(path))

        print(f'Successfully saved {len(dataframes)} DataFrame(s) to {zip_filename}')


    @staticmethod
    def read_dataframes_from_zip(
        zip_filename: str,
        format: str = 'parquet',
        backend: str = 'polars'
    ) -> Dict[str, Union[pd.DataFrame, pl.DataFrame]]:
        """
        Reads one or more DataFrames from a ZIP archive.

        Args:
            zip_filename: The path to the ZIP archive.
            format: The format of the files inside the ZIP ('parquet', 'csv').
            backend: The library to use for reading ('polars' or 'pandas').

        Returns:
            A dictionary where keys are the filenames (without extension) and
            values are the loaded DataFrames.
        """
        if backend not in ['polars', 'pandas']:
            raise ValueError("`backend` must be 'polars' or 'pandas'.")
        
        loaded_dataframes = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_filename, 'r') as zipf:
                zipf.extractall(temp_dir)

            extension = f'.{format}'
            for filename in os.listdir(temp_dir):
                if filename.endswith(extension):
                    file_path = os.path.join(temp_dir, filename)
                    name_without_ext = os.path.splitext(filename)[0]
                    
                    # --- Dispatch based on a chosen backend ---
                    if backend == 'polars':
                        df = pl.read_parquet(file_path) if format == 'parquet' else pl.read_csv(file_path)
                    else: # backend == 'pandas'
                        if format == 'parquet':
                            df = pd.read_parquet(file_path)
                        else:
                            try:
                                df = pd.read_csv(file_path, index_col=0)
                            except (ValueError, IndexError):                                
                                df = pd.read_csv(file_path)
                    
                    loaded_dataframes[name_without_ext] = df

        print(f"Successfully loaded {len(loaded_dataframes)} DataFrame(s) using {backend}.")
        return loaded_dataframes
    
    
    @staticmethod
    def generate_alphanum_codes(n: int, length: int = 8) -> np.ndarray:
        """
        Generates an array of random alphanumeric codes.

        This method is optimized for performance by using NumPy vectorized operations.

        Args:
            n: The number of codes to generate.
            length: The length of each code.

        Returns:
            A NumPy array of strings, where each string is a random code.
            
        Usage:
            tools = DSTools()
            codes = tools.generate_alphanum_codes(5, length=10)
            print(f"Generated codes:\n{codes}")
        """
        if n < 0 or length < 0:
            raise ValueError('Number of codes (n) and length must be non-negative.')

        if length == 0:
            return np.full(n, '', dtype=str)
        
        # A clean, non-repeating alphabet
        alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        alphabet_len = len(alphabet)
        
        # Generate all random indices at once
        random_indices = np.random.randint(0, alphabet_len, size=(n, length))
        
        # Use NumPy's advanced indexing to get characters
        # .view('S1') treats each character as a 1-byte string
        # .reshape converts back to the desired shape
        codes_as_chars = np.array(list(alphabet), dtype='S1')[random_indices]
        
        # .view('S{length}') joins the characters in each row into a single string
        # This is a highly optimized, low-level NumPy operation
        codes_as_bytes = codes_as_chars.view(f'S{length}')
        
        # Decode from bytes to a standard UTF-8 string array
        return np.char.decode(codes_as_bytes.flatten(), 'utf-8')
    
    
    def generate_distribution_from_metrics(
        self,
        n: int,
        metrics: Union[DistributionConfig, Dict[str, float]],
        int_flag: bool = True,
        output_as: Literal['numpy', 'pandas', 'polars'] = 'numpy',
        max_iterations: int = 100
    ) -> Union[np.ndarray, pd.Series, pl.Series]:
        """
        Generates a synthetic distribution matching given statistical metrics.

        This function uses an iterative approach to create a distribution that
        approximates the properties specified in the DistributionConfig.

        Args:
            n: Number of values to generate.
            metrics: A Pydantic `DistributionConfig` instance OR a dictionary
                     with the target statistical properties.
            int_flag: If True, returns integer values; otherwise, floats.
            output_as: The desired output format ('numpy', 'pandas', or 'polars').
            max_iterations: int value is a number of iterations for tries.

        Returns:
            An array or Series of generated values.
            
        Usage:
            tools = DSTools()
            try:
                metrics_dict = DistributionConfig(
                    mean=1042,
                    median=330,
                    std=1500,
                    min_val=1,
                    max_val=120000,
                    skewness=13.2,
                    kurtosis=245,
                    n=10000,
                    accuracy_threshold=0.05,
                    outlier_ratio=0.05
                )

                # 2. Generate the data using the metrics object
                generated_data = tools.generate_distribution_from_metrics(
                    n=1000,
                    metrics=metrics_dict,
                    int_flag=True,
                    output_as='numpy'
                )
                
                # 3. Analyze the result
                print("--- Target vs. Actual Statistics ---")
                print(f"Target Mean: {metrics.mean}, Actual Mean: {np.mean(generated_data):.2f}")
                print(f"Target Median: {metrics.median}, Actual Median: {np.median(generated_data):.2f}")
                print(f"Target Std: {metrics.std}, Actual Std: {np.std(generated_data):.2f}")
                print(f"Target Skew: {metrics.skewness}, Actual Skew: {stats.skew(generated_data):.2f}")
                print(f"Target Kurtosis: {metrics.kurtosis}, Actual Kurtosis: {stats.kurtosis(generated_data, fisher=False):.2f}")
                print(f"Target Min: {metrics.min_val}, Actual Min: {np.min(generated_data):.2f}")
                print(f"Target Max: {metrics.max_val}, Actual Max: {np.max(generated_data):.2f}")
                
            except ValueError as e:
                print(f"Error during configuration or generation: {e}")
        """
        if isinstance(metrics, dict):
            try:
                config = DistributionConfig(**metrics)
            except Exception as e:
                raise ValueError(f"Invalid metrics dictionary: {e}")
        elif isinstance(metrics, DistributionConfig):
            config = metrics
        else:
            raise TypeError("`metrics` must be a dict or a DistributionConfig instance.")
        
        if config.n is None:
             config.n = n
        elif config.n != n:
             print(f"Warning: `n` provided in both arguments ({n}) and config ({config.n}). "
                   f"Using value from arguments: {n}.")
             config.n = n
        
        if not self.validate_moments(config.std, config.skewness, config.kurtosis):
            raise ValueError(
                'Invalid moments. Check that std > 0 and kurtosis >= (skewness² - 2).'
            )

        # --- 1. Initial Data Generation ---
        num_outliers = int(config.n * config.outlier_ratio)
        num_base = config.n - num_outliers

        # Generate the main part of the distribution
        # Use a non-central t-distribution to introduce initial skew
        nc = config.skewness * (config.kurtosis / 3.0) # Heuristic for non-centrality
        df = max(5, int(6 + 2 * (config.kurtosis - 3))) # Degrees of freedom influence kurtosis
        
        base_data = stats.nct.rvs(df=df, nc=nc, size=num_base)
        
        # Generate outliers to control the tails
        if num_outliers > 0:
            # Generate outliers from a wider normal distribution
            outlier_scale = config.std * (1.5 + config.kurtosis / 5.0)
            outliers = np.random.normal(loc=config.mean, scale=outlier_scale, size=num_outliers)
            data = np.concatenate([base_data, outliers])
        else:
            data = base_data
            
        np.random.shuffle(data)
        
        # Initial scaling to get closer to the target
        data = config.mean + (data - np.mean(data)) * (config.std / (np.std(data) + EPSILON))

        # --- 2. Iterative Adjustment ---        
        for _ in range(max_iterations):
            # Calculate current moments
            current_mean = np.mean(data)
            current_std = np.std(data, ddof=1)
            current_median = np.median(data)

            # Check for convergence on primary metrics (mean, std, median)
            mean_ok = abs(current_mean - config.mean) < (abs(config.mean) * config.accuracy_threshold)
            std_ok = abs(current_std - config.std) < (config.std * config.accuracy_threshold)
            median_ok = abs(current_median - config.median) < (abs(config.median) * config.accuracy_threshold)
            
            if mean_ok and std_ok and median_ok:
                break

            # Adjustment for mean and std (rescale and shift)
            if current_std > EPSILON:
                data = config.mean + (data - current_mean) * (config.std / current_std)
            
            # Adjustment for median (gentle push towards the target)
            median_diff = config.median - np.median(data)
            # Apply a non-linear shift to move the median without ruining the mean/std too much
            data += median_diff * np.exp(-((data - np.median(data)) / config.std)**2) * 0.1

        # --- 3. Finalization ---
        # Final clipping and type casting
        data = np.clip(data, config.min_val, config.max_val)
        
        # Ensure min/max values are present
        if data.min() > config.min_val:
            data[np.argmin(data)] = config.min_val
        if data.max() < config.max_val:
            data[np.argmax(data)] = config.max_val
            
        if int_flag:
            data = np.round(data).astype(np.int64)

        # Convert to desired output format
        if output_as == 'pandas':
            return pd.Series(data, name='generated_values')
        elif output_as == 'polars':
            return pl.Series(name='generated_values', values=data)
        else:
            return data
    
    
    
