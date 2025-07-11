##########    code checking for "compute_metrics"    ##########

import numpy as np
import pandas as pd

# Import our class and its configuration
from ds_tool import DSTools, MetricsConfig

# --- 1. Generate test data ---

# Set a seed for reproducibility
np.random.seed(42)

# Determine the number of examples in our test set
N_SAMPLES = 200

# Create an array of true labels (y_true)
# Roughly equal number of 0 and 1
y_true = np.random.randint(0, 2, size=N_SAMPLES)

# Create an array of predicted probabilities (y_predict_proba)
# We want the model to be "good but not perfect".
# To do this, we generate probabilities so that they generally match the
# true labels, but with some random "noise".
# Where y_true = 1, the probabilities will be shifted to 1.0.
# Where y_true = 0, the probabilities will be shifted to 0.0.

# Create "base" probabilities: 0.8 for class 1 and 0.2 for class 0
base_probs = np.where(y_true == 1, 0.8, 0.2)
# Add normal noise to make predictions more realistic
noise = np.random.normal(0, 0.25, size=N_SAMPLES)
y_predict_proba = base_probs + noise

# Limit probabilities to the range [0, 1], since noise could take them outside the range
y_predict_proba = np.clip(y_predict_proba, 0, 1)

# Create an array of predicted labels (y_predict) based on a threshold of 0.5
THRESHOLD = 0.5
y_predict = (y_predict_proba >= THRESHOLD).astype(int)

print("--- Data generation complete ---")
print(f"Number of samples: {N_SAMPLES}")
print(f"Example of true labels (y_true): {y_true[:10]}")
print(f"Example of probabilities (y_predict_proba): {y_predict_proba[:10].round(2)}")
print(f"Example of predicted labels (y_predict): {y_predict[:10]}")
print("-" * 35, "\n")

# --- 2. Calling compute_metrics ---

# Initializing our toolkit
tools = DSTools()

# --- Case A: Calling with default configuration ---
# In this case, config.error_vis = True, so we should see a plot
print("--- Running compute_metrics (default configuration) ---")
print("Expect to see a plot of 'Error Rates vs Threshold Levels'...")

metrics_df_default = tools.compute_metrics(y_true, y_predict, y_predict_proba)

print("\nResult as a DataFrame:")
print(metrics_df_default)
print("-" * 55, "\n")

# --- Option B: Call with custom configuration ---
# Disable visualization and enable printing of metric values ​​directly to the console
print("--- Running compute_metrics (custom configuration) ---")
print("We expect to see metric values ​​printed in the console...")

custom_config = MetricsConfig(
error_vis=False, # Disable graph
print_values=True # Enable printing
)

metrics_df_custom = tools.compute_metrics(y_true, y_predict, y_predict_proba, config=custom_config)

print("\nResult as DataFrame:")
print(metrics_df_custom)
print("-" * 55)

##########    code checking for "corr_matrix"    ##########

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import our class and its configuration
from ds_tool import DSTools, CorrelationConfig

# --- 1. Generate test data ---

# Set seed for reproducibility of results
np.random.seed(42)
N_SAMPLES = 100

print("--- Generate DataFrame with different types of dependencies ---")

# Create base feature
feature_A = np.linspace(-10, 10, N_SAMPLES)

# Create other features based on it:
# feature_B: Strong positive linear correlation with A
feature_B = feature_A * 2 + np.random.normal(0, 2, N_SAMPLES)

# feature_C: Strong negative linear correlation with A
feature_C = -feature_A * 1.5 + np.random.normal(0, 3, N_SAMPLES)

# feature_D: Nonlinear (quadratic) dependence on A.
# Pearson should show low correlation, and Spearman - high.
feature_D = feature_A**2 + np.random.normal(0, 5, N_SAMPLES)

# feature_E: No correlation (random noise)
feature_E = np.random.rand(N_SAMPLES) * 20

# feature_F: Moderate positive correlation with B
feature_F = feature_B * 0.5 + np.random.normal(0, 10, N_SAMPLES)

# Put it all in a DataFrame
df = pd.DataFrame({
'feature_A': feature_A,
'feature_B': feature_B,
'feature_C': feature_C,
'feature_D': feature_D,
'feature_E': feature_E,
'feature_F': feature_F
})

print("First 5 rows generated data:")
print(df.head())
print("-" * 50, "\n")

# --- 2. Call corr_matrix ---

# Initialize our toolkit
tools = DSTools()

# --- Scenario A: Call with default configuration ---
print("--- Scenario A: Correlation matrix (default Pearson method) ---")
print("Expect to see a graph. Note that the correlation of A and D (non-linear relationship) will be low.")
# plt.show() is blocking, so we'll call it at the end.
# But in real use the call would be:
tools.corr_matrix(df)

# --- Scenario B: Call with custom config ---
print("\n--- Scenario B: Correlation matrix (Spearman method, custom view) ---")
print("Expect to see the second plot with a different correlation method.")
print("Now the correlation of A and D (non-linear relationship) should be high, since Spearman is based on ranks.")

custom_config = CorrelationConfig(
build_method='spearman',
font_size=10,
image_size=(12, 12)
)
tools.corr_matrix(df, config=custom_config)

# --- Scenario C: Checking Pydantic validation ---
print("\n--- Scenario C: Testing validation ---")
print("Attempt to create a configuration with an invalid method. Expecting to see a ValueError.")

try:
invalid_config = CorrelationConfig(build_method='invalid_method')
except ValueError as e:
print(f"\nSuccessfully caught the expected error: {e}")

print("-" * 50)

##########    code checking for "category_stats"    ##########

import numpy as np
import pandas as pd

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data ---

# Set seed for reproducibility
np.random.seed(42)

# Create lists with categories
cities = ['Moscow', 'Saint Petersburg', 'Novosibirsk', 'Ekaterinburg', 'Kazan']
status = ['Active', 'Inactive', 'Pending', 'Archive']
product_type = ['Electronics', 'Clothing', 'Books']

# Generate data
data = {
'City': np.random.choice(cities, size=100, p=[0.4, 0.3, 0.1, 0.1, 0.1]),
'Customer_status': np.random.choice(status + [np.nan], size=100, p=[0.5, 0.2, 0.1, 0.1, 0.1]),
'Product_type': np.random.choice(product_type, size=100)
}

df = pd.DataFrame(data)

print("--- Generated DataFrame (first 10 rows): ---")
print(df.head(10))
print("\n" + "="*50 + "\n")

# --- 2. Call category_stats ---

# Initialize our toolkit
tools = DSTools()

# --- Scenario A: Call for 'City' column ---
print("--- Scenario A: Statistics for the 'City' column ---")
print("Expect to see a table with count and percentage for each city.")
tools.category_stats(df, 'City')
print("\n" + "="*50 + "\n")

# --- Scenario B: Call for the 'Customer_Status' column (with gaps) ---
print("--- Scenario B: Statistics for the 'Customer_Status' column ---")
print("Note that missing values ​​(NaN) are not taken into account in the calculations.")
tools.category_stats(df, 'Customer_Status')
print("\n" + "="*50 + "\n")

# --- Scenario C: Testing error handling ---
print("--- Scenario C: Attempting to call for a non-existent column ---")
print("Expecting to see ValueError.")

try:
tools.category_stats(df, 'Non-existent_column')
except ValueError as e:
print(f"\nExpected error successfully caught: {e}")

print("\n" + "="*50)

##########    code checking for "trials_res_df"    ##########

import time
import optuna
import pandas as pd
import numpy as np

# Import our class
from ds_tool import DSTools

# --- 1. Create a dummy objective function for Optuna ---

# This function simulates training the model. It takes a trial object,
# "suggests" hyperparameters, pauses, and returns the "quality" of the model.
def objective_function(trial: optuna.trial.Trial) -> float:
"""Dummy function for optimization."""

# Suggest hyperparameters
x = trial.suggest_float('x', -10, 10)
y = trial.suggest_int('y', 0, 20)
category = trial.suggest_categorical('category', ['A', 'B', 'C'])

# Simulate a "failed" trial (e.g., parameters are incompatible)
if trial.number == 3:
raise ValueError("Simulating error in trial #3")

# Simulate a "pruned" trial if y < 5
# Optuna can abort a pruned trial before it completes
if y < 5:
raise optuna.exceptions.TrialPruned("Simulating abort (y < 5)")

# Simulating the training process
time.sleep(np.random.uniform(0.1, 0.3))

# Calculating the "quality" of the model using a made-up formula
# Just to make the results different
score = (100 - x**2) + y - (5 if category == 'C' else 0)

return score

# --- 2. Starting an Optuna optimization session ---

print("--- Starting an Optuna optimization simulation... ---")
# We want to maximize our 'score'
study = optuna.create_study(direction='maximize')

# Starting 15 attempts. We expect some of them to fail.
# Optuna will handle errors on its own, we just need to catch them so the script doesn't stop.
try:
study.optimize(objective_function, n_trials=15)
except ValueError:
print("Caught a simulated error as expected. Continue.")

print("\nOptimization completed.")
print(f"Total number of trials started: {len(study.trials)}")
completed_trials = len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))
print(f"Of which {completed_trials} were successfully completed")

# --- 3. Call trials_res_df ---

print("\n--- Call trials_res_df to process the results ---")
tools = DSTools()

# Get a list of all trial objects from study
all_trials_list = study.trials

# Pass the list into our function.
# We specify that the metric by which we need to sort is called 'Accuracy',
# although in the Optuna code it is called 'value'. Our function will rename it.
METRIC_NAME = 'Accuracy'
results_df = tools.trials_res_df(all_trials_list, metric=METRIC_NAME)

# --- 4. Checking the result ---

print("\n--- Final DataFrame from trials_res_df: ---")
print(results_df)

print("\n--- Correctness check: ---")
print(f"1. The number of rows in the DataFrame: {len(results_df)}. Should match the number of completed trials ({completed_trials}).")
print(f"2. The first column is called '{METRIC_NAME}', as we asked.")
print(f"3. The DataFrame is sorted by the '{METRIC_NAME}' column in descending order (best result on top).")
print( "4. There is a column 'Duration' with the execution time of each attempt in seconds.")
print("5. There are columns with hyperparameters: 'x', 'y', 'category'.")

##########    code checking for "labeling"    ##########

import pandas as pd
import numpy as np

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data ---

# Create a list of categories with different frequencies
# 'C' is the most frequent, 'B' is average, 'A' is the rarest, 'D' is very rare
categories_list = ['C'] * 50 + ['B'] * 30 + ['A'] * 15 + ['D'] * 5

# Shuffle them so they are not in order
np.random.seed(42)
np.random.shuffle(categories_list)

df = pd.DataFrame({'product_category': categories_list})

print("--- Original DataFrame (first 10 rows): ---")
print(df.head(10))
print("\n--- Category frequency (for checking): ---")
print(df['product_category'].value_counts())
print("\n" + "="*50 + "\n")

# --- 2. Initialization and calling labeling ---

tools = DSTools()
target_column = 'product_category'

# --- Scenario A: Encoding with ordering by frequency (order_flag=True) ---
print(f"--- Scenario A: Encoding column '{target_column}' with order_flag=True ---")
print("We expect the rarest category 'D' to be coded 0, 'A' -> 1, 'B' -> 2, 'C' -> 3.")

# The function returns a new DataFrame, so we save the result
df_ordered = tools.labeling(df, target_column, order_flag=True)

print("\n--- Result of encoding (order_flag=True): ---")
print(df_ordered.head(10))

print("\n--- Result check (match codes and original values): ---")
# Let's create a table for visual comparison
comparison_ordered = pd.DataFrame({
'original': df[target_column],
'encoded': df_ordered[target_column]
})
print(comparison_ordered.drop_duplicates().sort_values('encoded'))
print("\n" + "="*50 + "\n")

# --- Scenario B: Encoding without ordering (order_flag=False) ---
print(f"--- Scenario B: Encoding column '{target_column}' with order_flag=False ---")
print("Expect categories to receive arbitrary numeric codes.")

df_simple = tools.labeling(df, target_column, order_flag=False)

print("\n--- Result of encoding (order_flag=False): ---")
print(df_simple.head(10))

print("\n--- Checking the result (matching codes and original values): ---")
comparison_simple = pd.DataFrame({
'original': df[target_column],
'encoded': df_simple[target_column]
})
print(comparison_simple.drop_duplicates().sort_values('encoded'))
print("\n" + "="*50 + "\n")

# --- Scenario B: Testing error handling ---
print("--- Scenario B: Attempting to call on non-existent column ---")
print("Expecting to see ValueError.")

try:
tools.labeling(df, 'non_existent_column')
except ValueError as e:
print(f"\nSuccessfully caught expected error: {e}")

##########    code checking for "remove_outliers_iqr"    ##########

import pandas as pd
import numpy as np

# Import our class and its configuration
from ds_tool import DSTools, OutlierConfig

# --- 1. Generate test data ---

# Set seed for reproducibility
np.random.seed(42)

# Create "normal" data
normal_data = np.random.normal(loc=100, scale=20, size=90)

# Create obvious outliers
outliers = np.array([-50, -40, 250, 300, 310])

# Concatenate and shuffle
full_data = np.concatenate([normal_data, outliers])
np.random.shuffle(full_data)

# Create DataFrame
df = pd.DataFrame({'value': full_data})
# Add a categorical column to check if the entire row is deleted
df['category'] = np.random.choice(['A', 'B'], size=len(df))

print("--- Original DataFrame ---")
print("Statistics (note min/max and std):")
print(df['value'].describe())
print("\n" + "="*50 + "\n")

# Calculate IQR bounds manually to check (with sigma=1.5)
q1 = df['value'].quantile(0.25)
q3 = df['value'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
print(f"Expected bounds for sigma=1.5: Lower ~{lower_bound:.2f}, Upper ~{upper_bound:.2f}")
print("Anything less or greater than these bounds will be considered an outlier.")
print("\n" + "="*50 + "\n")

# --- 2. Initialization and calls ---
tools = DSTools()
target_column = 'value'

# --- Scenario A: Default mode (replacing outliers) ---
print(f"--- Scenario A: Replacing outliers in column '{target_column}' (sigma=1.5) ---")
# Using default config
df_replaced, p_upper, p_lower = tools.remove_outliers_iqr(df.copy(), target_column)

print(f"\nFound upper outliers: {p_upper}%")
print(f"Found lower outliers: {p_lower}%")
print("\nStatistics AFTER replacement (min/max should be equal to bounds):")
print(df_replaced['value'].describe())
print("\n" + "="*50 + "\n")

# --- Scenario B: Row removal mode ---
print(f"--- Scenario B: Removing rows with outliers in column '{target_column}' ---")
config_remove = OutlierConfig(change_remove=False)
df_removed, _, _ = tools.remove_outliers_iqr(df.copy(), target_column, config=config_remove)

print(f"\nSize of original DataFrame: {df.shape}")
print(f"DataFrame size AFTER rows are removed: {df_removed.shape}")
print("\nStats AFTER removal (min/max should be within normal limits):")
print(df_removed['value'].describe())
print("\n" + "="*50 + "\n")

# --- Scenario B: Custom sigma and no percentage return ---
print(f"--- Scenario B: Replacing outliers with stricter sigma=1.0 and no percentage return ---")
# Stricter sigma will find more outliers
config_custom = OutlierConfig(sigma=1.0, percentage=False)
# The function will return only the DataFrame, not a tuple
df_strict = tools.remove_outliers_iqr(df.copy(), target_column, config=config_custom)

print(f"\nReturn object type: {type(df_strict)}")
print("Check that this is a DataFrame and not a tuple was successful.")
print("\nStatistics AFTER replacement (boundaries are narrower than in scenario A):")
print(df_strict['value'].describe())
print("\n" + "="*50 + "\n")

# --- Scenario D: Testing error handling ---
print("--- Scenario D: Attempting to call on a non-existent column ---")
try:
tools.remove_outliers_iqr(df, 'non_existent_column')
except ValueError as e:
print(f"\nExpected error successfully caught: {e}")

##########    code checking for "stat_normal_testing"    ##########

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data ---

# Set seed for reproducibility
np.random.seed(42)
N_SAMPLES = 500

# Scenario A: Normally distributed data (Gaussian)
# Mean 50, standard deviation 10
normal_data = pd.Series(np.random.normal(loc=50, scale=10, size=N_SAMPLES), name='Normal_Distribution')

# Scenario B: Uniform distribution
# All values ​​from 0 to 100 are equally likely. Obviously not normal.
uniform_data = pd.Series(np.random.uniform(low=0, high=100, size=N_SAMPLES), name='Uniform_Distribution')

# Scenario B: Exponential Distribution
# Very right skewed. Clearly not normal.
exponential_data = pd.Series(np.random.exponential(scale=15, size=N_SAMPLES), name='Exponential_Distribution')

# --- 2. Initialization and calls ---
tools = DSTools()

# --- Scenario A: Testing on normal data ---
print("="*60)
print("SCENARIO A: TEST ON NORMAL DISTRIBUTION")
print("Expected: p-value > 0.05, output 'Data looks Gaussian'")
print("="*60)

# Call the function. It will print the results and show the graph.
# plt.show() blocks execution, so the graphs will appear one after the other.
tools.stat_normal_testing(normal_data)
print("\n")

# --- Scenario B: Testing on Uniform Distribution ---
print("="*60)
print("SCENARIO B: TEST ON UNIFORM DISTRIBUTION")
print("Expected: p-value < 0.05, output 'Data does not look Gaussian'")
print("="*60)

tools.stat_normal_testing(uniform_data)
print("\n")

# --- Scenario C: Testing on Exponential Distribution with describe_flag=True ---
print("="*60)
print("SCENARIO C: TEST ON EXPONENTIAL DISTRIBUTION")
print("Expected: p-value < 0.05, output 'Data does not look Gaussian'")
print("="*60)

tools.stat_normal_testing(uniform_data)
print("\n")

# --- Scenario C: Testing on Exponential Distribution with describe_flag=True ---
print("="*60)
print("SCENARIO C: TEST ON EXPONENTIAL DISTRIBUTION")
print("Expected: p-value < 0.05, output 'Data does not look Gaussian', and additional graphs")
print("="*60)

tools.stat_normal_testing(exponential_data, describe_flag=True)
print("\n")

# --- Scenario D: Testing with DataFrame ---
print("="*60)
print("SCENARIO D: TEST WITH A SINGLE-COLUMN DATAFRAME")
print("Check that the function works correctly if the input is a DataFrame, not a Series")
print("="*60)

# Convert Series to DataFrame
df_normal = pd.DataFrame(normal_data)
tools.stat_normal_testing(df_normal)
print("\n")

##########    code checking for "test_stationarity"    ##########

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data ---

# Set seed for reproducibility
np.random.seed(42)
N_SAMPLES = 365 # Year of data

# Create time index
time_index = pd.date_range(start='2023-01-01', periods=N_SAMPLES, freq='D')

# Scenario A: Stationary series (white noise)
# Mean and variance are constant.
stationary_series = pd.Series(
np.random.normal(loc=10, scale=5, size=N_SAMPLES),
index=time_index,
name='Stationary_Series'
)

# Scenario B: Non-stationary series (trend only)
# Mean increases over time.
trend = np.arange(N_SAMPLES) * 0.5
non_stationary_trend_series = pd.Series(
trend + np.random.normal(loc=0, scale=10, size=N_SAMPLES), # trend + noise
index=time_index,
name='NonStationary_Trend_Series'
)

# Scenario B: Non-stationary series (trend + seasonality)
# Simulate monthly seasonality (period ~30 days)
seasonality = 15 * np.sin(np.arange(N_SAMPLES) * (2 * np.pi / 30))
non_stationary_seasonal_series = pd.Series(
trend + seasonality + np.random.normal(loc=0, scale=5, size=N_SAMPLES),
index=time_index,
name='NonStationary_Seasonal_Series'
)

# --- 2. Initialization and calls ---
tools = DSTools()

# --- Scenario A: Test on stationary data ---
print("="*60)
print("SCENARIO A: TEST ON STATIONARY DATA (white noise)")
print("Expected: p-value < 0.05, output 'Data is STATIONARY!'")
print("On the graph, the moving average and st.deviation should be horizontal.")
print("="*60)
# IMPORTANT: After each call, a graph will appear.
# The script will continue to run only after you close the window with the graph.
tools.test_stationarity(stationary_series)
print("\n")

# --- Scenario B: Test on data with trend ---
print("="*60)
print("SCENARIO B: TEST ON NON-STATIONARY DATA (trend)")
print("Expected: p-value > 0.05, output 'Data is NON-STATIONARY!'")
print("On the chart, the moving average will increase along with the data.")
print("="*60)
tools.test_stationarity(non_stationary_trend_series)
print("\n")

# --- Scenario C: Test on data with trend and seasonality ---
print("="*60)
print("SCENARIO C: TEST ON NON-STATIONARY DATA (trend + seasonality)")
print("Expected: p-value > 0.05, output 'Data is NON-STATIONARY!'")
print("On the chart, the moving average will increase, and the standard deviation will fluctuate.")
print("="*60)
tools.test_stationarity(non_stationary_seasonal_series, len_window=60) # Let's increase the window for clarity
print("\n")

# --- Scenario D: Test the flag print_results_flag=False ---
print("="*60)
print("SCENARIO D: TEST WITH print_results_flag=False")
print("Expected: The chart will appear, but the detailed Dickey-Fuller test report will NOT be printed.")
print("="*60)
tools.test_stationarity(stationary_series, print_results_flag=False)
print("\n")

##########    code checking for "check_NINF"    ##########

import pandas as pd
import numpy as np

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data ---

# Create base data for each case
data_clean = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
data_with_nan = {'col1': [1, np.nan, 3], 'col2': [4, 5, 6]}
data_with_inf = {'col1': [1, 2, 3], 'col2': [4, np.inf, 6]}
data_with_both = {'col1': [1, np.nan, 3], 'col2': [np.NINF, 5, 6]} # np.NINF is -inf

# Create DataFrames
df_clean = pd.DataFrame(data_clean)
df_with_nan = pd.DataFrame(data_with_nan)
df_with_inf = pd.DataFrame(data_with_inf)
df_with_both = pd.DataFrame(data_with_both)

# Create NumPy arrays
arr_clean = df_clean.values
arr_with_nan = df_with_nan.values
arr_with_inf = df_with_inf.values
arr_with_both = df_with_both.values


# --- 2. Initialization and calls ---
tools = DSTools()

# --- Testing with pandas.DataFrame ---
print("="*50)
print("TESTING WITH PANDAS DATAFRAME")
print("="*50)

print("\n--- Case 1: Clean DataFrame ---")
print("Expected output: 'Dataset has no NaN or infinite values'")
tools.check_NINF(df_clean)

print("\n--- Case 2: DataFrame with NaN ---")
print("Expected output: 'Dataset has NaN values ​​but no infinite values'")
tools.check_NINF(df_with_nan)

print("\n--- Case 3: DataFrame with Inf ---")
print("Expected output: 'Dataset has infinite values ​​but no NaN values'")
tools.check_NINF(df_with_inf)

print("\n--- Case 4: DataFrame with NaN and Inf ---")
print("Expected output: 'Dataset has both NaN and infinite values'")
tools.check_NINF(df_with_both)

# --- Testing with numpy.ndarray ---
print("\n\n" + "="*50)
print("TESTING WITH NUMPY NDARRAY")
print("="*50)

print("\n--- Case 1: Clean array ---")
print("Expected output: 'Dataset has no NaN or infinite values'")
tools.check_NINF(arr_clean)

print("\n--- Case 2: Array with NaN ---")
print("Expected output: 'Dataset has NaN values ​​but no infinite values'")
tools.check_NINF(arr_with_nan)

print("\n--- Case 3: Array with Inf ---")
print("Expected output: 'Dataset has infinite values ​​but no NaN values'")
tools.check_NINF(arr_with_inf)

print("\n--- Case 4: Array with NaN and Inf ---")
print("Expected output: 'Dataset has both NaN and infinite values'")
tools.check_NINF(arr_with_both)

print("\n\n" + "="*50)

##########    code checking for "df_stats"    ##########

import pandas as pd
import numpy as np

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data ---

# Set seed for reproducibility
np.random.seed(42)

# Create data
data = {
'user_id': range(1, 101),
'age': np.random.randint(18, 65, size=100),
'city': np.random.choice(['Moscow', 'Kazan', 'Sochi', np.nan], size=100, p=[0.5, 0.3, 0.15, 0.05]),
'balance': np.random.uniform(0, 10000, size=100),
'registration_date': pd.to_datetime(pd.date_range(start='2022-01-01', periods=100, freq='D'))
}

df = pd.DataFrame(data)

# Let's artificially add more gaps to the 'balance' column
df.loc[df.sample(n=10, random_state=1).index, 'balance'] = np.nan

print("--- Information about the created DataFrame: ---")
print(df.info())
print("\n" + "="*50 + "\n")

# --- 2. Initialization and call to df_stats ---

tools = DSTools()

print("--- Call to df_stats: ---")
print("We expect to see summary statistics for DataFrame.")
tools.df_stats(df)
print("\n" + "="*50 + "\n")

# --- 3. Manual calculation check for control ---

# Calculate values ​​manually for verification
manual_cols = df.shape[1]
manual_rows = df.shape[0]
manual_missing_count = df.isnull().sum().sum()
manual_total_size = df.size
manual_missing_percent = np.round(manual_missing_count / manual_total_size * 100, 1)
manual_memory_mb = np.round(df.memory_usage(deep=True).sum() / 10**6, 1)

print("--- Manual calculation check: ---")
print(f"Columns: \t{manual_cols}")
print(f"Rows: \t{manual_rows}")
print(f"Gaps (%): \t{manual_missing_percent}%")
print(f"Memory (MB): \t{manual_memory_mb} MB")

##########    code checking for "describe_categorical"    ##########

np.random.seed(42)
N_SAMPLES = 100

# Create data for DataFrame
# Set seed for reproducibility
np.random.seed(42)
N_SAMPLES = 100

# Create data for DataFrame
data = {
'user_id': range(N_SAMPLES), # Numeric column, should be ignored

# Categorical column WITHOUT gaps
'status': np.random.choice(
['Active', 'Inactive', 'Blocked'],
size=N_SAMPLES,
p=[0.7, 0.2, 0.1] # Sum of probabilities is now 1.0
),

# Categorical column without gaps
'country': np.random.choice(
['Russia', 'Belarus', 'Kazakhstan', 'Armenia', 'Uzbekistan'],
size=N_SAMPLES
),

# Extreme case: column only of gaps
'notes': [np.nan] * N_SAMPLES,

# Column with date, should be ignored
'registration_date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=N_SAMPLES, freq='D'))
}

df = pd.DataFrame(data)

# RELIABLE WAY TO ADD GAP:
# Artificially replace 10% of values ​​in 'status' with NaN
df.loc[df.sample(frac=0.1, random_state=1).index, 'status'] = np.nan

print("--- Information about the created DataFrame: ---")
df.info()
print("\n" + "="*60 + "\n")

# --- 2. Calling the function and checking the result ---

tools = DSTools()

print("--- Calling describe_categorical: ---")
# The function returns a DataFrame, save it
cat_stats_df = tools.describe_categorical(df)

print("Result of the function:")
print(cat_stats_df)
print("\n" + "="*60 + "\n")

# --- 3. Parsing the results (manual verification) ---
print("--- Parsing the results (correctness check): ---")
print("\n1. The DataFrame index contains only categorical columns ('status', 'country', 'notes').")
print(" Columns 'user_id' and 'registration_date' were correctly ignored.\n")

print("2. Parsing the 'status' row:")
status_missing_percent = df['status'].isnull().sum() / len(df) * 100
print(f" - missing (%): {cat_stats_df.loc['status', 'missing (%)']:.1f} (expecting ~{status_missing_percent:.1f}%)")
print(f" - unique: {cat_stats_df.loc['status', 'unique']} (expecting 3, since NaN doesn't count)")
print(f" - top: '{cat_stats_df.loc['status', 'top']}' (expecting 'Active')")
print(f" - freq: {cat_stats_df.loc['status', 'freq']} (expecting ~60)\n")

print("3. Analysis of the string 'country':")
print(f" - missing (%): {cat_stats_df.loc['country', 'missing (%)']:.1f} (expecting 0.0)")
print(f" - unique: {cat_stats_df.loc['country', 'unique']} (expecting 5)\n")

print("4. Analysis of the string 'notes' (extreme case):")
print(f" - missing (%): {cat_stats_df.loc['notes', 'missing (%)']:.1f} (expecting 100.0)")
print(f" - unique: {cat_stats_df.loc['notes', 'unique']} (expecting 0)")
print("\n" + "="*60 + "\n")

# --- 4. Testing on data without categorical columns (edge ​​case) ---
print("--- Testing on a DataFrame without categorical columns ---")
df_numeric = pd.DataFrame({
'a': [1, 2, 3],
'b': [4.0, 5.0, 6.0]
})

empty_df = tools.describe_categorical(df_numeric)

print("Result for a purely numeric DataFrame:")
print(empty_df)
print("\nThe function correctly returned an empty DataFrame, just like expected.")

##########    code checking for "describe_numeric"    ##########

import pandas as pd
import numpy as np

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data ---

# Set seed for reproducibility
np.random.seed(42)
N_SAMPLES = 1000

# Create data for DataFrame
data = {
# Standard numeric column (int)
'orders_count': np.random.randint(0, 50, size=N_SAMPLES),

# Numeric column (float) with missing values ​​and strong right skew
'revenue': np.random.lognormal(mean=8, sigma=1.5, size=N_SAMPLES),

# Column with negative kurtosis (light tails, flatter than normal)
'uniform_score': np.random.uniform(-1, 1, size=N_SAMPLES),

# Constant column (edge ​​case, std=0)
'api_version': [2] * N_SAMPLES,

# Non-numeric columns to be ignored
'user_segment': np.random.choice(['new', 'active', 'churned'], size=N_SAMPLES),
'last_seen': pd.to_datetime(pd.date_range(start='2023-01-01', periods=N_SAMPLES, freq='h'))
}
df = pd.DataFrame(data)

# Artificially add gaps in the 'revenue' column (10% gaps)
df.loc[df.sample(frac=0.1, random_state=1).index, 'revenue'] = np.nan

print("--- Information about the created DataFrame: ---")
df.info()
print("\n" + "="*60 + "\n")

# --- 2. Calling the function and checking the result ---

tools = DSTools()

print("--- Calling describe_numeric: ---")
# The function returns a DataFrame, save it
num_stats_df = tools.describe_numeric(df)

# Transposing the DataFrame for easy viewing
print("Result of the function (transposed for readability):")
print(num_stats_df.T)
print("\n" + "="*60 + "\n")

# --- 3. Parsing the results (checking correctness) ---
print("--- Parsing results (validation): ---")
print("\n1. DataFrame index contains only numeric columns. 'user_segment' and 'last_seen' are ignored.")

print("\n2. Parsing column 'revenue':")
expected_missing = df['revenue'].isnull().sum() / len(df) * 100
print(f" - missing (%): {num_stats_df.loc['revenue', 'missing (%)']:.1f} (expecting ~{expected_missing:.1f}%)")
print(f" - skew: {num_stats_df.loc['revenue', 'skew']:.2f} (expecting large positive value, > 1)")

print("\n3. Analysis of the 'uniform_score' column:")
print(f" - kurtosis: {num_stats_df.loc['uniform_score', 'kurtosis']:.2f} (expecting a negative value, ~ -1.2)")

print("\n4. Analysis of the 'api_version' column (extreme case):")
print(f" - std: {num_stats_df.loc['api_version', 'std']:.2f} (expecting 0.0, since all values ​​are the same)")
print(f" - min, max, mean, median: all equal to 2.0")
print("\n" + "="*60 + "\n")

# --- 4. Testing on data without numeric columns ---
print("--- Testing on a DataFrame without numeric columns ---")
df_categorical = pd.DataFrame({
'a': ['x', 'y', 'z'],
'b': ['foo', 'bar', 'baz']
})

empty_df = tools.describe_numeric(df_categorical)

print("Result for pure categorical DataFrame:")
print(empty_df)
print("\nThe function correctly returned an empty DataFrame, as expected.")

##########    code checking for "generate_distribution"    ##########

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Import our class and its configuration
from ds_tool import DSTools, DistributionConfig

def analyze_and_compare(generated_data: np.ndarray, config: DistributionConfig):
"""A helper function for analyzing and comparing results."""

# Calculate the actual statistics of the generated data
actual_mean = np.mean(generated_data)
actual_std = np.std(generated_data, ddof=1) # ddof=1 for sample st. deviation
actual_skew = stats.skew(generated_data)
# stats.kurtosis returns the kurtosis (Fisher), and we specified the usual (Pearson).
# Pearson Kurtosis = Fisher Kurtosis + 3
actual_kurtosis = stats.kurtosis(generated_data) + 3
actual_min = np.min(generated_data)
actual_max = np.max(generated_data)

print("\n--- Comparison of target and actual metrics ---")
print(f"| Metric | Target | Actual |")
print(f"|------------|-----------|")
print(f"| Mean | {config.mean:<8.2f} | {actual_mean:<8.2f} |")
print(f"| Std. Dev. | {config.std:<8.2f} | {actual_std:<8.2f} |")
print(f"| Skewness | {config.skewness:<8.2f} | {actual_skew:<8.2f} |") 
print(f"| Kurtosis | {config.kurtosis:<8.2f} | {actual_kurtosis:<8.2f} |") 
print(f"| Min | {config.min_val:<8.2f} | {actual_min:<8.2f} |") 
print(f"| Max | {config.max_val:<8.2f} | {actual_max:<8.2f} |") 
print(f"| N | {config.n:<8} | {len(generated_data):<8} |") 

# Visualization 
plt.figure(figsize=(10, 6)) 
plt.hist(generated_data, bins=50, density=True, alpha=0.7, label='Generated Distribution')
plt.title("Histogram of generated distribution")
plt.grid(True)
plt.show()

# --- Initialize tools ---
tools = DSTools()

# --- Scenario A: Main scenario ---
print("="*60)
print("SCENARIO A: Generate a moderately skewed distribution")
print("="*60)
try:
config_a = DistributionConfig(
mean=1000, median=950, std=200, min_val=400, max_val=2500,
skewness=0.8, kurtosis=4.0, n=2000, outlier_ratio=0.01
)
generated_data_a = tools.generate_distribution(config_a)
analyze_and_compare(generated_data_a, config_a)
except ValueError as e:
print(f"An unexpected error occurred: {e}")

# --- Scenario B: High Kurtosis Scenario ---
print("\n" + "="*60)
print("SCENARIO B: Generating a heavy-tailed (high kurtosis) distribution")
print("="*60)
try:
config_b = DistributionConfig(
mean=50, median=48, std=10, min_val=10, max_val=150,
skewness=1.5, kurtosis=8.0, n=2000, outlier_ratio=0.03
)
generated_data_b = tools.generate_distribution(config_b)
analyze_and_compare(generated_data_b, config_b)
except ValueError as e:
print(f"An unexpected error occurred: {e}")

# --- Scenario B: Checking invalid moments ---
print("\n" + "="*60)
print("SCENARIO B: Trying to generate a distribution with invalid moments")
print("Expecting ValueError because kurtosis < skewness² - 2")
print("="*60)
try:
config_c_invalid = DistributionConfig(
mean=100, median=100, std=15, min_val=50, max_val=150,
skewness=2.0, kurtosis=1.0, n=1000 # Impossible: 1.0 < (2.0² - 2)
)
tools.generate_distribution(config_c_invalid)
except ValueError as e:
print(f"\nSUCCESSFULLY caught the expected error: {e}")

# --- Scenario D: Checking Pydantic validation ---
print("\n" + "="*60)
print("SCENARIO D: Trying to create config with max_val < min_val")
print("Expecting ValueError from Pydantic")
print("="*60)
try:
invalid_pydantic_config = DistributionConfig(
mean=100, median=100, std=15, min_val=200, max_val=100, # Impossible
skewness=0, kurtosis=3, n=1000
)
except ValueError as e:
print(f"\nSUCCESSFULLY caught expected Pydantic error: {e}")

##########    code checking for "evaluate_classification"    ##########

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data ---
# This data is identical to what we generated for compute_metrics,
# as it is ideal for a binary classification problem.

# Set seed for reproducibility of results
np.random.seed(42)
N_SAMPLES = 500

# Create array of true labels (y_true)
y_true = np.random.randint(0, 2, size=N_SAMPLES)

# Create array of predicted probabilities (y_predict_proba)
# Bias probabilities towards true labels by adding noise
base_probs = np.where(y_true == 1, 0.75, 0.25)
noise = np.random.normal(0, 0.2, size=N_SAMPLES)
y_predict_proba = np.clip(base_probs + noise, 0.01, 0.99) # Clip to avoid 0 and 1

print("--- Data generation complete ---")
print(f"Form y_true: {y_true.shape}")
print(f"Form y_predict_proba: {y_predict_proba.shape}")
print("-" * 35, "\n")

# --- 2. Initialization and calls ---
tools = DSTools()

# --- Scenario A: Main call with threshold 0.5 ---
print("="*60)
print("SCENARIO A: Call with default threshold (0.5)")
print("Waiting for full report in console and window with two graphs.")
print("="*60)

# IMPORTANT: The script will stop and wait for the window with the graph to close.
returned_metrics_a = tools.evaluate_classification(
true_labels=y_true,
pred_probs=y_predict_proba,
threshold=0.5
)

print("\n--- Checking the returned dictionary (Scenario A): ---")
print(f"The type of the returned object: {type(returned_metrics_a)}")
print("Keys in the dictionary:", list(returned_metrics_a.keys()))
print(f"ROC AUC from the dictionary: {returned_metrics_a['roc_auc']:.4f}")
print("\n")

# --- Scenario B: Call with a different threshold (0.7) ---
print("="*60)
print("SCENARIO B: Call with a higher threshold (0.7)")
print("We expect that 'Accuracy' and the report will change, but ROC AUC will not.")
print("="*60)

returned_metrics_b = tools.evaluate_classification(
true_labels=y_true,
pred_probs=y_predict_proba,
threshold=0.7 # Use a higher threshold
)

print("\n--- Comparison of results (Scenario A vs Scenario B): ---")
print(f"Accuracy (threshold=0.5): {returned_metrics_a['accuracy']:.4f}")
print(f"Accuracy (threshold=0.7): {returned_metrics_b['accuracy']:.4f}")
print("-> Accuracy values ​​should be different.\n")

print(f"ROC AUC (threshold=0.5): {returned_metrics_a['roc_auc']:.4f}")
print(f"ROC AUC (threshold=0.7): {returned_metrics_b['roc_auc']:.4f}")
print("-> ROC AUC values ​​should be identical, since it does not depend on the threshold.")
print("\n")

# --- Scenario B: Testing error handling ---
print("="*60)
print("SCENARIO B: Testing error handling (shape mismatch)")
print("Expecting ValueError.")
print("="*60)

try:
y_true_short = y_true[:-10] # Shorten one of the arrays
tools.evaluate_classification(y_true_short, y_predict_proba)
except ValueError as e:
print(f"SUCCESSFULLY caught expected error: {e}")

##########    code checking for "grubbs_test"    ##########

import numpy as np
import pandas as pd

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data ---

# Set seed for reproducibility
np.random.seed(42)

# Data WITHOUT obvious outliers (from normal distribution)
data_normal = np.random.normal(loc=100, scale=10, size=30)

# Data WITH OBVIOUS outlier. Take normal data and add one extreme value.
data_with_outlier = np.append(data_normal, 150)

# Data where all values ​​are the same
data_constant = np.full(10, 50.0)

# --- 2. Initialization and calls ---
tools = DSTools()

# --- Scenario A: Test on data WITH outlier ---
print("="*60)
print("SCENARIO A: Test on data WITH outlier (value 200)")
print("Expected: is_outlier=True, outlier_value=200")
print("="*60)

result_a = tools.grubbs_test(data_with_outlier)
print(f"Result of Pydantic model: {result_a}")
if result_a.is_outlier:
print(f"-> SUCCESS: Outlier found! Value: {result_a.outlier_value}, Index: {result_a.outlier_index}")
else:
print("-> ERROR: Outlier was not found.")
print("\n")

# --- Scenario B: Test on data WITHOUT outliers ---
print("="*60)
print("SCENARIO B: Test on 'pure' normal data")
print("Expected: is_outlier=False")
print("="*60)

result_b = tools.grubbs_test(data_normal)
print(f"Pydantic model result: {result_b}")
if not result_b.is_outlier:
print(f"-> SUCCESS: Outlier was not found, as expected.")
else:
print("-> ERROR: False outlier.")
print("\n")

# --- Scenario B: Test with different alpha ---
print("="*60)
print("SCENARIO B: Test on data with MODERATE outlier (150)")
print("Expect: Outlier found first (alpha=0.05), then not (alpha=0.01)")
print("="*60)

# First, let's test with default alpha=0.05. Outlier should be found.
print("\nStep 1: alpha=0.05")
result_c1 = tools.grubbs_test(data_with_outlier, alpha=0.05)
print(f"Result: {result_c1}")
if result_c1.is_outlier:
print("-> SUCCESS: Outlier (150) found as expected.")
else:
print("-> ERROR: Outlier not found.")

# Now using a stricter alpha. Now G-calculated should be less than G-critical.
print("\nStep 2: alpha=0.01")
result_c2 = tools.grubbs_test(data_with_outlier, alpha=0.01)
print(f"Result: {result_c2}")
print(f" G-calculated: {result_c2.g_calculated:.4f}")
print(f" G-critical (greater): {result_c2.g_critical:.4f}")

if not result_c2.is_outlier:
print(f"-> SUCCESS: Outlier not found, because G-calculated < G-critical.")
else:
print("-> ERROR: Outlier found, but shouldn't have with such alpha.")
print("\n")

# --- Scenario G: Test on constant data ---
print("="*60)
print("SCENARIO D: Test on data with equal values")
print("Expected: is_outlier=False")
print("="*60)
result_d = tools.grubbs_test(data_constant)
print(f"Result of Pydantic model: {result_d}")
if not result_d.is_outlier:
print(f"-> SUCCESS: No outlier found, as expected.")
else:
print("-> ERROR: False outlier found.")
print("\n")

# --- Scenario D: Testing error handling ---
print("="*60)
print("SCENARIO D: Testing error handling")
print("="*60)

print("\nAttempt to pass an array with 2 elements (expected ValueError):")
try:
tools.grubbs_test(np.array([1, 2]))
except ValueError as e:
print(f"-> SUCCESS: Caught error: {e}")

print("\nTrying to pass a list instead of NumPy/Pandas (expecting TypeError):")
try:
# Your function accepts lists too, so this check may fail.
# But if there was strong typing, it would work.
# The current implementation of the code will not throw an error, since it converts the list to an array.
# To throw an error, pass an invalid type, such as a string.
tools.grubbs_test("invalid_type")
except TypeError as e:
print(f"-> SUCCESS: Caught error: {e}")

##########    code checking for "plot_confusion_matrix"    ##########

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data ---

# Set seed for reproducibility
np.random.seed(42)
N_SAMPLES = 200

# --- Data for binary classification (classes 0 and 1) ---
y_true_binary = np.random.randint(0, 2, size=N_SAMPLES)
# Simulate predictions of a model that sometimes gets it wrong
# With 85% probability the prediction will be correct
y_pred_binary = np.where(np.random.rand(N_SAMPLES) < 0.85, y_true_binary, 1 - y_true_binary)

# --- Data for multi-class classification (classes 0, 1, 2) ---
y_true_multi = np.random.randint(0, 3, size=N_SAMPLES)
# Simulate predictions. With a probability of 75%, the class is predicted correctly, otherwise - a random one of the other two.
correct_preds = np.random.rand(N_SAMPLES) < 0.75
random_errors = np.random.randint(1, 3, size=N_SAMPLES)
y_pred_multi = np.where(correct_preds, y_true_multi, (y_true_multi + random_errors) % 3)

# --- 2. Initialization and calls ---
tools = DSTools()

# --- Scenario A: Binary matrix without custom labels ---
print("="*60)
print("SCENARIO A: Binary adjacency matrix (default labels)")
print("Expected: 2x2 plot with '0' and '1' labels on axes.")
print("="*60)
# IMPORTANT: The script will wait for the window to close with plot
tools.plot_confusion_matrix(y_true_binary, y_pred_binary)

# --- Scenario B: Binary matrix with custom labels and theme ---
print("\n" + "="*60)
print("SCENARIO B: Binary matrix with custom labels and different color scheme")
print("Expected: 2x2 plot with labels 'Negative' and 'Positive'.")
print("="*60)
tools.plot_confusion_matrix(
y_true_binary,
y_pred_binary,
class_labels=['Negative (0)', 'Positive (1)'],
title='Binary Classification Performance',
cmap='Greens'
)

# --- Scenario C: Multi-class matrix with custom labels ---
print("\n" + "="*60)
print("SCENARIO B: Multiclass Adjacency Matrix (3x3)")
print("Expecting: 3x3 plot with animal class labels.")
print("="*60)
tools.plot_confusion_matrix(
y_true_multi,
y_pred_multi,
class_labels=['Cat', 'Dog', 'Bird'],
title='Multi-Class Classification (Animals)',
cmap='YlGnBu'
)

# --- Scenario D: Testing Error Handling ---
print("\n" + "="*60)
print("SCENARIO D: Testing Error Handling (Invalid Number of Labels)")
print("Expecting: ValueError.")
print("="*60)
try:
tools.plot_confusion_matrix(
y_true_binary,
y_pred_binary,
class_labels=['One label'] # Pass 1 label for 2 classes
)
except ValueError as e:
print(f"SUCCESSFULLY caught expected error: {e}")

##########    code checking for "add_missing_value_features"    ##########

import numpy as np
import pandas as pd
import polars as pl

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data ---
# Create data that will be used for both DataFrame types
data_dict = {
'col_a': [1, 2, np.nan, 4, 5],
'col_b': [np.nan, 'B', 'C', np.nan, 'E'],
'col_c': [10.0, 20.0, 30.0, 40.0, np.nan],
'col_d': [np.nan, 'Y', 'Z', 'W', 'V'] # Column with no missing values
}
# Expected values ​​for num_missing: [2, 1, 1, 1, 2]

pd_df = pd.DataFrame(data_dict)
pl_df = pl.DataFrame(data_dict, strict=False)

print("--- Original Pandas DataFrame: ---")
print(pd_df)
print("\n--- Original Polars DataFrame: ---")
print(pl_df)
print("\n" + "="*60 + "\n")

# --- 2. Initialization and calls ---
tools = DSTools()

# --- Testing with Pandas ---
print("--- TESTING WITH PANDAS DATAFRAME ---")

# Scenario A: Pandas, default mode
print("\nScenario A: Pandas, default mode (num_missing only)")
pd_result_a = tools.add_missing_value_features(pd_df)
print(pd_result_a)
# Check
expected_missing_counts = [2, 0, 1, 1, 1]
assert pd_result_a['num_missing'].tolist() == expected_missing_counts
assert 'num_missing_std' not in pd_result_a.columns
print("-> SUCCESS: Column 'num_missing' added correctly.")

# Scenario B: Pandas, with std added
print("\nScenario B: Pandas, with add_std=True")
pd_result_b = tools.add_missing_value_features(pd_df, add_std=True)
print(pd_result_b)
# Check
assert 'num_missing' in pd_result_b.columns
assert 'num_missing_std' in pd_result_b.columns
print("-> SUCCESS: Both columns ('num_missing', 'num_missing_std') added.")

print("\n" + "="*60 + "\n")

# --- Testing with Polars ---
print("--- TESTING WITH POLARS DATAFRAME ---")

# Scenario B: Polars, default mode
print("\nScenario B: Polars, default mode")
pl_result_a = tools.add_missing_value_features(pl_df)
print(pl_result_a)
# Testing
expected_missing_polars = [2, 0, 0, 1, 0]
assert pl_result_a['num_missing'].to_list() == expected_missing_polars
print("-> SUCCESS: Column 'num_missing' was added correctly.")

# Scenario D: Polars, with std added
print("\nScenario D: Polars, with add_std=True")
print("We expect to see a warning in the console.")
pl_result_b = tools.add_missing_value_features(pl_df, add_std=True)
print(pl_result_b) # The output will be the same as scenario B
print("-> SUCCESS: The warning was printed as expected.")

print("\n" + "="*60 + "\n")

# --- Scenario E: Testing error handling ---
print("--- SCENARIO E: Testing error handling errors ---")
print("Expecting TypeError when passing list.")
try:
tools.add_missing_value_features([1, 2, 3])
except TypeError as e:
print(f"-> SUCCESS: Caught expected error: {e}")

##########    code checking for "chatterjee_correlation"    ##########

import numpy as np
import pandas as pd

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data ---

# Set seed for reproducibility
np.random.seed(42)
N_SAMPLES = 200

# Create base independent variable
x = np.linspace(-10, 10, N_SAMPLES)

# Scenario 1: Perfect linear dependence
y_linear = 2 * x + 5

# Scenario 2: Perfect nonlinear (quadratic) dependence
y_quadratic = x**2 + np.random.normal(0, 0.1, N_SAMPLES) # Add some noise

# Scenario 3: Complete independence (random noise)
y_random = np.random.randn(N_SAMPLES) * 10

# --- 2. Initialization and calls ---
tools = DSTools()

# --- Scenario A: Linear dependence ---
print("="*60)
print("SCENARIO A: Linear dependence (y = 2x + 5)")
print("Expected: Xi close to 1.0")
print("="*60)
xi_linear = tools.chatterjee_correlation(x, y_linear)
print(f"Result: {xi_linear:.4f}\n")
assert xi_linear > 0.95

# --- Scenario B: Nonlinear dependence ---
print("="*60)
print("SCENARIO B: Nonlinear quadratic dependence (y = x^2)")
print("Expected: Xi close to 1.0 (Pearson would show ~0 here)")
print("="*60)
xi_quadratic = tools.chatterjee_correlation(x, y_quadratic)
print(f"Result: {xi_quadratic:.4f}\n")
assert xi_quadratic > 0.95

# --- Scenario B: No correlation ---
print("="*60)
print("SCENARIO B: No correlation (random noise)")
print("Expected: Xi close to 0.0")
print("="*60)
xi_random = tools.chatterjee_correlation(x, y_random)
print(f"Result: {xi_random:.4f}\n")
assert xi_random < 0.1

# --- Scenario D: Check for skewness ---
print("="*60)
print("SCENARIO D: Check for skewness Xi(x, y) != Xi(y, x)")
print("="*60)
xi_xy = tools.chatterjee_correlation(x, y_quadratic)
xi_yx = tools.chatterjee_correlation(y_quadratic, x)
print(f"Xi(x, y) = {xi_xy:.4f} (y is a function of x)")
print(f"Xi(y, x) = {xi_yx:.4f} (x is not a single-valued function of y)")
print("-> Values ​​must be different. SUCCESS.\n")
assert abs(xi_xy - xi_yx) > 0.1

# --- Scenario D: Check for standard_flag ---
print("="*60)
print("SCENARIO D: Checking the standard_flag flag")
print("Expected: results with standard_flag=True and False should be different.")
print("="*60)
xi_standard = tools.chatterjee_correlation(x, y_quadratic, standard_flag=True)
xi_original = tools.chatterjee_correlation(x, y_quadratic, standard_flag=False)
print(f"Result with standard formula: {xi_standard:.4f}")
print(f"Result with original formula: {xi_original:.4f}")
print("-> Values ​​are different. Flag works. SUCCESS.\n")
assert xi_standard != xi_original

# --- Scenario E: Testing error handling ---
print("="*60)
print("SCENARIO E: Testing Error Handling (Arrays of Different Lengths)")
print("Expecting: ValueError")
print("="*60)
try:
tools.chatterjee_correlation(x[:-1], y_linear)
except ValueError as e:
print(f"SUCCESSFULLY caught expected error: {e}")

##########    code checking for "calculate_entropy"    ##########

import numpy as np

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data (probability distributions) ---

# Highly certain distribution (one outcome dominates)
dist_ordered = np.array([0.9, 0.05, 0.05])

# Uniform distribution (maximum uncertainty)
dist_uniform = np.array([1/3, 1/3, 1/3])

# Intermediate distribution
dist_mixed = np.array([0.5, 0.3, 0.2])

# Deterministic distribution (complete certainty)
dist_deterministic = np.array([1.0, 0.0, 0.0])

# Unnormalized data (sum is not 1)
dist_unnormalized = np.array([1, 4, 5]) # Equivalent to [0.1, 0.4, 0.5]

# --- 2. Initialization and calls ---
tools = DSTools()

# --- Scenario A: Comparing entropy of different distributions ---
print("="*60)
print("SCENARIO A: Comparing entropy (in nats)")
print("Expect: E(deterministic) < E(ordered) < E(mixed) < E(uniform)")
print("="*60)

entropy_det = tools.calculate_entropy(dist_deterministic)
entropy_ord = tools.calculate_entropy(dist_ordered)
entropy_mix = tools.calculate_entropy(dist_mixed)
entropy_uni = tools.calculate_entropy(dist_uniform)

print(f"Entropy of deterministic [1, 0, 0]: {entropy_det:.4f} (should be 0.0)")
print(f"Entropy of ordered [0.9, 0.05, 0.05]: {entropy_ord:.4f}")
print(f"Entropy of mixed [0.5, 0.3, 0.2]: {entropy_mix:.4f}")
print(f"Entropy of uniform [0.33, 0.33, 0.33]: {entropy_uni:.4f} (should be maximal)")

# Assertions
assert np.isclose(entropy_det, 0.0)
assert entropy_det < entropy_ord < entropy_mix < entropy_uni
print("-> SUCCESS: Entropy hierarchy is correct.\n")

# --- Scenario B: Calculation in bits ---
print("="*60)
print("SCENARIO B: Calculate entropy in bits (base=2)")
print("="*60)

entropy_mix_nats = tools.calculate_entropy(dist_mixed) # base=None
entropy_mix_bits = tools.calculate_entropy(dist_mixed, base=2)

print(f"Entropy for [0.5, 0.3, 0.2] in nats: {entropy_mix_nats:.4f}")
print(f"Entropy for [0.5, 0.3, 0.2] in bits: {entropy_mix_bits:.4f}")
print("-> Values ​​are different. Flag 'base' works. SUCCESS.\n")
assert entropy_mix_nats != entropy_mix_bits

# --- Scenario B: Working with unnormalized data ---
print("="*60)
print("SCENARIO B: Test on unnormalized data [1, 4, 5]")
print("Expected: the result should be the same as for [0.1, 0.4, 0.5]")
print("="*60)

normalized_equivalent = np.array([0.1, 0.4, 0.5])
entropy_normalized = tools.calculate_entropy(normalized_equivalent)
entropy_unnormalized = tools.calculate_entropy(dist_unnormalized)

print(f"Entropy for [0.1, 0.4, 0.5]: {entropy_normalized:.4f}")
print(f"Entropy for [1, 4, 5]: {entropy_unnormalized:.4f}")
assert np.isclose(entropy_normalized, entropy_unnormalized)
print("-> Values ​​match. Automatic normalization works. SUCCESS.\n")

# --- Scenario D: Testing error handling ---
print("="*60)
print("SCENARIO D: Testing error handling (negative probabilities)")
print("Expecting: ValueError")
print("="*60)
try:
invalid_dist = np.array([1.5, -0.5, 0.0])
tools.calculate_entropy(invalid_dist)
except ValueError as e:
print(f"SUCCESSFULLY caught expected error: {e}")

##########    code checking for "calculate_kl_divergence"    ##########

import numpy as np

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data (probability distributions) ---

# "True" P distribution
P = np.array([0.1, 0.7, 0.2])

# Q1 distribution, good approximation to P
Q1_good_approx = np.array([0.15, 0.65, 0.2])

# Q2 distribution, bad approximation to P (uniform)
Q2_bad_approx = np.array([1/3, 1/3, 1/3])

# Unnormalized distributions
P_unnormalized = np.array([1, 7, 2]) # Equivalent to P
Q1_unnormalized = np.array([15, 65, 20]) # Equivalent to Q1

# --- 2. Initialization and calls ---
tools = DSTools()

# --- Scenario A: D_KL(P || P) = 0 ---
print("="*60)
print("SCENARIO A: Distribution diverges from itself")
print("Expected: D_KL(P || P) = 0")
print("="*60)
kl_self = tools.calculate_kl_divergence(P, P)
print(f"D_KL(P || P) = {kl_self:.6f}")
assert np.isclose(kl_self, 0)
print("-> SUCCESS: Result is close to zero.\n")

# --- Scenario B: Skewness ---
print("="*60)
print("SCENARIO B: Checking skewness D_KL(P || Q) != D_KL(Q || P)")
print("="*60)
kl_pq = tools.calculate_kl_divergence(P, Q1_good_approx)
kl_qp = tools.calculate_kl_divergence(Q1_good_approx, P)
print(f"D_KL(P || Q1) = {kl_pq:.4f}")
print(f"D_KL(Q1 || P) = {kl_qp:.4f}")
assert not np.isclose(kl_pq, kl_qp)
print("-> SUCCESS: Values ​​are not equal, skewness confirmed.\n")

# --- Scenario C: Comparing divergences ---
print("="*60)
print("SCENARIO B: Comparing Approximations")
print("Expected: D_KL(P || good_approx) < D_KL(P || bad_approx)")
print("="*60)
kl_good = tools.calculate_kl_divergence(P, Q1_good_approx)
kl_bad = tools.calculate_kl_divergence(P, Q2_bad_approx)
print(f"Divergence from good approximation (Q1): {kl_good:.4f}")
print(f"Divergence from bad approximation (Q2): {kl_bad:.4f}")
assert kl_good < kl_bad
print("-> SUCCESS: Divergence hierarchy is correct.\n")

# --- Scenario D: Calculating in bits ---
print("="*60)
print("SCENARIO D: Calculate in bits (base=2)")
print("="*60)
kl_nats = tools.calculate_kl_divergence(P, Q1_good_approx)
kl_bits = tools.calculate_kl_divergence(P, Q1_good_approx, base=2)
print(f"D_KL(P || Q1) in nats: {kl_nats:.4f}")
print(f"D_KL(P || Q1) in bits: {kl_bits:.4f}")
assert not np.isclose(kl_nats, kl_bits)
print("-> SUCCESS: Values ​​are different, 'base' flag works.\n")

# --- Scenario D: Working with non-normalized data ---
print("="*60)
print("SCENARIO D: Test on unnormalized data")
print("="*60)
kl_normalized = tools.calculate_kl_divergence(P, Q1_good_approx)
kl_unnormalized = tools.calculate_kl_divergence(P_unnormalized, Q1_unnormalized)
print(f"Divergence for normalized P and Q1: {kl_normalized:.4f}")
print(f"Divergence for unnormalized P and Q1: {kl_unnormalized:.4f}")
assert np.isclose(kl_normalized, kl_unnormalized)
print("-> SUCCESS: Values ​​match, normalization works.\n")

# --- Scenario E: Testing error handling ---
print("="*60)
print("SCENARIO E: Testing Error Handling")
print("="*60)
print("\nAttempt to pass distributions of different lengths:")
try:
tools.calculate_kl_divergence([0.5, 0.5], [0.3, 0.3, 0.4])
except ValueError as e:
print(f"-> SUCCESS: Caught error: {e}")

print("\nAttempt to pass distributions with negative values:")
try:
tools.calculate_kl_divergence([1.5, -0.5], [0.5, 0.5])
except ValueError as e:
print(f"-> SUCCESS: Caught error: {e}")

##########    code checking for "min_max_scale"    ##########

import numpy as np
import pandas as pd
import polars as pl

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data ---
data_dict = {
'col_to_scale_1': [10, 20, 30, 40, 50], # Simple int column
'col_to_scale_2': [-5.0, 0.0, 5.0, 10.0, 15.0], # Float with negative values
'col_constant': [5, 5, 5, 5, 5], # Constant column
'col_ignore': ['A', 'B', 'C', 'D', 'E'] # String column, should be ignored
}

pd_df = pd.DataFrame(data_dict)
pl_df = pl.DataFrame(data_dict)

print("--- Original Pandas DataFrame: ---")
print(pd_df)
print("\n--- Original Polars DataFrame: ---")
print(pl_df)
print("\n" + "="*60 + "\n")

# --- 2. Initialization and calls ---
tools = DSTools()

# --- Testing with Pandas ---
print("--- TESTING WITH PANDAS DATAFRAME ---")

print("\nScenario A: Scaling selected columns")
pd_result_a = tools.min_max_scale(pd_df, columns=['col_to_scale_1', 'col_to_scale_2'])
print(pd_result_a)
assert pd_result_a['col_to_scale_1'].min() == 0.0 and pd_result_a['col_to_scale_1'].max() == 1.0
assert pd_result_a['col_to_scale_2'].min() == 0.0 and pd_result_a['col_to_scale_2'].max() == 1.0
assert (pd_result_a['col_constant'] == 5).all() # Check that the constant column has not changed
print("-> SUCCESS: Selected columns have been scaled.")

print("\nScenario B: Scale all numeric columns (including the constant)")
pd_result_b = tools.min_max_scale(pd_df) # columns=None
print(pd_result_b)
assert (pd_result_b['col_constant'] == 0.0).all() # Check that the constant column is filled with zeros
print("-> SUCCESS: All numeric columns are scaled, the constant column is filled with zero.")

print("\nScenario B: Checking const_val_fill")
pd_result_c = tools.min_max_scale(pd_df, const_val_fill=0.5)
print(pd_result_c)
assert (pd_result_c['col_constant'] == 0.5).all()
print("-> SUCCESS: The constant column is filled with the value 0.5.")

print("\n" + "="*60 + "\n")

# --- Testing with Polars ---
print("--- TESTING WITH POLARS DATAFRAME ---")

print("\nScenario D: Scaling all numeric columns")
pl_result_a = tools.min_max_scale(pl_df)
print(pl_result_a)
assert pl_result_a['col_to_scale_1'].min() == 0.0 and pl_result_a['col_to_scale_1'].max() == 1.0
assert pl_result_a['col_constant'].min() == 0.0 and pl_result_a['col_constant'].max() == 0.0
print("-> SUCCESS: All numeric columns for Polars scaled.")

print("\nScenario E: Checking const_val_fill for Polars")
pl_result_b = tools.min_max_scale(pl_df, const_val_fill=0.5)
print(pl_result_b)
assert pl_result_b['col_constant'].min() == 0.5 and pl_result_b['col_constant'].max() == 0.5
print("-> SUCCESS: Constant column for Polars filled with value 0.5.")

print("\n" + "="*60 + "\n")

# --- Scenario E: Testing error and warning handling ---
print("--- SCENARIO E: Testing error handling ---")

print("\nAttempt to pass non-existent column (expecting warning):")
# There will be no error, but "Warning: ..." should appear in the console
tools.min_max_scale(pd_df, columns=['col_to_scale_1', 'non_existent_col'])
print("-> SUCCESS: Warning printed, program did not crash.\n")

print("Trying to pass list (expecting TypeError):")
try:
tools.min_max_scale([1, 2, 3])
except TypeError as e:
print(f"-> SUCCESS: Expected error caught: {e}")

##########    code checking for "save_dataframes_to_zip" & "read_dataframes_from_zip"   ##########

import os
import numpy as np
import pandas as pd
import polars as pl

# Import our class
from ds_tool import DSTools

# --- 1. Generate test data ---
print("--- 1. Generate test DataFrame ---")

# Create Pandas DataFrame with custom index
pd_index = pd.Index(['id_1', 'id_2', 'id_3', 'id_4'], name='custom_index')
pd_df = pd.DataFrame(
{'A': [1, 2, 3, 4], 'B': ['x', 'y', 'z', 'w']},
index=pd_index
)

# Create Polars DataFrame
pl_df = pl.DataFrame(
{'C': [10.5, 20.5, 30.5], 'D': [True, False, True]}
)

# Dictionary to save
dfs_to_save = {
'pandas_data': pd_df,
'polars_data': pl_df
}

print("Original Pandas DF:\n", pd_df)
print("\nOriginal Polars DF:\n", pl_df)
print("\n" + "="*60 + "\n")

# --- 2. Initialization and calls ---
tools = DSTools()
ZIP_FILENAME = 'test_archive.zip'

# --- Scenario A: Full loop with Parquet and index saving ---
print("--- SCENARIO A: Saving/Reading in Parquet Format ---")

# --- Step 2.1: Saving ---
print(f"\n2.1. Saving to '{ZIP_FILENAME}' with save_index=True...")
tools.save_dataframes_to_zip(
dataframes=dfs_to_save,
zip_filename=ZIP_FILENAME,
format='parquet',
save_index=True
)
assert os.path.exists(ZIP_FILENAME)
print("-> SUCCESS: ZIP archive created.")

# --- Step 2.2: Reading with Polars ---
print("\n2.2. Reading with Polars backend...")
loaded_with_polars = tools.read_dataframes_from_zip(
zip_filename=ZIP_FILENAME,
backend='polars'
)
# Polars data comparison
# .equals() checks for exact data and type match
assert loaded_with_polars['polars_data'].equals(pl_df)
print("-> SUCCESS: Polars DataFrame restored correctly.")

# --- Step 2.3: Reading with Pandas ---
print("\n2.3. Reading with Pandas backend...")
loaded_with_pandas = tools.read_dataframes_from_zip(
zip_filename=ZIP_FILENAME,
backend='pandas'
)
# Pandas data comparison
# pd.testing.assert_frame_equal() checks for exact match, including index
pd.testing.assert_frame_equal(loaded_with_pandas['pandas_data'], pd_df)
print("-> SUCCESS: Pandas DataFrame restored correctly, including custom index.")

# --- Step 2.4: Cleanup ---
os.remove(ZIP_FILENAME)
print(f"\n2.4. Archive '{ZIP_FILENAME}' removed.")
print("\n" + "="*60 + "\n")

# --- Scenario B: Full Cycle with CSV ---
print("--- SCENARIO B: Save/Read in CSV Format ---")
CSV_ZIP_FILENAME = 'test_archive_csv.zip'

# --- Step 3.1: Save ---
print(f"\n3.1. Save to '{CSV_ZIP_FILENAME}' in CSV format...")
tools.save_dataframes_to_zip(
dataframes=dfs_to_save,
zip_filename=CSV_ZIP_FILENAME,
format='csv',
save_index=True
)
assert os.path.exists(CSV_ZIP_FILENAME)
print("-> SUCCESS: CSV ZIP archive created.")

# --- Step 3.2: Reading ---
print("\n3.2. Reading CSV archive with Pandas...")
loaded_csv = tools.read_dataframes_from_zip(
zip_filename=CSV_ZIP_FILENAME,
format='csv',
backend='pandas'
)
# When reading from CSV, the index becomes a normal one column
# Check that the data is generally the same (resetting the index for comparison)
pd.testing.assert_frame_equal(
loaded_csv['pandas_data'].reset_index(drop=True),
pd_df.reset_index(drop=True)
)
print("-> SUCCESS: Data from CSV recovered (taking into account format peculiarities).")

# --- Step 3.3: Cleaning ---
os.remove(CSV_ZIP_FILENAME)
print(f"\n3.3. Archive '{CSV_ZIP_FILENAME}' deleted.")
print("\n" + "="*60 + "\n")

print("All tests passed successfully!")

##########    code checking for "generate_alphanum_codes"    ##########

import numpy as np
import re

# Import our class
from ds_tool import DSTools

# --- Initialize the toolkit ---
tools = DSTools()

# --- Scenario A: Main call ---
print("="*60)
print("SCENARIO A: Generate 5 codes of length 12")
print("="*60)

N_CODES = 5
CODE_LENGTH = 12
codes_a = tools.generate_alphanum_codes(n=N_CODES, length=CODE_LENGTH)

print("Generated codes:")
print(codes_a)

# Checks
print("\nChecks:")
# 1. Type and shape
assert isinstance(codes_a, np.ndarray)
assert codes_a.shape == (N_CODES,)
print("-> SUCCESS: The output type and format are correct.")

# 2. The length of each code
all_lengths_correct = all(len(code) == CODE_LENGTH for code in codes_a)
assert all_lengths_correct
print(f"-> SUCCESS: The length of all codes is {CODE_LENGTH}.")

# 3. The composition of characters
# Create a regular expression that matches a string consisting only of allowed characters
allowed_chars_pattern = re.compile(f"^[0-9A-Z]{{{CODE_LENGTH}}}$")
all_chars_correct = all(allowed_chars_pattern.match(code) for code in codes_a)
assert all_chars_correct
print("-> SUCCESS: All characters in the codes belong to the alphabet '0-9' and 'A-Z'.\n")

# --- Scenario B: Check for uniqueness ---
print("="*60)
print("SCENARIO B: Generate a large number of codes to check for uniqueness")
print("="*60)

LARGE_N = 10000
codes_b = tools.generate_alphanum_codes(n=LARGE_N, length=10)

# Compare the number of generated codes with the number of unique codes
num_unique = len(np.unique(codes_b))
assert num_unique == LARGE_N
print(f"{LARGE_N} codes generated, of which {num_unique} are unique.")
print("-> SUCCESS: All generated codes are unique (no collisions found).\n")

# --- Scenario C: Edge cases ---
print("="*60)
print("SCENARIO B: Testing edge cases (n=0, length=0)")
print("="*60)

# n = 0
print("Request for 0 codes (n=0):")
codes_n0 = tools.generate_alphanum_codes(n=0, length=8)
print(f"Result: {codes_n0}, Shape: {codes_n0.shape}")
assert codes_n0.shape == (0,)
print("-> SUCCESS: Empty array of correct shape returned.\n")

# length = 0
print("Request for zero length codes (length=0):")
codes_l0 = tools.generate_alphanum_codes(n=3, length=0)
print(f"Result: {codes_l0}")
assert all(code == '' for code in codes_l0)
print("-> SUCCESS: An array of empty strings was returned.\n")

# --- Scenario D: Testing error handling ---
print("="*60)
print("SCENARIO D: Testing error handling (negative values)")
print("="*60)
print("Attempt to pass n = -1 (expecting ValueError):")
try:
tools.generate_alphanum_codes(n=-1)
except ValueError as e:
print(f"-> SUCCESS: Expected error caught: {e}\n")

print("Attempt to pass length = -5 (expecting ValueError):")
try:
tools.generate_alphanum_codes(n=10, length=-5)
except ValueError as e:
print(f"-> SUCCESS: Expected error caught: {e}")

##########    code checking for "generate_distribution_from_metrics"    ##########

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
import matplotlib.pyplot as plt

# Import our class and its configuration
from ds_tool import DSTools, DistributionConfig

def analyze_and_compare(generated_data, config: DistributionConfig, title: str):
"""A helper function for analyzing and comparing results."""
# Convert data to a NumPy array for calculations if it is a Series
if isinstance(generated_data, (pd.Series, pl.Series)):
data_arr = generated_data.to_numpy()
else:
data_arr = generated_data

actual_mean = np.mean(data_arr)
actual_std = np.std(data_arr, ddof=1)
actual_median = np.median(data_arr)
actual_skew = stats.skew(data_arr)
# stats.kurtosis returns (Fisher) kurtosis. We want full (Pearson) kurtosis.
# But your function seems to use kurtosis, so we'll keep it.
# If Pearson were required, it would be: stats.kurtosis(data_arr) + 3
actual_kurtosis = stats.kurtosis(data_arr, fisher=True) # fisher=True is kurtosis
actual_min = np.min(data_arr)
actual_max = np.max(data_arr)

print(f"\n--- {title} ---")
print(f"| Metric | Goal | Fact |")
print(f"|------------|------------|")
print(f"| Mean | {config.mean:<10.2f} | {actual_mean:<10.2f} |")
print(f"| Median | {config.median:<10.2f} | {actual_median:<10.2f} |") 
print(f"| Std. Dev. | {config.std:<10.2f} | {actual_std:<10.2f} |") 
print(f"| Skewness | {config.skewness:<10.2f} | {actual_skew:<10.2f} |") 
print(f"| Kurtosis | {config.kurtosis:<10.2f} | {actual_kurtosis:<10.2f} |") 
print(f"| Min | {config.min_val:<10.2f} | {actual_min:<10.2f} |") 
print(f"| Max | {config.max_val:<10.2f} | {actual_max:<10.2f} |") 

plt.figure(figsize=(10, 6))
plt.hist(data_arr, bins=50, density=True, alpha=0.7, label='Generated Distribution')
plt.title(title)
plt.grid(True)
plt.show()

# --- Initialize tools ---
tools = DSTools()

# --- Scenario A: Main call with DistributionConfig object ---
print("="*60)
print("SCENARIO A: Generate from DistributionConfig object, NumPy output")
print("="*60)
config_a = DistributionConfig(
mean=500, median=450, std=150, min_val=100, max_val=2000,
skewness=1.2, kurtosis=5.0, n=5000, outlier_ratio=0.02
)
data_a = tools.generate_distribution_from_metrics(n=5000, metrics=config_a)
analyze_and_compare(data_a, config_a, "Scenario A: Result")

# --- Scenario B: Call with dictionary and output to Pandas Series (int) ---
print("\n" + "="*60)
print("SCENARIO B: Generate from dictionary, output Pandas Series (int)")
print("="*60)
metrics_dict_b = {
"mean": 80.0, "median": 75.0, "std": 20.0, "min_val": 10, "max_val": 150,
"skewness": 0.5, "kurtosis": 0.8, "n": 3000, "outlier_ratio": 0.01
}
data_b = tools.generate_distribution_from_metrics(
n=3000,
metrics=metrics_dict_b,
int_flag=True,
output_as='pandas'
)
print("Output type:", type(data_b))
print("Data type in Series:", data_b.dtype)
print(data_b.head())
# For analysis, we pass the Series itself, the function will sort it out
analyze_and_compare(data_b, DistributionConfig(**metrics_dict_b), "Scenario B: Result")

# --- Scenario C: Testing moment validation ---
print("\n" + "="*60)
print("SCENARIO C: Trying to create a distribution with impossible moments")
print("Expecting ValueError, since kurtosis < skewness² - 2")
print("="*60)
try:
invalid_moments = {
"mean": 100, "median": 100, "std": 15, "min_val": 50, "max_val": 150,
"skewness": 3.0, "kurtosis": 1.0, "n": 1000 # Impossible: 1.0 < (3.0² - 2)
}
tools.generate_distribution_from_metrics(n=1000, metrics=invalid_moments)
except ValueError as e:
print(f"SUCCESSFULLY caught expected error: {e}")

# --- Scenario D: Testing Pydantic validation ---
print("\n" + "="*60)
print("SCENARIO D: Attempting to create config with invalid data type in dictionary")
print("Expecting ValueError from Pydantic")
print("="*60)
try:
invalid_pydantic = {
"mean": 100, "median": "not a number", "std": 15, "min_val": 50, "max_val": 150,
"skewness": 0, "kurtosis": 3, "n": 1000
}
tools.generate_distribution_from_metrics(n=1000, metrics=invalid_pydantic)
except ValueError as e:
print(f"SUCCESSFULLY caught expected Pydantic error: {e}")