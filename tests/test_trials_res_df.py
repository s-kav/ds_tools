import time
import optuna
import pandas as pd
import numpy as np

# Import our class
from src.ds_tool import DSTools

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