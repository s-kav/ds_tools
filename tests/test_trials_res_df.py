import time

import numpy as np
import optuna
import pandas as pd


# --- Dummy Objective Function for Testing ---
def dummy_objective(trial: optuna.trial.Trial) -> float:
    """A dummy objective function that simulates model training."""
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_int("y", 0, 20)
    category = trial.suggest_categorical("category", ["A", "B", "C"])

    # Simulate a failure for a specific trial number
    if trial.number == 3:
        raise ValueError("Simulated error in trial #3")

    # Simulate pruning for a specific condition
    if y < 5:
        raise optuna.exceptions.TrialPruned("Simulating pruning for y < 5")

    # Simulate training time
    time.sleep(np.random.uniform(0.01, 0.03))

    # Calculate a dummy score
    score = (100 - x**2) + y - (5 if category == "C" else 0)
    return score


def test_optuna_optimization_and_results_df(tools):
    """
    Tests the trials_res_df method by running a mock Optuna study
    and verifying the output DataFrame.
    """
    study = optuna.create_study(direction="maximize")

    # Run optimization; errors and pruning are handled by Optuna
    try:
        study.optimize(dummy_objective, n_trials=15, catch=(ValueError,))
    except optuna.exceptions.TrialPruned:
        # We expect some trials to be pruned, so we can ignore this
        pass

    all_trials = study.trials
    completed_trials = [
        t for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    # There must be at least one completed trial for the test to be meaningful
    assert (
        len(completed_trials) > 0
    ), "Test setup failed: No trials completed successfully."

    # Call your method to convert trials list to DataFrame
    metric_name = "Accuracy"
    df = tools.trials_res_df(all_trials, metric=metric_name)

    # --- Assertions ---
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == len(completed_trials)
    assert df.columns[0] == metric_name
    assert "Duration" in df.columns
    assert "x" in df.columns and "y" in df.columns and "category" in df.columns

    # Check that the dataframe is sorted correctly
    assert df[metric_name].is_monotonic_decreasing
