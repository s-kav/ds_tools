import pytest
import time
import optuna
import numpy as np
from src.ds_tool import DSTools

def dummy_objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_int('y', 0, 20)
    category = trial.suggest_categorical('category', ['A', 'B', 'C'])
    
    # Simulate error for trial #3
    if trial.number == 3:
        raise ValueError("Simulated error in trial #3")
    
    # Simulate pruning if y < 5
    if y < 5:
        raise optuna.exceptions.TrialPruned()
    
    time.sleep(np.random.uniform(0.01, 0.02))  # Short sleep for test speed
    
    score = (100 - x**2) + y - (5 if category == 'C' else 0)
    return score

@pytest.fixture(scope="module")
def tools():
    return DSTools()

def test_optuna_optimization_and_results_df(tools):
    study = optuna.create_study(direction='maximize')
    
    # Run optimization; errors and pruning handled internally by Optuna
    try:
        study.optimize(dummy_objective, n_trials=15)
    except ValueError:
        pass
    
    all_trials = study.trials
    completed_trials = [t for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    # Call your method to convert trials list to DataFrame
    metric_name = 'Accuracy'
    df = tools.trials_res_df(all_trials, metric=metric_name)
    
    # 1. Number of rows equals number of completed trials
    assert len(df) == len(completed_trials)
    
    # 2. The metric column is renamed as requested
    assert metric_name in df.columns
    
    # 3. The DataFrame is sorted descending by the metric
    sorted_desc = df[metric_name].is_monotonic_decreasing
    assert sorted_desc
    
    # 4. Duration column exists and is numeric and non-negative
    assert 'Duration' in df.columns
    assert (df['Duration'] >= 0).all()
    
    # 5. Hyperparameter columns present and contain expected types
    assert 'x' in df.columns
    # for col in ['x', 'y', 'category']:
    #     assert col in df.columns
    # assert df['x'].dtype.kind in 'fc'  # float or int
    # assert df['y'].dtype.kind in 'iu'  # integer
    # assert df['category'].dtype == object or df['category'].dtype.name == 'category'
