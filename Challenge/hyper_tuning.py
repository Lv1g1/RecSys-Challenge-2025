import pandas as pd
import optuna
from typing import Tuple
import os

from Challenge import paths
# Ensure the directory for OPTUNA_STORAGE exists
os.makedirs(os.path.dirname(paths.OPTUNA_STORAGE), exist_ok=True)
default_storage = f"sqlite:///{paths.OPTUNA_STORAGE}"

# Callback class to save results
class SaveResults:
    def __init__(self):
        self.results = []

    def __call__(self, study, trial):
        data = trial.params.copy()
        data["score"] = trial.value
        data["trial"] = trial.number
        self.results.append(data)

# Function to perform hyperparameter tuning
# Takes an objective function as input
def hyperparameter_tuning(objective_function, study_name, n_trials=50, n_jobs=-1, storage=default_storage, seed=42) ->Tuple[SaveResults, optuna.study.Study]:
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True
    )
    
    callback = SaveResults()

    study.optimize(
        objective_function,
        callbacks=[callback],
        n_trials=n_trials,
        show_progress_bar=True
    )

    results_df = pd.DataFrame(callback.results)

    # Print study results
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print()
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print()
    print("Best Value:", study.best_value)
    print("Best Params:", study.best_params)

    return results_df, study