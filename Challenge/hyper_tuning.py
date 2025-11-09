import json, os
import optuna

from Challenge import paths

""" Structure of performance_data:
{
    "study_name": {
        "best_trial": int,
        "best_score": float,
        "best_params": {param_name: param_value, ...},
        "folds": {
            <fold_number>: {
                "best_trial": int,
                "best_score": float,
                "best_params": {param_name: param_value, ...}
            },
            ...
        }
    },
    ...
}
"""

class ModelOptimizer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.file_path = os.path.join(paths.PERFORMANCE_LOG, f"{model_name}.json")
        
        self.performance_data = {}
        
        if os.path.exists(self.file_path):
            self.performance_data = self.load_performance()

        self.folds_logged = False
        self.folds = {}
    
    def log_fold_performance(self, fold_number, trial, score, params):
        self.folds_logged = True
        self.folds[fold_number] = {
            "best_trial": trial,
            "best_score": score,
            "best_params": params
        }

    # Callback method for Optuna
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.Trial):
        # Check if current trial is the best
        if study.best_trial.number == trial.number:
            self.performance_data[study.study_name] = {
                "best_trial": trial.number,
                "best_score": trial.value,
                "best_params": trial.params
            }
            
            if self.folds_logged:
                self.performance_data[study.study_name]["folds"] = self.folds
                self.folds_logged = False
                self.folds = {}
            
            self._save_to_file()

    def _save_to_file(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.performance_data, f, indent=4)

    def load_performance(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                self.performance_data = json.load(f)
        return self.performance_data
    
    def create_study(self, study_name, direction="maximize", load_if_exists=True):
        self.study = optuna.create_study(
            study_name=study_name,
            storage=paths.OPTUNA_STORAGE,
            direction=direction,
            load_if_exists=load_if_exists
        )
        return self.study

    def optimize(self, objective_function, n_trials=50, load_if_exists=True):
        self.study.optimize(
            objective_function,
            callbacks=[self],
            n_trials=n_trials,
            show_progress_bar=True
        )

        # Print study results
        pruned_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        print()
        print("Study statistics: ")
        print("  Number of finished trials: ", len(self.study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))
        print()
        print("Best Value:", self.study.best_value)
        print("Best Params:", self.study.best_params)
        return self.study

    def create_and_optimize_study(self, study_name, objective_function, n_trials=50, direction="maximize", load_if_exists=True):
        self.create_study(study_name, direction, load_if_exists)
        self.optimize(objective_function, n_trials, load_if_exists)
        return self.study

    def get_best_params(self, study_name):
        if study_name in self.performance_data:
            return self.performance_data[study_name]["best_params"]
        else:
            return None
        
    def print_performance_data(self):
        print(json.dumps(self.performance_data, indent=4))