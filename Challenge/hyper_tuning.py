import paths

import pandas as pd
import scipy.sparse as sps
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Evaluation.Evaluator import EvaluatorHoldout
import optuna

# Load data
URM_train = sps.load_npz(paths.URM_TRAIN)
URM_validation = sps.load_npz(paths.URM_VALIDATION)

# Initialize evaluator
evaluator_validation = EvaluatorHoldout(URM_validation, [10])

# Objective function that will be run by the optimizer
def objective_function(optuna_trial):
    
    recommender_instance = ItemKNNCFRecommender(URM_train)
    recommender_instance.fit(topK = optuna_trial.suggest_int("topK", 5, 1000),
                             shrink = optuna_trial.suggest_int("shrink", 0, 1000),
                             similarity = "cosine",
                             normalize = optuna_trial.suggest_categorical("normalize", [True, False])
                            )
    
    result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)
    
    return result_df.loc[10]["MAP"]


# Callback class to save results
class SaveResults(object):
    
    def __init__(self):
        self.results_df = pd.DataFrame(columns = ["result"])
    
    def __call__(self, optuna_study, optuna_trial):
        hyperparam_dict = optuna_trial.params.copy()
        hyperparam_dict["result"] = optuna_trial.values[0]
        
        self.results_df = self.results_df.append(hyperparam_dict, ignore_index=True)

# Run optimization
optuna_study = optuna.create_study(direction="maximize")
        
save_results = SaveResults()
        
optuna_study.optimize(objective_function,
                      callbacks=[save_results],
                      n_trials = 50)


# Print study results
pruned_trials = [t for t in optuna_study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in optuna_study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print("Study statistics: ")
print("  Number of finished trials: ", len(optuna_study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
print("  Value Validation: ", optuna_study.best_trial.value)

print("  Params: ", optuna_study.best_trial.params)
print("All results:")
print(save_results.results_df)

# Train the best model on the full training + validation set
recommender_instance = ItemKNNCFRecommender(URM_train + URM_validation)
recommender_instance.fit(**optuna_study.best_trial.params)