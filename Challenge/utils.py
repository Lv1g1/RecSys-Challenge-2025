import numpy as np
from joblib import Parallel, delayed
from joblib import cpu_count
from tqdm import tqdm

CPU_COUNT = cpu_count()

def _worker_evaluate_subset(user_subset, recommender, at, URM_validation, batch_size):
    """
    This function runs on a separate CPU core.
    It processes a specific subset of users in batches.
    """
    cumulative_recall = 0.0
    num_eval = 0
    
    # Loop over the subset in chunks (batches)
    for i in range(0, len(user_subset), batch_size):
        # Get the chunk of user_ids
        batch_users = user_subset[i : i + batch_size]
        
        # 1. VECTORIZED PREDICTION (The fast part)
        batch_recommendations = recommender.recommend(batch_users, cutoff=at)
        
        # 2. METRIC CALCULATION
        for j, user_id in enumerate(batch_users):
            start_pos = URM_validation.indptr[user_id]
            end_pos = URM_validation.indptr[user_id+1]
            relevant_items = URM_validation.indices[start_pos:end_pos]
            
            if len(relevant_items) > 0:
                num_eval += 1
                recommended_items = batch_recommendations[j]
                
                is_relevant = np.isin(recommended_items, relevant_items, assume_unique=True)
                recall_score = np.sum(is_relevant, dtype=np.float64) / relevant_items.shape[0]
                cumulative_recall += recall_score
                
    return cumulative_recall, num_eval

def evaluate_recommender_parallel(recommender, at, URM_validation, n_jobs=-1, batch_size=1000):
    """
    n_jobs: Number of CPU cores (-1 = all cores)
    batch_size: How many users to predict at once inside each core
    """
    users_to_eval = np.arange(URM_validation.shape[0])

    effective_jobs = CPU_COUNT if n_jobs == -1 else n_jobs
    
    # Split the user array into roughly equal parts
    user_splits = np.array_split(users_to_eval, effective_jobs)

    # Dispatch to Joblib
    # prefer="processes" is essential to bypass the Python GIL
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_worker_evaluate_subset)(
            split, 
            recommender, 
            at, 
            URM_validation, 
            batch_size
        ) 
        for split in user_splits
    )
    
    # Aggregate results from all cores
    total_recall = sum(r[0] for r in results)
    total_eval = sum(r[1] for r in results)
    
    if total_eval == 0:
        return 0.0
        
    return total_recall / total_eval

def evaluate_recommender(recommender, at, URM_validation, batch_size=1000):
    """
    Batched single-core evaluation.
    """
    cumulative_recall = 0.0
    num_eval = 0
    num_users = URM_validation.shape[0]

    # Iterate over users in batches
    for start_pos in tqdm(range(0, num_users, batch_size), desc="Eval Batches"):
        end_pos = min(start_pos + batch_size, num_users)
        
        users_batch = np.arange(start_pos, end_pos)
        target_block = URM_validation[start_pos:end_pos]
        
        # Get Recommendations
        recommended_items_batch = recommender.recommend(users_batch, cutoff=at)
        
        # Calculate Metric for the batch
        for i in range(len(users_batch)):
            start_ptr = target_block.indptr[i]
            end_ptr = target_block.indptr[i+1]
            relevant_items = target_block.indices[start_ptr:end_ptr]
            
            if len(relevant_items) > 0:
                num_eval += 1
                
                recs = recommended_items_batch[i]
                
                hit_count = np.isin(recs, relevant_items, assume_unique=True).sum()
                
                cumulative_recall += hit_count / len(relevant_items)

    if num_eval == 0:
        return 0.0
        
    return cumulative_recall / num_eval

def evaluate_recommender_implicit(recommender, at, URM_train, URM_validation, batch_size=1000):
    """
    Batched single-core evaluation adapted for the 'implicit' library.
    """
    cumulative_recall = 0.0
    num_eval = 0
    num_users = URM_validation.shape[0]
    
    # Implicit requires the training matrix to filter already liked items.
    # Ensure it is in CSR format for speed.
    URM_train = URM_train.tocsr()
    URM_validation = URM_validation.tocsr()

    # Iterate over users in batches
    for start_pos in tqdm(range(0, num_users, batch_size), desc="Eval Batches"):
        end_pos = min(start_pos + batch_size, num_users)
        
        users_batch = np.arange(start_pos, end_pos)
        
        # Slices of matrices for this batch
        target_block = URM_validation[start_pos:end_pos]
        train_block = URM_train[start_pos:end_pos]
        
        # --- CHANGES HERE ---
        # 1. Pass the training data slice so implicit can filter seen items
        # 2. Unpack the tuple: implicit returns (indices, scores)
        recommended_items_batch, scores_batch = recommender.recommend(
            users_batch, 
            train_block, 
            N=at, 
            filter_already_liked_items=True
        )
        
        # Calculate Metric for the batch
        for i in range(len(users_batch)):
            start_ptr = target_block.indptr[i]
            end_ptr = target_block.indptr[i+1]
            relevant_items = target_block.indices[start_ptr:end_ptr]
            
            if len(relevant_items) > 0:
                num_eval += 1
                
                # Get the row of recommendations for this specific user
                recs = recommended_items_batch[i]
                
                hit_count = np.isin(recs, relevant_items, assume_unique=True).sum()
                
                cumulative_recall += hit_count / len(relevant_items)

    if num_eval == 0:
        return 0.0
        
    return cumulative_recall / num_eval

from Recommenders.BaseRecommender import BaseRecommender
from Challenge import paths
import os, json
from typing import Dict, Type

def get_best_params(json_path: str) -> dict:
    with open(json_path, "r") as f:
        data = json.load(f)
        
        best_study = None
        for _, values in data.items():
            if best_study is None:
                best_study = values
                continue

            if values["best_score"] > best_study["best_score"]:
                best_study = values
                
    return best_study["best_params"]


def train_and_save_model(URM_train, model_name, model_class: Type[BaseRecommender], model_folder):
    print(f"  Training model: {model_name}")
    model_instance = model_class(URM_train)
    
    try:
        params = get_best_params(os.path.join(paths.PERFORMANCE_LOG, f"{model_name}.json"))
    except Exception as e:
        print(f"    Could not load parameters for {model_name}: {e}")
        print("    Using default parameters")
        params = {}

    model_instance.fit(**params)
    model_instance.save_model(model_folder, model_name)

    # Save parameters used
    with open(os.path.join(model_folder, model_name+"_params.json"), "w") as f:
        json.dump(params, f)

import os
import gc
from typing import Dict, Type, Iterator, Tuple

def load_models(URM_train, mapping: Dict[str, Type[BaseRecommender]], model_folder) -> Iterator[Tuple[str, BaseRecommender]]:
    model_folder = os.path.join(paths.MODEL_DIR, model_folder)
    os.makedirs(model_folder, exist_ok=True)
    
    # Check that all the models are available, if not train and save them
    for model_name, model_class in mapping.items():
        if not os.path.exists(os.path.join(model_folder, model_name+".zip")):
            print("Model not found.")
            train_and_save_model(URM_train, model_name, model_class, model_folder)
            
            gc.collect()  # Clean up memory
        else:
            print(f"Model found: {model_name}")

            # Check if parameters file exists
            params_path = os.path.join(model_folder, model_name+"_params.json")
            if not os.path.exists(params_path):
                print(f"Parameters file not found for {model_name}. Regenerating...")
                train_and_save_model(URM_train, model_name, model_class, model_folder)
                gc.collect()  # Clean up memory
            else:
                # Check if the parameters are up to date
                with open(params_path, "r") as f:
                    saved_params = json.load(f)
                try:
                    best_params = get_best_params(os.path.join(paths.PERFORMANCE_LOG, f"{model_name}.json"))
                    if saved_params != best_params:
                        print(f"Parameters for {model_name} are outdated. Regenerating...")
                        train_and_save_model(URM_train, model_name, model_class, model_folder)
                        gc.collect()  # Clean up memory
                except Exception as e:
                    print(f"    Could not load parameters for {model_name}: {e}")
                    print("    Regenerating model with default parameters")
                    train_and_save_model(URM_train, model_name, model_class, model_folder)
                    gc.collect()  # Clean up memory
            
    # Load all models (generator expression to save memory)
    for model_name, model_class in mapping.items():
        print(f"Loading {model_name}...")
        model_instance = model_class(URM_train)
        model_instance.load_model(model_folder, model_name)
        
        yield model_name, model_instance

        # CLEANUP: This runs when the caller asks for the NEXT item
        print(f"Unloading {model_name}...")
        del model_instance
        gc.collect()    
