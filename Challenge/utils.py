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

import os
import gc  # Garbage Collector
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

class XGBoostRerankerRecommender:
    RECOMMENDER_NAME = "XGBoostRerankerRecommender"
    
    def __init__(self, XGB_model, df):
        assert df['UserID'].is_monotonic_increasing, "UserID column is not sorted in increasing order!"
        # df = df.sort_values(by='UserID').reset_index(drop=True)

        # Separate Features and Indices
        self.XGB_model = XGB_model
        self.index_df = df[['UserID', 'ItemID']].copy()
        self.df = df.drop(columns=["UserID", "ItemID", "Label"], errors='ignore')
        
        # Pre-calculate start/end positions for every user
        self.user_map = {}
        
        # Groupby on the sorted dataframe to get indices efficiently
        grouped = self.index_df.groupby('UserID')
        
        # Store the indices array directly in a dict
        for user_id, indices in grouped.indices.items():
            self.user_map[user_id] = indices

    def recommend(self, user_ids, cutoff=20):
        batch_indices = []
        user_segment_info = []
        
        for uid in user_ids:
            if uid in self.user_map:
                indices = self.user_map[uid]
                batch_indices.append(indices)
                user_segment_info.append((uid, len(indices)))
            else:
                raise ValueError(f"User ID {uid} not found in the data.")

        if not batch_indices:
            return np.array([[] for _ in user_ids])

        # Flatten list of arrays into one big array for slicing
        flat_indices = np.concatenate(batch_indices)
        
        # Vectorized Extraction
        X_batch = self.df.iloc[flat_indices]
        items_batch = self.index_df['ItemID'].iloc[flat_indices].to_numpy()
        
        # Single Prediction Call, minimizing XGBoost overhead
        all_scores = self.XGB_model.predict(X_batch)
        
        # Split results back into users
        recommendations = []
        current_ptr = 0
        
        for uid, count in user_segment_info:
            if count == 0:
                recommendations.append([])
                continue
            
            # Slice the specific segment for this user
            scores = all_scores[current_ptr : current_ptr + count]
            items = items_batch[current_ptr : current_ptr + count]
            current_ptr += count
            
            # Sort Top-K for this user
            if count > cutoff:
                # Get indices of top K
                top_k_idx = np.argpartition(scores, -cutoff)[-cutoff:]
                # Sort those top K strictly descending
                sorted_top_k = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
                top_items = items[sorted_top_k]
            else:
                raise ValueError(f"User ID {uid} has not enough candidates to rerank.")
                
            recommendations.append(top_items.tolist())
            
        return np.array(recommendations, dtype=object)    
