import numpy as np
from joblib import Parallel, delayed
from joblib import cpu_count

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

def evaluate_recommender(recommender, at, URM_validation, n_jobs=-1, batch_size=1000):
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