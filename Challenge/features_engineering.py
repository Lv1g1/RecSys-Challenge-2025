import numpy as np
import pandas as pd
import scipy.sparse as sps
import gc
import tqdm

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Neural.MultVAE_PyTorch_Recommender import MultVAERecommender_PyTorch_OptimizerMask
from implicit.cpu.als import AlternatingLeastSquares

from typing import List, Dict, Tuple
from Challenge.utils import get_user_batches

def generate_candidates(
        URM_train: sps.csr_matrix,
        user_ids: np.ndarray,
        models: List[Tuple[str, BaseRecommender | AlternatingLeastSquares]],
        models_cutoff: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """ Generate candidate user-item pairs from multiple recommenders.
     Args:
         URM_train (sps.csr_matrix): User-Rating Matrix for training.
         user_ids (np.ndarray): Array of user IDs to generate candidates for.
         models (List[List[str, BaseRecommender | AlternatingLeastSquares]]): 
             List of tuples containing model names and their corresponding trained model instances.
         models_cutoff (Dict[str, Dict[str, int]]): 
             Dictionary mapping model names to their cutoff values for recommendations.
     Returns:
         pd.DataFrame: DataFrame containing unique user-item candidate pairs.
    """

    # We iterate through all budget configs to find the maximum required candidates for each model.
    generation_limits = {}
    
    # Also validate structure
    if not models_cutoff:
        raise ValueError("models_cutoff dictionary cannot be empty.")
        
    for budget_key, config in models_cutoff.items():
        for model_name, limit in config.items():
            current_max = generation_limits.get(model_name, 0)
            generation_limits[model_name] = max(current_max, limit)

    # Number of rows to pre-allocate (upper bound)
    estimated_rows = len(user_ids) * sum(generation_limits.values())
    
    all_users = np.empty(estimated_rows, dtype=np.int16) # 27k users fit in int16 (32k limit)
    all_items = np.empty(estimated_rows, dtype=np.int16) #  7k items fit in int16 (32k limit)
    
     # Flag Columns (Pre-allocate as int8)
    # Map each budget key (e.g., "top_10") to a numpy array
    flag_arrays = {}
    for key in models_cutoff.keys():
        flag_arrays[key] = np.zeros(estimated_rows, dtype=np.int8)

    current_ptr = 0

    for model_name, model in models:
        print(f"Computing candidates with {model_name}...")

        # Max cutoff for this model across all configs
        cutoff = generation_limits.get(model_name, 0)
        if cutoff == 0:
            print(f"Skipping {model_name} (Max cutoff is 0)...")
            continue

        print(f"Computing candidates with {model_name} (Max Cutoff: {cutoff})...")

        if type(model) == AlternatingLeastSquares:
            recommended_items, scores = model.recommend(
                user_ids, 
                URM_train, 
                N=cutoff, 
                filter_already_liked_items=True
            )

        elif type(model) == MultVAERecommender_PyTorch_OptimizerMask:
            # Do in batches to avoid OOM in GPU
            recommended_items = []
            for user_batch in get_user_batches(user_ids, batch_size=500):
                batch_recs = model.recommend(user_batch, cutoff=cutoff)
                recommended_items.extend(batch_recs)
            recommended_items = recommended_items

        else:
            recommended_items = model.recommend(user_ids, cutoff=cutoff)

        # Ensure rectangular numpy array (N_users, Cutoff)
        recs = np.array(recommended_items)
        
        n_users_batch, n_cols = recs.shape
        num_new = n_users_batch * n_cols

        # Flatten items
        flat_items = recs.flatten()
        
        # Create corresponding users array
        # Repeat every user_id N times, where N is the number of columns (cutoff)
        # shape[1] covers cases where model might return fewer than cutoff
        flat_users = np.repeat(user_ids, n_cols)
        
        # Calculate Ranks
        # Ranks are simply 0..(n_cols-1) repeated for each user
        ranks_matrix = np.tile(np.arange(n_cols), (n_users_batch, 1))
        flat_ranks = ranks_matrix.flatten()

        # Fill the pre-allocated arrays
        end_ptr = current_ptr + num_new
        all_users[current_ptr:end_ptr] = flat_users
        all_items[current_ptr:end_ptr] = flat_items
        
        # Update flag columns
        for budget_key, config in models_cutoff.items():
            # Get the strict limit for THIS budget key
            limit = config.get(model_name, 0)
            
            if limit > 0:
                # Flag is 1 if rank < limit
                # We reuse flat_ranks calculated above
                flag_arrays[budget_key][current_ptr:end_ptr] = (flat_ranks < limit).astype(np.int8)

        current_ptr = end_ptr

    # Cut arrays to actual filled size
    all_users = all_users[:current_ptr]
    all_items = all_items[:current_ptr]
    for key in flag_arrays:
        flag_arrays[key] = flag_arrays[key][:current_ptr]
    
    # CREATE DATAFRAME
    print("Constructing DataFrame...")
    data = {
        'UserID': all_users, 
        'ItemID': all_items
    }

    # Add flags directly using the keys from the dictionary (e.g., "top_10")
    for key, arr in flag_arrays.items():
        # Clean column name if needed, or just use the key
        col_name = key if key.startswith("top") or key.startswith("In_Budget") else f"In_Budget_{key}"
        data[col_name] = arr
        
    df = pd.DataFrame(data)

    # --- DEDUPLICATION & AGGREGATION ---
    print("Deduplicating and merging flags...")
    
    # Max aggregation handles the logical OR for flags
    # Group by should sort data
    df = df.groupby(['UserID', 'ItemID'], as_index=False).max()
    
    # Optimize types
    df['UserID'] = df['UserID'].astype(np.int16) # 27k < 32k
    df['ItemID'] = df['ItemID'].astype(np.int16) # 7k  < 32k
    for col in df.columns:
        if "top" in col or "Budget" in col:
            df[col] = df[col].astype(np.int8)
            
    # Check to see if is sorted
    assert df['UserID'].is_monotonic_increasing, "Dataframe not sorted."

    return df


def add_models_features(
        df: pd.DataFrame,
        URM: sps.csr_matrix,
        models: List[Tuple[str, BaseRecommender | AlternatingLeastSquares]]) -> pd.DataFrame:
    
    assert 'UserID' in df.columns and 'ItemID' in df.columns, "DataFrame must contain 'UserID' and 'ItemID' columns"
    assert df['UserID'].is_monotonic_increasing, "DataFrame 'UserID' column must be sorted in ascending order"

    new_features = {}

    n_users_global, n_items_global = URM.shape

    cand_users = df['UserID'].values
    cand_items = df['ItemID'].values
    unique_users = df['UserID'].unique()

    # Map Global UserIDs to Local Matrix Indices (0 to N_unique)
    # Since df is sorted, unique_users is sorted. searchsorted is fast.
    # This array tells us: "For row i in df, which row in the scores matrix should I look at?"
    local_user_indices = np.searchsorted(unique_users, cand_users)

    # unique users count should be all the users in df
    # but for safety we subset URM
    URM_subset = URM[unique_users]

    # Pre-calculate seen indices relative to the subset
    # seen_rows will be 0..N_unique-1, matching the scores matrix
    seen_rows, seen_cols = URM_subset.nonzero()

    for label, recommender in models:
        print(f"Processing features for model: {label}")

        if type(recommender) == MultVAERecommender_PyTorch_OptimizerMask:
            # Do in batches to avoid OOM in GPU
            scores = np.zeros((len(unique_users), n_items_global), dtype=np.float32)
            start_idx = 0
            for user_batch in get_user_batches(unique_users, batch_size=500):
                batch_scores = recommender._compute_item_score(user_id_array=user_batch)
                end_idx = start_idx + len(user_batch)
                scores[start_idx:end_idx] = batch_scores
                start_idx = end_idx
        
        elif type(recommender) == AlternatingLeastSquares:
            recommended_items, scores_raw = recommender.recommend(
                unique_users,
                URM_subset,
                N=n_items_global,
                filter_already_liked_items=False
            )

            # Safety check
            assert scores_raw.shape[1] == n_items_global, "ALS did not return scores for all items"

            # Simple Fix: Un-sort the results
            # rec_items contains ItemIDs. argsort gives us the indices that would sort those IDs (0, 1, 2...)
            sorted_indices = np.argsort(recommended_items, axis=1)
            
            # We use take_along_axis to apply those indices to the scores
            # This aligns the scores so Column 0 is ItemID 0, Column 1 is ItemID 1...
            scores = np.take_along_axis(scores_raw, sorted_indices, axis=1)

            # Cleanup raw large arrays
            del recommended_items, scores_raw
        
        else:
            scores = recommender._compute_item_score(user_id_array=unique_users)

        scores = np.array(scores)  # Ensure numpy array

        # Normalize
        norm_factor = np.linalg.norm(scores, np.inf, axis=1, keepdims=True)
        norm_factor[norm_factor == 0] = 1.0 
        linf_scores = scores / norm_factor

        # Remove seen items        
        # Set those specific entries to -inf
        linf_scores[seen_rows, seen_cols] = -np.inf

        # Calculate Ranks
        # argsort sorts ascending, so we use [::-1] to get descending (highest score first)
        # This returns INDICES of items. 
        # shape: (user_count, n_items)
        rank_order = np.argsort(linf_scores, axis=1)[:, ::-1]
        
        # We need the inverse mapping: item_id -> rank_position
        n_scores, n_items = linf_scores.shape
        rank_matrix = np.empty((n_scores, n_items), dtype=np.int32)
        
        # Fancy numpy trick to invert the permutation vectors in one go
        # Arrays of shape (n_scores, 1) needed for broadcasting
        row_indices = np.arange(n_scores)[:, None] 
        rank_matrix[row_indices, rank_order] = np.arange(n_items)
        
        # Extract Values for Candidate Pairs
        scores_final = linf_scores[local_user_indices, cand_items]
        ranks_final = rank_matrix[local_user_indices, cand_items]
        
        # Assign directly to DF
        new_features[f"{label}_Score"] = scores_final
        new_features[f"{label}_RankPosition"] = ranks_final
        new_features[f"{label}_Recommended"] = (ranks_final < 20).astype(np.int8)
    
        # Free memory immediately
        del scores, linf_scores, rank_matrix, rank_order, scores_final, ranks_final
        gc.collect()

    # Join new features to the original dataframe
    new_cols_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, new_cols_df], axis=1)

    return df


def calculate_item_item_features_fast(
        df: pd.DataFrame,
        URM: sps.csr_matrix,
        similarity_models: List[Tuple[str, BaseRecommender]]) -> pd.DataFrame:
    """
    Calculates features based on Item-Item similarity matrices.
    Generic version: pass any list of (label, model) tuples.
    
    Recommended Features:
    - MaxSim: The strongest single link to user history.
    - MeanSim: Average affinity to user history.
    - StdSim: Consistency of affinity.
    - MatchCount: How many items in history does this candidate look like? (Support)
    """
    # Check df is sorted by UserID
    assert df['UserID'].is_monotonic_increasing, "DataFrame must be sorted by UserID in increasing order."

    new_features = {}

    # Pre-calculate mapping: UserID -> [List of Candidate ItemIDs]
    # We use numpy split for max speed
    user_ids = df['UserID'].values
    item_ids = df['ItemID'].values
    
    # Find indices where user changes
    unique_users, user_starts = np.unique(user_ids, return_index=True)
    # Map UserID to (start_index, end_index) in the sorted arrays
    user_map = {}
    for i, user_id in enumerate(unique_users):
        start = user_starts[i]
        end = user_starts[i+1] if i + 1 < len(unique_users) else len(user_ids)
        user_map[user_id] = (start, end)
    
    # Prepare result dictionary
    N_ROWS = len(df)
    
    for label, recommender in similarity_models:
        print(f"Extracting MAX similarity for {label}...")
        
        # Get Sparse Matrix
        W_sparse = recommender.W_sparse
        if not sps.issparse(W_sparse):
            W_sparse = sps.csr_matrix(W_sparse)
            
        # Feature arrays
        max_sims = np.zeros(N_ROWS, dtype=np.float32)
        mean_sims = np.zeros(N_ROWS, dtype=np.float32)
        std_sims = np.zeros(N_ROWS, dtype=np.float32)
        match_counts = np.zeros(N_ROWS, dtype=np.int16)
        
        # Iterate over unique users in the candidates
        for user_id in tqdm.tqdm(unique_users):
            start, end = user_map[user_id]
            
            # Get Seen Items for this user (Indices)
            seen_items = URM.indices[URM.indptr[user_id]:URM.indptr[user_id+1]]
            
            if len(seen_items) == 0:
                continue
                
            # Get Candidate Items for this user
            cand_items = item_ids[start:end]
            
            # --- THE CORE OPTIMIZATION ---
            # Instead of .toarray(), we slice sparse matrix
            # Submatrix: Rows=Candidates, Cols=SeenItems
            # This is efficient because W is CSR (fast row slicing)
            sub_W = W_sparse[cand_items, :][:, seen_items]
            
            # Check if sub_W is effectively empty
            if sub_W.nnz > 0:
                # 1. Max Similarity: "Does it look like the best thing I bought?"
                max_sims[start:end] = sub_W.max(axis=1).toarray().flatten()
                
                # 2. Mean Similarity: "Does it look like everything I bought?"
                # Convert matrix object to array then flatten
                mean_sims[start:end] = np.array(sub_W.mean(axis=1)).flatten()

                # 3. Match Count: "How many of my items does this candidate look like?"
                # Counts non-zero elements per row
                match_counts[start:end] = sub_W.getnnz(axis=1)

                # 4. Std Dev: "Is the similarity consistent?"
                # Densify only if needed (Std requires dense usually)
                dense_batch = sub_W.toarray()
                std_sims[start:end] = dense_batch.std(axis=1)

        # Assign directly to DF
        new_features[f'{label}_MaxSim'] = max_sims
        new_features[f'{label}_MeanSim'] = mean_sims
        new_features[f'{label}_StdSim'] = std_sims
        new_features[f'{label}_MatchCount'] = match_counts

    # Join new features to the original dataframe
    new_cols_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, new_cols_df], axis=1)
    
    return df


def add_embedding_features(
        df: pd.DataFrame,
        als_model: AlternatingLeastSquares, 
        n_item_clusters=20, n_user_clusters=30,
        n_pca_components=5,
        batch_size=500, seed=42) -> pd.DataFrame:
    """
    Adds features based on ALS embeddings:
    - User and Item vector norms
    - PCA components of User and Item factors
    - Clustering of User and Item factors
    - Distances to cluster centroids
    """

    new_features = {}

    # Get factors from trained ALS model
    user_factors = als_model.user_factors
    item_factors = als_model.item_factors
    
    # PCA on User and Item factors
    print(f"  Running PCA to extract top {n_pca_components} components...")
    
    # Fit and Transform Users
    pca_u = PCA(n_components=n_pca_components, random_state=seed)
    user_pca = pca_u.fit_transform(user_factors).astype(np.float32)
    
    # Fit and Transform Items
    pca_i = PCA(n_components=n_pca_components, random_state=seed)
    item_pca = pca_i.fit_transform(item_factors).astype(np.float32)

    # Calculate Norms
    print("  Calculating Vector Norms...")
    user_norms = np.linalg.norm(user_factors, axis=1).astype(np.float32)
    item_norms = np.linalg.norm(item_factors, axis=1).astype(np.float32)

    # Cluster Users
    print("Clustering Users and Items...")
    kmeans_users = KMeans(n_clusters=n_user_clusters, random_state=seed)
    user_clusters = kmeans_users.fit_predict(user_factors)
    user_centroid_matrix = kmeans_users.cluster_centers_.astype(np.float32)

    # Cluster Items
    kmeans_items = KMeans(n_clusters=n_item_clusters, random_state=seed)
    item_clusters = kmeans_items.fit_predict(item_factors)
    item_centroid_matrix = kmeans_items.cluster_centers_.astype(np.float32)

    # --- 3. PRE-ALLOCATE RESULT ARRAYS ---
    # We allocate arrays for the full length of DF to avoid DataFrame fragmentation
    N_ROWS = len(df)
    
    # Output arrays
    res_u_cluster = np.zeros(N_ROWS, dtype=np.int16)
    res_i_cluster = np.zeros(N_ROWS, dtype=np.int16)
    res_inter_cluster = np.zeros(N_ROWS, dtype=np.int16)
    res_u_dist = np.zeros(N_ROWS, dtype=np.float32)
    res_i_dist = np.zeros(N_ROWS, dtype=np.float32)
    res_cross_dist = np.zeros(N_ROWS, dtype=np.float32)

    # New Arrays for Norms
    res_u_norm = np.zeros(N_ROWS, dtype=np.float32)
    res_i_norm = np.zeros(N_ROWS, dtype=np.float32)

    # PCA Arrays (Matrix of size N_ROWS x n_components)
    res_u_pca = np.zeros((N_ROWS, n_pca_components), dtype=np.float32)
    res_i_pca = np.zeros((N_ROWS, n_pca_components), dtype=np.float32)

    # Input arrays from DF (Read-only)
    all_user_ids = df['UserID'].values
    all_item_ids = df['ItemID'].values

    # --- 4. BATCHED FEATURE CALCULATION ---
    print(f"Computing features in batches of {batch_size}...")
    
    for start in tqdm.tqdm(range(0, N_ROWS, batch_size)):
        end = min(start + batch_size, N_ROWS)
        
        # A. Get IDs for this batch
        batch_u_ids = all_user_ids[start:end]
        batch_i_ids = all_item_ids[start:end]
        
        # B. Map to Clusters
        batch_u_clusters = user_clusters[batch_u_ids]
        batch_i_clusters = item_clusters[batch_i_ids]
        
        # Store Clusters
        res_u_cluster[start:end] = batch_u_clusters
        res_i_cluster[start:end] = batch_i_clusters
        res_inter_cluster[start:end] = batch_u_clusters * n_item_clusters + batch_i_clusters

        # C. Retrieve Factors & Centroids for this batch
        # Advanced Indexing -> Creates temporary small arrays (size of batch)
        # 1. User Vector vs User Centroid
        u_vecs = user_factors[batch_u_ids]
        u_cents = user_centroid_matrix[batch_u_clusters]
        
        # 2. Item Vector vs Item Centroid
        i_vecs = item_factors[batch_i_ids]
        i_cents = item_centroid_matrix[batch_i_clusters]
        
        # D. Calculate Distances
        # User eccentricity
        res_u_dist[start:end] = np.linalg.norm(u_vecs - u_cents, axis=1)
        
        # Item eccentricity
        res_i_dist[start:end] = np.linalg.norm(i_vecs - i_cents, axis=1)
        
        # Cross Distance (User Vector vs Item-Cluster Centroid)
        # This is the "Genre Affinity" score
        res_cross_dist[start:end] = np.linalg.norm(u_vecs - i_cents, axis=1)

        # Assign Norms directly
        res_u_norm[start:end] = user_norms[batch_u_ids]
        res_i_norm[start:end] = item_norms[batch_i_ids]

        # Assign PCA components directly
        res_u_pca[start:end] = user_pca[batch_u_ids]
        res_i_pca[start:end] = item_pca[batch_i_ids]

    # --- 5. ASSIGN TO DATAFRAME ---
    print("Assigning columns to DataFrame...")
    new_features['User_Cluster'] = res_u_cluster
    new_features['Item_Cluster'] = res_i_cluster
    new_features['Cluster_Interaction'] = res_inter_cluster
    new_features['User_Cluster_Dist'] = res_u_dist
    new_features['Item_Cluster_Dist'] = res_i_dist
    new_features['User_to_ItemCluster_Dist'] = res_cross_dist

    # Add the Norm features
    new_features['User_Vector_Norm'] = res_u_norm
    new_features['Item_Vector_Norm'] = res_i_norm
    
    # Add PCA Columns
    for i in range(n_pca_components):
        new_features[f'User_PCA_{i}'] = res_u_pca[:, i]
        new_features[f'Item_PCA_{i}'] = res_i_pca[:, i]

    # How well do the main traits match?
    new_features['PCA_Interaction_Score'] = (res_u_pca * res_i_pca).sum(axis=1)

    # User Eccentricity
    print("  Calculating User Eccentricity...")
    # Calculate global mean distance per cluster
    # We use a temporary DF for the groupby, then map it back
    # This is safe because it uses the data we just calculated
    temp_df = pd.DataFrame({'C': res_u_cluster, 'D': res_u_dist})
    cluster_means = temp_df.groupby('C')['D'].transform('mean').values
    
    # Avoid division by zero
    cluster_means[cluster_means == 0] = 1.0 
    
    # "How much further am I from the center than the average user in my cluster?"
    new_features['User_Eccentricity'] = res_u_dist / cluster_means

    # Join new features to the original dataframe
    new_cols_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, new_cols_df], axis=1)

    return df


def add_aggregate_features_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates:
    1. Aggregate statistics (Mean, Std, Min, Max)
    2. Model Agreement (Rank Spread)
    3. Relative Confidence (Score Ratios)
    """
    new_features = {}
    epsilon = 1e-9 # To prevent division by zero

    # -------------------------------------------------------------------------
    # 1. RANK STATISTICS & AGREEMENT
    # -------------------------------------------------------------------------
    # Identify columns ending in _RankPosition (Base Model Ranks)
    position_columns = [col for col in df.columns if col.endswith('_RankPosition')]
    
    if position_columns:
        print(f"Calculating Rank Stats on {len(position_columns)} models...")
        
        # Basic Aggregates
        new_features['Mean_RankPosition'] = df[position_columns].mean(axis=1)
        new_features['Std_RankPosition'] = df[position_columns].std(axis=1)
        
        # Best Case (Min) and Worst Case (Max)
        new_features['Min_RankPosition'] = df[position_columns].min(axis=1)   # Best Rank (e.g., 1)
        new_features['Worst_RankPosition'] = df[position_columns].max(axis=1) # Worst Rank (e.g., 100)
        
        # AGREEMENT: Rank Spread
        new_features['Model_Rank_Spread'] = new_features['Worst_RankPosition'] - new_features['Min_RankPosition']
    
    position_columns = [col for col in df.columns if col.endswith('_Recommended')]
    # Number of models that would rank this item in top 20
    if position_columns:
        print(f"Calculating Vote Count on {len(position_columns)} models...")
        new_features['Recommendation_Vote_Count'] = df[position_columns].sum(axis=1) 

    # -------------------------------------------------------------------------
    # 2. SCORE STATISTICS & RATIOS
    # -------------------------------------------------------------------------
    # Identify columns ending in _Score (Base Model Scores)
    score_columns = [col for col in df.columns if col.endswith('_Score')]

    if score_columns:
        print(f"Calculating Score Stats & Ratios on {len(score_columns)} models...")
        
        # Basic Aggregates
        new_features['Mean_Score'] = df[score_columns].mean(axis=1)
        new_features['Std_Score'] = df[score_columns].std(axis=1)
        new_features['Max_Score'] = df[score_columns].max(axis=1)
        new_features['Min_Score'] = df[score_columns].min(axis=1) # Useful to see the floor

        # RELATIVE FEATURES (Ratios)
        # We loop through the base columns and compare them to the aggregates
        # This tells XGBoost: "Is this specific model higher than the average consensus?"
        for col in score_columns:
            # Ratio to Mean (How much better/worse than average?)
            new_features[f'{col}_to_Mean_Ratio'] = df[col] / (new_features['Mean_Score'] + epsilon)
            
            # Ratio to Max (How close to the most confident model?)
            new_features[f'{col}_to_Max_Ratio'] = df[col] / (new_features['Max_Score'] + epsilon)
            
            # (Optional) Normalized Score Deviation
            # df[f'{col}_ZScore'] = (df[col] - df['Mean_Score']) / (df['Std_Score'] + epsilon)


    # Join everything at once
    new_cols_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, new_cols_df], axis=1)

    return df


def add_user_stats(df: pd.DataFrame, URM: sps.csr_matrix) -> pd.DataFrame:
    """
    Adds User and Item statistics.
    Enhanced to include 'User Mainstreamness' (Average popularity of items bought by user).
    """
    new_features = {}
    
    # 1. Map IDs
    user_ids = df['UserID'].values
    item_ids = df['ItemID'].values
    
    # 2. Basic Counts
    # User Profile Length
    user_profile_len = np.ediff1d(URM.indptr)
    new_features['User_Profile_Len'] = user_profile_len[user_ids]
    
    # Item Global Popularity
    # We use .tocsc() to slice columns efficiently, or just sum the binary URM columns
    # URM is binary (0/1), so column sum = popularity
    item_popularity = np.array(URM.sum(axis=0)).flatten()
    new_features['Item_Global_Popularity'] = item_popularity[item_ids]
    
    # 3. ADVANCED: User Mainstreamness
    # "Does this user usually buy popular items or niche items?"
    # Math: (URM * Popularity_Vector) / Profile_Length
    
    # Calculate sum of popularity of items in user's history
    # URM (N, M) dot Popularity (M, 1) -> (N, 1)
    user_total_popularity = URM.dot(item_popularity)
    
    # Calculate Average (handle division by zero for cold users)
    # We use a safe division helper
    with np.errstate(divide='ignore', invalid='ignore'):
        user_avg_popularity = user_total_popularity / user_profile_len
        user_avg_popularity[np.isnan(user_avg_popularity)] = 0  # Handle 0 profile length
        user_avg_popularity[np.isinf(user_avg_popularity)] = 0
        
    new_features['User_Avg_Item_Popularity'] = user_avg_popularity[user_ids]
    
    # Normalize to avoid scale issues (Log is usually safer for popularity)
    new_features['Log_Item_Pop'] = np.log1p(new_features['Item_Global_Popularity'])
    new_features['Log_User_Pop'] = np.log1p(new_features['User_Avg_Item_Popularity'])
    
    # The Mismatch
    # Low value = This item fits the user's usual "vibe"
    # High value = This item is too popular/niche for this user
    new_features['Pop_Mismatch'] = np.abs(new_features['Log_Item_Pop'] - new_features['Log_User_Pop'])
    
    # Join new features to the original dataframe
    new_cols_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, new_cols_df], axis=1)
    
    return df


def optimize_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcasts columns to the smallest possible type to save RAM.
    """
    print("Optimizing memory usage...")
    
    # 1. Drop Constant Columns (e.g. top_200 which is always 1)
    nunique = df.nunique()
    cols_to_drop = nunique[nunique == 1].index
    if len(cols_to_drop) > 0:
        print(f"Dropping constant columns: {list(cols_to_drop)}")
        df.drop(columns=cols_to_drop, inplace=True)

    # 2. Downcast Integers (Manual Sections)
    
    # --- A. INT8 (Max 127) ---
    # Flags, Clusters, Small Counts
    # Added: 'Vote_Count', 'Recommended'
    int8_candidates = [
        c for c in df.columns if 
        'top_' in c or 
        'Budget' in c or 
        'Label' in c or 
        'Recommended' in c or 
        'Vote_Count' in c or
        ('Cluster' in c and 'Dist' not in c and 'Interaction' not in c)
    ]
    for col in int8_candidates:
        df[col] = df[col].astype(np.int8)

    # --- B. INT16 (Max 32,767) ---
    # IDs, Ranks, Interaction Clusters, Small Stats
    # Added: 'MatchCount', 'Rank_Spread', 'Min_Rank', 'Worst_Rank'
    int16_candidates = [
        c for c in df.columns if 
        'UserID' in c or 
        'ItemID' in c or 
        'RankPosition' in c or
        'Rank_Spread' in c or
        'Cluster_Interaction' in c or
        'MatchCount' in c or
        'User_Profile_Len' in c or
        'Item_Global_Popularity' in c
    ]
    
    for col in int16_candidates:
        # Safety check
        if df[col].max() > 32000:
             df[col] = df[col].astype(np.int32)
        else:
             df[col] = df[col].astype(np.int16)

    # 3. Downcast Floats (Everything else)
    fcols = df.select_dtypes(include=['float', 'float64']).columns
    for col in fcols:
        df[col] = df[col].astype(np.float32)
        
    return df


def sanity_check(df, verbose=True):
    print("--- STARTING SANITY CHECK ---")
    problems_found = False

    # 0. Check DataFrame is sorted by UserID
    if not df['UserID'].is_monotonic_increasing:
        print("\n[CRITICAL] DataFrame is not sorted by UserID in increasing order.")
        problems_found = True

    # 1. Check for Missing Values (NaN)
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("\n[CRITICAL] NaN Values Found:")
        print(null_counts[null_counts > 0])
        problems_found = True
    else:
        if verbose: print("[OK] No NaNs found.")

    # 2. Check for Infinite Values (inf / -inf)
    # Common issue when normalizing by zero variance or dividing scores
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum()
    if inf_counts.sum() > 0:
        print("\n[CRITICAL] Infinite Values Found (Division by Zero?):")
        print(inf_counts[inf_counts > 0])
        problems_found = True
    else:
        if verbose: print("[OK] No Infinite values found.")

    # 3. Check for Duplicates (UserID, ItemID)
    # Stacking fails if you have multiple rows for the same User-Item pair
    if df.duplicated(subset=['UserID', 'ItemID']).any():
        n_dupes = df.duplicated(subset=['UserID', 'ItemID']).sum()
        print(f"\n[CRITICAL] Duplicate (UserID, ItemID) pairs found: {n_dupes}")
        problems_found = True
    else:
        if verbose: print("[OK] Keys (UserID, ItemID) are unique.")

    # 4. Check Data Types (IDs must be int)
    # Merges (pd.merge) often convert ints to float if there were missing keys initially
    if df['UserID'].dtype not in [int, np.int16, np.int32, np.int64]:
        print(f"\n[WARNING] UserID is {df['UserID'].dtype}, expected int. (Did a merge fail?)")
        # Auto-fix attempt
        # df['UserID'] = df['UserID'].astype(int) 
    
    if df['ItemID'].dtype not in [int, np.int16, np.int32, np.int64]:
        print(f"\n[WARNING] ItemID is {df['ItemID'].dtype}, expected int.")

    # 5. Check for Constant Columns (Zero Variance)
    # These crash some implementations of Normalization and add no info to XGBoost
    std_devs = df[numeric_cols].std()
    constant_cols = std_devs[std_devs == 0].index.tolist()
    if len(constant_cols) > 0:
        print("\n[WARNING] The following columns have ZERO variance (Constant values):")
        print(constant_cols)
        print("Recommendation: Drop them.")
    
    # 6. Check Logic (Ranks shouldn't be negative)
    rank_cols = [c for c in df.columns if 'Rank' in c and 'Skew' not in c and 'Kurtosis' not in c]
    if rank_cols:
        min_ranks = df[rank_cols].min()
        if (min_ranks < 0).any():
             print("\n[CRITICAL] Negative Ranks found (Logic Error):")
             print(min_ranks[min_ranks < 0])
             problems_found = True

    if problems_found:
        print("\n--- SANITY CHECK FAILED: Fix errors before training ---")
        # raise ValueError("Data Integrity Issues Found") # Uncomment to force stop
    else:
        print("\n--- SANITY CHECK PASSED: Data is clean ---")

    return not problems_found