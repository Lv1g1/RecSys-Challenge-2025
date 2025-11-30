import numpy as np
import pandas as pd
import xgboost as xgb
import time
import gc
import sys

class EfficientRankerIterator(xgb.DataIter):
    def __init__(self, file_paths, drop_cols=None, categorical_cols=None):
        self.file_paths = file_paths
        self.drop_cols = drop_cols if drop_cols is not None else []
        self.categorical_cols = categorical_cols if categorical_cols is not None else []
        self.current_file_idx = 0
        self.total_files = len(file_paths)
        super().__init__(cache_prefix=None) 

    def reset(self):
        print(f"\n--- Resetting Iterator (Starting new pass) ---")
        self.current_file_idx = 0

    def next(self, input_data):
        # Stop condition
        if self.current_file_idx >= self.total_files:
            print("--- End of Iteration ---")
            return 0 

        # Timer start
        start_time = time.time()
        path = self.file_paths[self.current_file_idx]
        
        # --- LOGGING START ---
        print(f"[Batch {self.current_file_idx + 1}/{self.total_files}] Loading: {path.split('/')[-1]}", end=' ... ')
        sys.stdout.flush() # Force print to appear immediately

        # --- LOADING ---
        df = pd.read_parquet(path)
        
        # Extract Group Info
        group_counts = df.groupby('UserID', sort=False).size().to_numpy()
        y = df['Label']

        # Stats for logging
        n_rows = len(df)
        n_groups = len(group_counts)

        # Drop ID cols and Target
        drop_cols = ['UserID', 'ItemID', 'Label'] + self.drop_cols
        X = df.drop(columns=drop_cols)

        # Handle Categoricals
        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype('category')
        
        # --- PASS TO XGBOOST ---
        # Note: This step takes time as XGBoost builds the histogram sketch here
        input_data(
            data=X,
            label=y,
            group=group_counts,
            feature_names=list(X.columns),
            feature_types=list(X.dtypes.apply(lambda x: 'c' if x.name == 'category' else 'q')) 
        )
        
        # --- CLEANUP ---
        self.current_file_idx += 1
        
        # Explicit deletion
        del df, X, y, group_counts
        
        # Force Garbage Collection and get count of objects freed
        freed = gc.collect() 
        
        elapsed = time.time() - start_time
        
        # --- LOGGING END ---
        print(f"Done. ({elapsed:.2f}s) | Rows: {n_rows} | Groups: {n_groups} | RAM Cleanup: {freed} objs")
        sys.stdout.flush()

        return 1


class ProgressCallback(xgb.callback.TrainingCallback):
    def __init__(self, total_trees):
        self.total_trees = total_trees
    def after_iteration(self, model, epoch, evals_log):
        print(f"\r[Training] Tree {epoch + 1}/{self.total_trees}", end="")
        return False


class XGBoostNativeRecommender:
    RECOMMENDER_NAME = "XGBoostNativeRecommender"
    
    def __init__(self, XGB_model, df):
        """
        XGB_model: The trained xgb.Booster object
        df: The evaluation dataframe containing features + UserID + ItemID
        """
        # 1. Ensure DataFrame is sorted by UserID
        assert df['UserID'].is_monotonic_increasing, "UserID column is not sorted in increasing order!"
            
        self.XGB_model = XGB_model
        
        # 2. Store IDs separately for final result mapping
        self.index_df = df[['UserID', 'ItemID']].copy()
        
        # 3. Prepare Features (Drop Metadata)
        # CRITICAL: We drop Label here too. We don't need it for prediction.
        self.df_features = df.drop(columns=["UserID", "ItemID", "Label"], errors='ignore')
        
        # 4. CRITICAL: Enforce Categorical Types
        # The model expects specific columns to be 'category' dtype.
        # If we pass them as int/float, the model will throw an error or give garbage results.
        cat_cols = ['User_Cluster', 'Item_Cluster', 'Cluster_Interaction']
        for col in cat_cols:
            if col in self.df_features.columns:
                self.df_features[col] = self.df_features[col].astype('category')

        # 5. Build the User Map for fast lookups
        # Groupby on the sorted dataframe to get start/end indices efficiently
        # Since data is sorted, we can use searchsorted or groupby indices
        print("Indexing users...")
        grouped = self.index_df.groupby('UserID')
        # This maps UserID -> Array of Row Indices
        self.user_map = grouped.indices 

    def recommend(self, user_ids, cutoff=20):
        batch_indices = []
        user_segment_info = []
        
        # A. Gather all row indices for this batch of users
        for uid in user_ids:
            if uid in self.user_map:
                indices = self.user_map[uid]
                batch_indices.append(indices)
                user_segment_info.append((uid, len(indices)))
            else:
                # Handle unknown users (optional: skip or raise)
                # print(f"Warning: User {uid} not found.")
                user_segment_info.append((uid, 0))

        if not batch_indices:
            return np.array([[] for _ in user_ids])

        # Flatten list of arrays into one big array for slicing
        flat_indices = np.concatenate(batch_indices)
        
        # B. Vectorized Extraction from DataFrame
        # This keeps the dataframe in RAM, but only extracts the small batch needed
        X_batch = self.df_features.iloc[flat_indices]
        items_batch = self.index_df['ItemID'].iloc[flat_indices].to_numpy()
        
        # --- THE KEY CHANGE FOR NATIVE XGBOOST ---
        # 1. Create DMatrix (No label required)
        # 2. Enable categorical support explicitly
        dtest = xgb.DMatrix(X_batch, enable_categorical=True)
        
        # 3. Predict using the Booster
        all_scores = self.XGB_model.predict(dtest)
        
        # C. Split results back into users
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
            
            # D. Rerank Logic
            if count > cutoff:
                # Optimized Top-K sorting
                # argpartition puts the top K elements at the end (unsorted)
                top_k_idx = np.argpartition(scores, -cutoff)[-cutoff:]
                # Now sort only those top K
                best_indices = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
                top_items = items[best_indices]
            else:
                # If fewer items than cutoff, just sort everything descending
                top_items = items[np.argsort(scores)[::-1]]
                
            recommendations.append(top_items.tolist())
            
        return np.array(recommendations, dtype=object)
