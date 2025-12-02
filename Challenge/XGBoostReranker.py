import numpy as np
import xgboost as xgb

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
    
    
class ProgressCallback(xgb.callback.TrainingCallback):
    def __init__(self, total_trees, period=50):
        self.total_trees = total_trees
        self.period = period

    def after_iteration(self, model, epoch, evals_log):
        # XGBoost epochs are 0-indexed
        if (epoch + 1) % self.period == 0:
            print(f"\r[Training] Tree {epoch + 1}/{self.total_trees}", end="")
        
        # Return False to continue training (True would stop it)
        return False