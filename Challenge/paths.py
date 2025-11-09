import os

cwd = os.getcwd()

if "/kaggle" in cwd:
    ENV = "kaggle"
    PERSISTENT_STORAGE = "/kaggle/working"

elif "/content" in cwd:
    ENV = "colab"
    PERSISTENT_STORAGE = "/content/drive/MyDrive/RecSys"

else:
    ENV = "local"
    PERSISTENT_STORAGE = "/home/luigi/RecSys"

print(f"Running on {ENV} — storage at: {PERSISTENT_STORAGE}")

# Create necessary directories
DATA_DIR  = os.path.join(PERSISTENT_STORAGE, "data")
MODEL_DIR = os.path.join(PERSISTENT_STORAGE, "models")

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Challenge data paths
CHALLENGE_DATASET = os.path.join(DATA_DIR, "data_train.csv")
CHALLENGE_USER_IDS_TEST = os.path.join(DATA_DIR, "data_target_users_test.csv")

# Preprocessed data paths
URM_PATH = os.path.join(DATA_DIR, "URM_all.npz")

# Split data paths
URM_TRAIN = os.path.join(DATA_DIR, "URM_train.npz")
URM_VALIDATION = os.path.join(DATA_DIR, "URM_validation.npz")

# Optuna studies storage path
OPTUNA_STORAGE = "sqlite:///" + os.path.join(PERSISTENT_STORAGE, "optuna_storage.db")

# Submissions path
SUBMISSIONS = os.path.join(PERSISTENT_STORAGE, "submissions")
os.makedirs(SUBMISSIONS, exist_ok=True)

# Performance logs path
PERFORMANCE_LOG = os.path.join(PERSISTENT_STORAGE, "performance_logs")
os.makedirs(PERFORMANCE_LOG, exist_ok=True)

if ENV == "kaggle":
    CHALLENGE_DATASET = "/kaggle/input/recommender-systems-2025-challenge-polimi/data_train.csv"
    CHALLENGE_USER_IDS_TEST = "/kaggle/input/recommender-systems-2025-challenge-polimi/data_target_users_test.csv"

    URM_TRAIN = "/kaggle/input/data-recsys/URM_train.npz"
    URM_VALIDATION = "/kaggle/input/data-recsys/URM_validation.npz"

# Functions to save and load data splits
import scipy.sparse as sps

def save_holdout_split(URM_train, URM_validation):
    if ENV == "kaggle":
        return

    sps.save_npz(URM_TRAIN, URM_train)
    sps.save_npz(URM_VALIDATION, URM_validation)

def load_holdout_split():
    URM_train = sps.load_npz(URM_TRAIN)
    URM_validation = sps.load_npz(URM_VALIDATION)

    return URM_train, URM_validation

def save_cv_folds(folds):
    if ENV == "kaggle":
        return

    k = len(folds)
    dir_path = os.path.join(DATA_DIR, f"{k}_folds")
    os.makedirs(dir_path, exist_ok=True)

    for i, (URM_train, URM_validation) in enumerate(folds):
        sps.save_npz(os.path.join(dir_path, f"URM_train_fold_{i}.npz"), URM_train)
        sps.save_npz(os.path.join(dir_path, f"URM_validation_fold_{i}.npz"), URM_validation)

def load_cv_folds(k):
    folds = []
    
    dir_path = os.path.join(DATA_DIR, f"{k}_folds")
    if ENV == "kaggle":
        dir_path = "/kaggle/input/data-recsys/" + f"{k}_folds"

    for i in range(k):
        URM_train = sps.load_npz(os.path.join(dir_path, f"URM_train_fold_{i}.npz"))
        URM_validation = sps.load_npz(os.path.join(dir_path, f"URM_validation_fold_{i}.npz"))
        folds.append((URM_train, URM_validation))

    return folds