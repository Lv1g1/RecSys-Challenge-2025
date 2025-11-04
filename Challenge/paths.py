import os

cwd = os.getcwd()

if "/kaggle" in cwd:
    ENV = "kaggle"
    PERSISTENT_STORAGE = "/kaggle/working"

elif "/content" in cwd:
    ENV = "colab"
    PERSISTENT_STORAGE = "/content/drive/MyDrive/RecSys/demo"

else:
    ENV = "local"
    PERSISTENT_STORAGE = cwd

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
TEST_USER_IDS = os.path.join(DATA_DIR, "test_user_ids_mapped.csv")
ITEM_MAPPING = os.path.join(DATA_DIR, "item_original_ID_to_index.csv")
USER_MAPPING = os.path.join(DATA_DIR, "user_original_ID_to_index.csv")

# Split data paths
URM_TRAIN = os.path.join(DATA_DIR, "URM_train.npz")
URM_VALIDATION = os.path.join(DATA_DIR, "URM_validation.npz")

# Optuna studies storage path
OPTUNA_STORAGE = "sqlite:///" + os.path.join(PERSISTENT_STORAGE, "optuna_storage.db")