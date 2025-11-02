import os

ENV = "local"

# Detect Kaggle
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    BASE_DIR = "/kaggle/working"
    ENV = "kaggle"

# Detect Colab
elif '/content' in os.getcwd():
    BASE_DIR = "/content/drive/MyDrive/RecSys/demo"
    ENV = "colab"

else:
    BASE_DIR = os.getcwd()

# Default directories
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Running on: {ENV} — BASE_DIR = {BASE_DIR}")

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